"""Admin API: 霸王餐管理 (校验码管理 + 商家管理).

Endpoints (all admin-only):
  GET    /admin-api/meal/overview        — 开关状态 + 当前码 + 倒计时 + 累计数据
  GET    /admin-api/meal/activations     — 实时校验动态 (轮询)
  GET    /admin-api/meal/stats           — 日期范围统计 (UTC+8 按天聚合激活/核销)
  GET    /admin-api/meal/expired         — 已过期券明细 (激活超 N 天未核销 + 用户)
  GET    /admin-api/meal/redemption-failures — 指定日核销失败明细 (超上限 + 用户)
  PUT    /admin-api/meal/code-enabled    — 开启/关闭校验码功能
  GET    /admin-api/meal/merchants       — 商家列表 + 各自核销数
  POST   /admin-api/meal/merchants       — 新增商家
  PUT    /admin-api/meal/merchants/{id}  — 修改商家资料/停用/开启扫码核销
  DELETE /admin-api/meal/merchants/{id}  — 删除商家 (已核销记录保留, merchant 置空)
  GET    /admin-api/meal/merchants/{id}/redemptions — 该商家核销用户明细
  DELETE /admin-api/meal/vouchers/{id}/redemption   — 清除核销 (回到已激活)
  DELETE /admin-api/meal/vouchers/{id}/activation   — 清除校验 (整券归零)
"""

from __future__ import annotations

import logging
from datetime import date as date_cls
from datetime import datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services import meal_voucher as mv

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/meal",
    tags=["admin", "meal"],
    dependencies=[Depends(require_admin_jwt)],
)

class CodeEnabledRequest(BaseModel):
    enabled: bool


class MerchantCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    contact_name: str | None = Field(default=None, max_length=64)
    contact_phone: str | None = Field(default=None, max_length=32)


class MerchantUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    contact_name: str | None = Field(default=None, max_length=64)
    contact_phone: str | None = Field(default=None, max_length=32)
    code_active: bool | None = None


def _merchant_payload(merchant, redeemed_count: int) -> dict:
    return {
        "id": merchant.id,
        "name": merchant.name,
        "contact_name": merchant.contactName,
        "contact_phone": merchant.contactPhone,
        "code_active": merchant.codeActive,
        "redeemed_count": redeemed_count,
        "created_at": merchant.createdAt.isoformat() if merchant.createdAt else None,
    }


@router.get("/overview")
async def overview():
    enabled = await mv.is_code_enabled()
    stats = await mv.voucher_stats()
    body: dict = {"enabled": enabled, **stats}
    if enabled:
        code, expires_in = await mv.activation_code_now()
        body.update(
            code=code, expires_in=expires_in, window_seconds=mv.CODE_WINDOW_SECONDS
        )
    else:
        body.update(code=None, expires_in=None, window_seconds=mv.CODE_WINDOW_SECONDS)
    return body


@router.get("/activations")
async def activations(limit: int = 50):
    return await mv.activation_feed(limit=min(max(limit, 1), 200))


@router.put("/code-enabled")
async def set_code_enabled(data: CodeEnabledRequest):
    await mv.set_code_enabled(data.enabled)
    return {"enabled": data.enabled}


# 业务口径固定 UTC+8 自然日 (与项目时间系统一致).
_CN_OFFSET = timedelta(hours=8)
_MAX_RANGE_DAYS = 366

_DAY_COUNT_SQL = """
    SELECT to_char({column} + INTERVAL '8 hours', 'YYYY-MM-DD') AS day,
           COUNT(*)::int AS cnt
    FROM meal_vouchers
    WHERE {column} >= $1::timestamp AND {column} < $2::timestamp
    GROUP BY 1
"""


async def _daily_counts(column: str, utc_start: str, utc_end: str) -> dict[str, int]:
    # column comes from a fixed internal whitelist — never user input.
    rows = await db.query_raw(
        _DAY_COUNT_SQL.format(column=column), utc_start, utc_end
    )
    return {row["day"]: int(row["cnt"]) for row in rows or []}


@router.get("/stats")
async def range_stats(start: str, end: str):
    """按天统计激活/核销数 (UTC+8 自然日, 两端含). 缺数据的日期补零."""
    try:
        start_d = date_cls.fromisoformat(start)
        end_d = date_cls.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式不正确 (YYYY-MM-DD)")
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="开始日期不能晚于结束日期")
    if (end_d - start_d).days >= _MAX_RANGE_DAYS:
        raise HTTPException(status_code=400, detail="时间跨度不能超过一年")

    # CN 自然日边界换算成存储侧的 UTC naive timestamp.
    utc_start = (datetime.combine(start_d, time()) - _CN_OFFSET).isoformat(sep=" ")
    utc_end = (
        datetime.combine(end_d + timedelta(days=1), time()) - _CN_OFFSET
    ).isoformat(sep=" ")

    activated = await _daily_counts("activated_at", utc_start, utc_end)
    redeemed = await _daily_counts("redeemed_at", utc_start, utc_end)

    days = []
    cursor = start_d
    while cursor <= end_d:
        key = cursor.isoformat()
        days.append(
            {
                "date": key,
                "activated": activated.get(key, 0),
                "redeemed": redeemed.get(key, 0),
            }
        )
        cursor += timedelta(days=1)

    return {
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "activated_total": sum(activated.values()),
        "redeemed_total": sum(redeemed.values()),
        "days": days,
    }


@router.get("/expired")
async def expired_vouchers(limit: int = 100):
    """已过期券明细: 激活后超过有效期 (默认 7 天) 仍未核销的券 + 用户.

    活动开始不足有效期天数时返回空 — 数据要满 N 天才会出现 (spec 需求 3).
    """
    return await mv.expired_vouchers_feed(limit=min(max(limit, 1), 500))


@router.get("/redemption-failures")
async def redemption_failures(date: str | None = None, limit: int = 200):
    """指定自然日 (默认今天, UTC+8) 因当日核销上限被拒的失败明细 + 用户.

    每用户每日去重 → 返回行数即当日「没抢到」的人数.
    """
    day: date_cls | None = None
    if date:
        try:
            day = date_cls.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式不正确 (YYYY-MM-DD)")
    items = await mv.redemption_failures_feed(day=day, limit=min(max(limit, 1), 500))
    resolved = day or datetime.now(mv._CN_TZ).date()
    return {"date": resolved.isoformat(), "total": len(items), "items": items}


async def _redeemed_counts() -> dict[str, int]:
    rows = await db.query_raw(
        """
        SELECT merchant_id, COUNT(*)::int AS cnt
        FROM meal_vouchers
        WHERE status = 'redeemed' AND merchant_id IS NOT NULL
        GROUP BY merchant_id
        """
    )
    return {row["merchant_id"]: int(row["cnt"]) for row in rows or []}


@router.get("/merchants")
async def list_merchants():
    merchants = await db.mealmerchant.find_many(order={"createdAt": "asc"})
    counts = await _redeemed_counts()
    return [_merchant_payload(m, counts.get(m.id, 0)) for m in merchants]


@router.post("/merchants")
async def create_merchant(data: MerchantCreateRequest):
    name = data.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="商家名称不能为空")
    merchant = await db.mealmerchant.create(
        data={
            "name": name,
            "contactName": (data.contact_name or "").strip() or None,
            "contactPhone": (data.contact_phone or "").strip() or None,
        }
    )
    logger.info(
        "meal merchant created",
        extra={"event": "meal_merchant_created", "merchant_id": merchant.id},
    )
    return _merchant_payload(merchant, 0)


@router.put("/merchants/{merchant_id}")
async def update_merchant(merchant_id: str, data: MerchantUpdateRequest):
    merchant = await db.mealmerchant.find_unique(where={"id": merchant_id})
    if not merchant:
        raise HTTPException(status_code=404, detail="商家不存在")

    updates: dict = {}
    if data.name is not None:
        name = data.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="商家名称不能为空")
        updates["name"] = name
    if data.contact_name is not None:
        updates["contactName"] = data.contact_name.strip() or None
    if data.contact_phone is not None:
        updates["contactPhone"] = data.contact_phone.strip() or None
    if data.code_active is not None:
        updates["codeActive"] = data.code_active

    if not updates:
        counts = await _redeemed_counts()
        return _merchant_payload(merchant, counts.get(merchant_id, 0))

    updated = await db.mealmerchant.update(where={"id": merchant_id}, data=updates)
    counts = await _redeemed_counts()
    logger.info(
        "meal merchant updated",
        extra={"event": "meal_merchant_updated", "merchant_id": merchant_id},
    )
    return _merchant_payload(updated, counts.get(merchant_id, 0))


@router.get("/merchants/{merchant_id}/redemptions")
async def merchant_redemptions(merchant_id: str, limit: int = 100):
    """该商家的核销用户明细 (微信昵称/脱敏手机号 + 核销时间, 新→旧)."""
    merchant = await db.mealmerchant.find_unique(where={"id": merchant_id})
    if not merchant:
        raise HTTPException(status_code=404, detail="商家不存在")
    rows = await db.mealvoucher.find_many(
        where={"merchantId": merchant_id, "status": mv.VOUCHER_REDEEMED},
        order={"redeemedAt": "desc"},
        take=min(max(limit, 1), 500),
    )
    profiles = await mv.resolve_user_redemption_profiles([row.userId for row in rows])
    total = await db.mealvoucher.count(
        where={"merchantId": merchant_id, "status": mv.VOUCHER_REDEEMED}
    )
    return {
        "merchant_name": merchant.name,
        "total": total,
        "items": [
            {
                "voucher_id": row.id,
                **profiles.get(
                    row.userId,
                    {
                        "user_id": row.userId,
                        "username": row.userId,
                        "user_display": row.userId,
                        "phone_masked": None,
                        "wechat_nickname": None,
                        "wechat_avatar_url": None,
                        "wechat_openid": None,
                        "wechat_unionid": None,
                    },
                ),
                "activated_at": (
                    row.activatedAt.isoformat() if row.activatedAt else None
                ),
                "expires_at": (
                    expires.isoformat()
                    if (expires := mv.voucher_expires_at(row))
                    else None
                ),
                "redeemed_at": row.redeemedAt.isoformat() if row.redeemedAt else None,
            }
            for row in rows
        ],
    }


@router.delete("/vouchers/{voucher_id}/redemption")
async def clear_redemption(voucher_id: str, payload: dict = Depends(require_admin_jwt)):
    """清除核销记录: 券回到「已激活」, 可在其他商家重新核销."""
    voucher = await db.mealvoucher.find_unique(where={"id": voucher_id})
    if not voucher:
        raise HTTPException(status_code=404, detail="记录不存在")
    try:
        await mv.clear_redemption(voucher_id)
    except mv.MealVoucherError as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    logger.info(
        "admin cleared redemption",
        extra={
            "event": "meal_admin_clear_redemption",
            "voucher_id": voucher_id,
            "admin_id": payload.get("sub"),
        },
    )
    return {"ok": True}


@router.delete("/vouchers/{voucher_id}/activation")
async def clear_activation(voucher_id: str, payload: dict = Depends(require_admin_jwt)):
    """清除校验记录: 整券归零到「未激活」(如已核销一并清除), 用户可重新激活."""
    voucher = await db.mealvoucher.find_unique(where={"id": voucher_id})
    if not voucher:
        raise HTTPException(status_code=404, detail="记录不存在")
    try:
        await mv.clear_activation(voucher_id)
    except mv.MealVoucherError as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    logger.info(
        "admin cleared activation",
        extra={
            "event": "meal_admin_clear_activation",
            "voucher_id": voucher_id,
            "admin_id": payload.get("sub"),
        },
    )
    return {"ok": True}


@router.delete("/merchants/{merchant_id}")
async def delete_merchant(merchant_id: str):
    merchant = await db.mealmerchant.find_unique(where={"id": merchant_id})
    if not merchant:
        raise HTTPException(status_code=404, detail="商家不存在")
    await db.mealmerchant.delete(where={"id": merchant_id})
    logger.info(
        "meal merchant deleted",
        extra={"event": "meal_merchant_deleted", "merchant_id": merchant_id},
    )
    return {"ok": True}
