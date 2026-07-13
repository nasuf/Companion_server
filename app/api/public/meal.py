"""霸王餐 public endpoints: 用户券操作 / 服务员轮换码 / 商家自助核销台.

Merchant self-service auth: identity = merchant dropdown choice + contact
name/phone exact match, exchanged for a short-lived JWT with
``role=meal_merchant`` (separate from user tokens; ``sub`` is the merchant id).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.api.deps import require_redis
from app.api.jwt_auth import require_user
from app.config import settings
from app.db import db
from app.services import meal_qr
from app.services import meal_voucher as mv
from app.services import wechat_jssdk
from app.services.auth import create_jwt, decode_jwt
from app.services.auth_security import (
    enforce_login_rate_limit,
    record_login_failure,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meal", tags=["meal"])

_MERCHANT_ROLE = "meal_merchant"


class VoucherCodeRequest(BaseModel):
    code: str = Field(min_length=4, max_length=8)


class MerchantLoginRequest(BaseModel):
    merchant_id: str = Field(min_length=1, max_length=64)
    contact: str = Field(min_length=1, max_length=64)


class MerchantScanRequest(BaseModel):
    value: str = Field(min_length=16, max_length=256)


def _voucher_payload(voucher, merchant_name: str | None = None) -> dict:
    expires_at = mv.voucher_expires_at(voucher)
    return {
        "status": voucher.status,
        "activated_at": voucher.activatedAt.isoformat() if voucher.activatedAt else None,
        "redeemed_at": voucher.redeemedAt.isoformat() if voucher.redeemedAt else None,
        "merchant_name": merchant_name,
        # 有效期: 激活后 N 天. 前端据此在个人页展示「有效期至…」并标出过期态.
        "expires_at": expires_at.isoformat() if expires_at else None,
        "expired": mv.is_voucher_expired(voucher),
        "validity_days": settings.meal_validity_days,
    }


async def _merchant_name_of(voucher) -> str | None:
    if not getattr(voucher, "merchantId", None):
        return None
    merchant = await db.mealmerchant.find_unique(where={"id": voucher.merchantId})
    return merchant.name if merchant else None


# ── 用户: 券状态 / 激活 / 核销 ────────────────────────────────────────


@router.get("/voucher")
async def get_voucher(payload: dict = Depends(require_user)):
    voucher = await mv.get_or_create_voucher(payload["sub"])
    return {
        **_voucher_payload(voucher, await _merchant_name_of(voucher)),
        "code_enabled": await mv.is_code_enabled(),
    }


@router.post("/voucher/activate")
async def activate_voucher(
    data: VoucherCodeRequest, request: Request, payload: dict = Depends(require_user)
):
    user_id = payload["sub"]
    # 6 位码空间只有 1e6, 必须限暴力尝试 (IP+user 5 次/15min).
    await enforce_login_rate_limit(request, f"mealact:{user_id}")
    try:
        voucher = await mv.activate_voucher(user_id, data.code)
    except mv.MealVoucherError as exc:
        if exc.reason == "bad_code":
            await record_login_failure(request, f"mealact:{user_id}")
        raise HTTPException(status_code=400, detail=exc.message)
    return _voucher_payload(voucher)


@router.post("/voucher/qr-token", dependencies=[Depends(require_redis)])
async def voucher_qr_token(payload: dict = Depends(require_user)):
    """Issue the user's only active, short-lived QR redemption grant."""
    voucher = await mv.get_or_create_voucher(payload["sub"])
    if voucher.status == mv.VOUCHER_REDEEMED:
        raise HTTPException(status_code=400, detail="该券已核销")
    if voucher.status != mv.VOUCHER_ACTIVATED:
        raise HTTPException(status_code=400, detail="请先激活霸王餐券")
    if mv.is_voucher_expired(voucher):
        raise HTTPException(
            status_code=400,
            detail={"message": "霸王餐券已过有效期，无法生成核销码", "reason": "expired"},
        )
    return await meal_qr.issue(voucher.id, voucher.userId)


# ── 服务员: 轮换校验码展示页 ──────────────────────────────────────────


@router.get("/staff/code")
async def staff_code(key: str = ""):
    """5 分钟轮换校验码. MEAL_STAFF_KEY 配置后必须携带匹配的 ?key=."""
    expected = settings.meal_staff_key.strip()
    if expected and key.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="无访问权限"
        )
    if not await mv.is_code_enabled():
        return {"enabled": False, "code": None, "expires_in": None}
    code, expires_in = await mv.activation_code_now()
    return {
        "enabled": True,
        "code": code,
        "expires_in": expires_in,
        "window_seconds": mv.CODE_WINDOW_SECONDS,
    }


# ── 商家: 下拉列表 / 身份确认 / 核销码 / 核销统计 ─────────────────────


def _require_merchant(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="请先确认商家身份")
    try:
        payload = decode_jwt(auth[7:].strip())
    except Exception:
        raise HTTPException(status_code=401, detail="登录已过期，请重新确认身份")
    if payload.get("role") != _MERCHANT_ROLE:
        raise HTTPException(status_code=403, detail="无访问权限")
    return str(payload["sub"])


@router.get("/merchants")
async def list_merchants_public():
    """商家 H5 下拉框: 只暴露 id+名称, 联系方式不出网."""
    merchants = await db.mealmerchant.find_many(order={"createdAt": "asc"})
    return [{"id": m.id, "name": m.name} for m in merchants]


@router.post("/merchant/login")
async def merchant_login(data: MerchantLoginRequest, request: Request):
    await enforce_login_rate_limit(request, f"mealmch:{data.merchant_id}")
    merchant = await db.mealmerchant.find_unique(where={"id": data.merchant_id})
    if not merchant or not mv.merchant_contact_matches(merchant, data.contact):
        await record_login_failure(request, f"mealmch:{data.merchant_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="商家信息不匹配，请核对联系人姓名或手机号",
        )
    token = create_jwt(
        merchant.id,
        _MERCHANT_ROLE,
        expiry_hours=settings.meal_merchant_jwt_expiry_hours,
    )
    logger.info(
        "meal merchant logged in",
        extra={"event": "meal_merchant_login", "merchant_id": merchant.id},
    )
    return {"token": token, "merchant": {"id": merchant.id, "name": merchant.name}}


@router.get(
    "/merchant/jssdk-config",
    dependencies=[Depends(require_redis)],
)
async def merchant_jssdk_config(
    request: Request,
    url: str = Query(min_length=8, max_length=2048),
):
    """Sign the exact merchant H5 URL for wx.scanQRCode."""
    _require_merchant(request)
    try:
        return await wechat_jssdk.build_config(url)
    except wechat_jssdk.WeChatJSSDKError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("failed to build WeChat JS-SDK config")
        raise HTTPException(status_code=503, detail="微信扫一扫初始化失败，请稍后重试")


@router.post(
    "/merchant/redeem-scan",
    dependencies=[Depends(require_redis)],
)
async def merchant_redeem_scan(data: MerchantScanRequest, request: Request):
    """Consume a customer QR grant and redeem it as the authenticated merchant."""
    merchant_id = _require_merchant(request)
    # Validate merchant availability before consuming the customer's one-time QR.
    merchant = await db.mealmerchant.find_unique(where={"id": merchant_id})
    if not merchant or not merchant.codeActive:
        raise HTTPException(
            status_code=403,
            detail={"message": "商家核销功能已停用", "reason": "merchant_disabled"},
        )
    try:
        grant = await meal_qr.consume(data.value)
        voucher = await mv.redeem_voucher_by_merchant(
            grant["voucher_id"], grant["user_id"], merchant_id
        )
    except meal_qr.MealQRError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": exc.message, "reason": exc.reason},
        )
    except mv.MealVoucherError as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": exc.message, "reason": exc.reason},
        )
    logger.info(
        "meal voucher redeemed by merchant QR scan",
        extra={
            "event": "meal_voucher_qr_redeemed",
            "voucher_id": voucher.id,
            "user_id": voucher.userId,
            "merchant_id": merchant_id,
        },
    )
    return {
        "voucher_id": voucher.id,
        "user_display": await mv.resolve_user_display(voucher.userId),
        "merchant_name": merchant.name,
        "redeemed_at": voucher.redeemedAt.isoformat() if voucher.redeemedAt else None,
    }


@router.get("/merchant/stats")
async def merchant_stats(request: Request):
    merchant_id = _require_merchant(request)
    merchant = await db.mealmerchant.find_unique(where={"id": merchant_id})
    if not merchant:
        raise HTTPException(status_code=404, detail="商家不存在")
    redeemed = await db.mealvoucher.find_many(
        where={"merchantId": merchant_id, "status": mv.VOUCHER_REDEEMED},
        order={"redeemedAt": "desc"},
        take=50,
    )
    displays = await mv.resolve_user_displays([row.userId for row in redeemed])
    recent = [
        {
            "user_display": displays.get(row.userId, row.userId),
            "redeemed_at": row.redeemedAt.isoformat() if row.redeemedAt else None,
        }
        for row in redeemed
    ]
    total = await db.mealvoucher.count(
        where={"merchantId": merchant_id, "status": mv.VOUCHER_REDEEMED}
    )
    return {"merchant_name": merchant.name, "redeemed_total": total, "recent": recent}
