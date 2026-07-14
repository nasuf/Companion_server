"""霸王餐 public endpoints: 用户动态二维码 / 服务员校验 / 商家核销.

Merchant self-service auth: identity = merchant dropdown choice + contact
name/phone exact match, exchanged for a short-lived JWT with
``role=meal_merchant`` (separate from user tokens; ``sub`` is the merchant id).
"""

from __future__ import annotations

import hmac
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
_STAFF_ROLE = "meal_staff"
_STAFF_SUBJECT = "meal_staff"


class StaffLoginRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64)


class MerchantLoginRequest(BaseModel):
    merchant_id: str = Field(min_length=1, max_length=64)
    contact: str = Field(min_length=1, max_length=64)


class MealScanRequest(BaseModel):
    value: str = Field(min_length=16, max_length=256)


def _voucher_payload(voucher, merchant_name: str | None = None) -> dict:
    expires_at = mv.voucher_expires_at(voucher)
    return {
        "status": voucher.status,
        "activated_at": voucher.activatedAt.isoformat() if voucher.activatedAt else None,
        "redeemed_at": voucher.redeemedAt.isoformat() if voucher.redeemedAt else None,
        "merchant_name": merchant_name,
        # 有效期: 服务员校验后 N 天. 前端据此展示截止时间并标出过期态.
        "expires_at": expires_at.isoformat() if expires_at else None,
        "expired": mv.is_voucher_expired(voucher),
        "validity_days": settings.meal_validity_days,
    }


async def _merchant_name_of(voucher) -> str | None:
    if not getattr(voucher, "merchantId", None):
        return None
    merchant = await db.mealmerchant.find_unique(where={"id": voucher.merchantId})
    return merchant.name if merchant else None


# ── 用户: 券状态 / 两阶段动态二维码 ───────────────────────────────────


@router.get("/voucher")
async def get_voucher(payload: dict = Depends(require_user)):
    voucher = await mv.get_or_create_voucher(payload["sub"])
    return {
        **_voucher_payload(voucher, await _merchant_name_of(voucher)),
        "code_enabled": await mv.is_code_enabled(),
    }


@router.post("/voucher/qr-token", dependencies=[Depends(require_redis)])
async def voucher_qr_token(payload: dict = Depends(require_user)):
    """Issue the QR for the voucher's current staff/merchant stage."""
    voucher = await mv.get_or_create_voucher(payload["sub"])
    if voucher.status == mv.VOUCHER_REDEEMED:
        raise HTTPException(status_code=400, detail="该券已核销")
    if voucher.status == mv.VOUCHER_INACTIVE:
        if not await mv.is_code_enabled():
            raise HTTPException(status_code=400, detail="服务员扫码校验功能暂未开放")
        action = "activate"
    elif voucher.status == mv.VOUCHER_ACTIVATED:
        action = "redeem"
    else:
        raise HTTPException(status_code=400, detail="霸王餐券状态异常")
    if mv.is_voucher_expired(voucher):
        raise HTTPException(
            status_code=400,
            detail={"message": "霸王餐券已过有效期，无法生成核销码", "reason": "expired"},
        )
    return await meal_qr.issue(voucher.id, voucher.userId, action)


# ── 服务员: 登录 / 扫码校验 ──────────────────────────────────────────


def _require_scoped_role(request: Request, role: str, login_message: str) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail=login_message)
    try:
        payload = decode_jwt(auth[7:].strip())
    except Exception:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    if payload.get("role") != role:
        raise HTTPException(status_code=403, detail="无访问权限")
    return str(payload["sub"])


def _require_staff(request: Request) -> str:
    return _require_scoped_role(request, _STAFF_ROLE, "请先登录服务员入口")


@router.post("/staff/login")
async def staff_login(data: StaffLoginRequest, request: Request):
    """Exchange the configured staff key for a short-lived scoped JWT."""
    await enforce_login_rate_limit(request, "mealstaff")
    expected = settings.meal_staff_key.strip()
    candidate = data.key.strip().upper()
    if not expected and settings.is_production():
        logger.error("MEAL_STAFF_KEY is missing in production")
        raise HTTPException(status_code=503, detail="服务员入口尚未配置")
    if expected and not hmac.compare_digest(candidate, expected.upper()):
        await record_login_failure(request, "mealstaff")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="服务员口令错误"
        )
    token = create_jwt(
        _STAFF_SUBJECT,
        _STAFF_ROLE,
        expiry_hours=settings.meal_staff_jwt_expiry_hours,
    )
    logger.info("meal staff logged in", extra={"event": "meal_staff_login"})
    return {"token": token}


@router.get("/staff/jssdk-config", dependencies=[Depends(require_redis)])
async def staff_jssdk_config(
    request: Request,
    url: str = Query(min_length=8, max_length=2048),
):
    """Sign the exact staff H5 URL for wx.scanQRCode."""
    _require_staff(request)
    try:
        return await wechat_jssdk.build_config(url)
    except wechat_jssdk.WeChatJSSDKError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("failed to build staff WeChat JS-SDK config")
        raise HTTPException(status_code=503, detail="微信扫一扫初始化失败，请稍后重试")


@router.post("/staff/activate-scan", dependencies=[Depends(require_redis)])
async def staff_activate_scan(data: MealScanRequest, request: Request):
    """Consume a customer's validation QR and mark the voucher as validated."""
    _require_staff(request)
    if not await mv.is_code_enabled():
        raise HTTPException(
            status_code=403,
            detail={"message": "服务员扫码校验功能已关闭", "reason": "disabled"},
        )
    try:
        grant = await meal_qr.consume(data.value, "activate")
        voucher = await mv.activate_voucher_by_staff(
            grant["voucher_id"], grant["user_id"]
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
    return {
        "voucher_id": voucher.id,
        "user_display": await mv.resolve_user_display(voucher.userId),
        "activated_at": voucher.activatedAt.isoformat() if voucher.activatedAt else None,
    }


# ── 商家: 下拉列表 / 身份确认 / 核销码 / 核销统计 ─────────────────────


def _require_merchant(request: Request) -> str:
    return _require_scoped_role(request, _MERCHANT_ROLE, "请先确认商家身份")


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
async def merchant_redeem_scan(data: MealScanRequest, request: Request):
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
        grant = await meal_qr.consume(data.value, "redeem")
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
