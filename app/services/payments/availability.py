"""Uncached global gates for initiating real-money purchases, never settlement.

Apple in-app purchase and Service Account WeChat Pay have separate switches so
one channel can be open while the other is closed. The Apple switch keeps its
historical name ``real_money_payments_enabled``: released Flutter builds read it.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException

from app.db import db

logger = logging.getLogger(__name__)

# API field -> Prisma field.
_FIELDS = {
    "real_money_payments_enabled": "realMoneyPaymentsEnabled",
    "wechat_pay_purchases_enabled": "wechatPayPurchasesEnabled",
    "android_auto_renew_enabled": "androidAutoRenewEnabled",
}


async def get_payment_settings() -> dict[str, bool]:
    try:
        row = await db.systemconfig.find_unique(where={"id": 1})
    except Exception as exc:
        logger.exception("payment availability read failed")
        raise HTTPException(503, "payment_settings_unavailable") from exc
    return {key: bool(getattr(row, field, False)) for key, field in _FIELDS.items()}


async def get_apple_purchase_settings() -> dict[str, bool]:
    settings = await get_payment_settings()
    return {"real_money_payments_enabled": settings["real_money_payments_enabled"]}


async def get_android_store_settings() -> dict[str, bool]:
    """安卓商城要的两个开关，一次读完：能否走服务号 H5 付款、是否展示连续包月。"""
    settings = await get_payment_settings()
    return {
        "wechat_pay_purchases_enabled": settings["wechat_pay_purchases_enabled"],
        "android_auto_renew_enabled": settings["android_auto_renew_enabled"],
    }


async def require_real_money_payments() -> dict[str, bool]:
    """Apple in-app purchase gate."""
    settings = await get_apple_purchase_settings()
    if not settings["real_money_payments_enabled"]:
        raise HTTPException(403, "real_money_payments_disabled")
    return settings


async def wechat_pay_purchases_enabled() -> bool:
    return (await get_payment_settings())["wechat_pay_purchases_enabled"]


async def require_wechat_pay_purchases() -> None:
    # Same error code as the Apple gate: the H5 store already shows it as
    # "支付暂未开放", and the user does not care which switch is closed.
    if not await wechat_pay_purchases_enabled():
        raise HTTPException(403, "real_money_payments_disabled")


async def set_payment_settings(*, actor_id: str, **changes: bool) -> dict[str, bool]:
    """Write only the switches that were passed; the others keep their value."""
    unknown = set(changes) - set(_FIELDS)
    if unknown or not changes:
        raise ValueError(f"invalid payment settings: {sorted(unknown) or 'empty'}")
    data = {_FIELDS[key]: value for key, value in changes.items()}
    await db.systemconfig.upsert(
        where={"id": 1},
        data={"create": {"id": 1, **data}, "update": data},
    )
    for key, value in changes.items():
        logger.info(
            "real-money payment gate updated",
            extra={
                "event": "real_money_payment_gate_updated",
                "kind": key,
                "enabled": value,
                "actor_id": actor_id,
            },
        )
    return await get_payment_settings()
