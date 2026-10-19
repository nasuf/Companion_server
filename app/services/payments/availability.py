"""Uncached global gate for initiating real-money purchases, never settlement."""
from __future__ import annotations

import logging

from fastapi import HTTPException

from app.db import db

logger = logging.getLogger(__name__)


async def get_payment_settings() -> dict[str, bool]:
    try:
        row = await db.systemconfig.find_unique(where={"id": 1})
    except Exception as exc:
        logger.exception("payment availability read failed")
        raise HTTPException(503, "payment_settings_unavailable") from exc
    return {
        "real_money_payments_enabled": bool(
            getattr(row, "realMoneyPaymentsEnabled", False)
        )
    }


async def require_real_money_payments() -> dict[str, bool]:
    settings = await get_payment_settings()
    if not settings["real_money_payments_enabled"]:
        raise HTTPException(403, "real_money_payments_disabled")
    return settings


async def set_payment_settings(*, enabled: bool, actor_id: str) -> dict[str, bool]:
    await db.systemconfig.upsert(
        where={"id": 1},
        data={
            "create": {"id": 1, "realMoneyPaymentsEnabled": enabled},
            "update": {"realMoneyPaymentsEnabled": enabled},
        },
    )
    logger.info(
        "real-money payment gate updated",
        extra={
            "event": "real_money_payment_gate_updated",
            "enabled": enabled,
            "actor_id": actor_id,
        },
    )
    return await get_payment_settings()
