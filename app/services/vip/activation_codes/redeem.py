from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.observability.events import EVT_VIP_CODE_REDEEM
from app.services import wallet
from app.services.runtime.tasks import fire_background
from app.services.vip import grants as vip_grants
from app.services.vip.activation_codes.errors import VipActivationError
from app.services.vip.activation_codes.normalize import normalize_code
from app.services.vip.activation_codes import repo
from app.services.vip.entitlements import preview_code_segment_end, recompute_vip_entitlements

logger = logging.getLogger(__name__)


def _code_is_valid_window(code: dict, now: datetime) -> bool:
    vf = code.get("valid_from")
    vu = code.get("valid_until")
    if vf is not None:
        start = vf if isinstance(vf, datetime) else datetime.fromisoformat(str(vf))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if now < start:
            return False
    if vu is not None:
        end = vu if isinstance(vu, datetime) else datetime.fromisoformat(str(vu))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if now > end:
            return False
    return True


def _remaining_slots(code: dict) -> int | None:
    cap = code.get("max_redemptions")
    if cap is None:
        return None
    return int(cap) - int(code.get("redemption_count") or 0)


async def redeem_code(user_id: str, raw_code: str) -> dict[str, Any]:
    """Redeem an activation code for the user; stack after paid VIP entitlement."""
    try:
        normalized = normalize_code(raw_code)
    except ValueError as exc:
        raise VipActivationError("invalid_code", "激活码格式无效") from exc

    await wallet.ensure_wallet(user_id)
    was_vip = await wallet.is_vip(user_id)

    redemption_id: str | None = None
    code_id: str | None = None
    duration_days = 0
    effective_start: datetime | None = None
    effective_end: datetime | None = None

    async with db.tx() as tx:
        code = await repo.fetch_code_by_normalized(normalized, client=tx)
        if code is None:
            raise VipActivationError("invalid_code", "激活码不存在")

        locked = await repo.fetch_code_for_update(code["id"], client=tx)
        if locked is None:
            raise VipActivationError("invalid_code", "激活码不存在")

        now = datetime.now(timezone.utc)
        if not locked.get("enabled", True):
            raise VipActivationError("code_disabled", "激活码已停用")
        if not _code_is_valid_window(locked, now):
            raise VipActivationError("code_expired", "激活码不在有效期内")

        remaining = _remaining_slots(locked)
        if remaining is not None and remaining <= 0:
            raise VipActivationError("code_exhausted", "激活码已被领完")

        if await repo.user_has_granted_redemption(locked["id"], user_id, client=tx):
            raise VipActivationError("already_redeemed", "您已兑换过该激活码")

        duration_days = int(locked["duration_days"])
        code_id = locked["id"]
        effective_start, effective_end = await preview_code_segment_end(
            user_id, duration_days=duration_days, client=tx
        )

        redemption_id = await repo.insert_redemption(
            code_id=locked["id"],
            user_id=user_id,
            duration_days=duration_days,
            effective_start=effective_start,
            effective_end=effective_end,
            client=tx,
        )
        await repo.increment_redemption_count(locked["id"], client=tx)
        await recompute_vip_entitlements(user_id, client=tx, clear_lapse=False)

    if not was_vip:
        fire_background(_grant_monthly_safe(user_id))

    snapshot = await wallet.full_wallet(user_id)
    logger.info(
        "vip code redeemed user=%s code=%s days=%s",
        user_id[:8],
        normalized[:8],
        duration_days,
        extra={
            "event": EVT_VIP_CODE_REDEEM,
            "code_id": code_id,
            "redemption_id": redemption_id,
            "duration_days": duration_days,
        },
    )
    return {
        "vip": snapshot,
        "redemption": {
            "id": redemption_id,
            "code_id": code_id,
            "duration_days": duration_days,
            "redeemed_at": datetime.now(timezone.utc).isoformat(),
            "effective_start": effective_start.isoformat(),
            "effective_end": effective_end.isoformat(),
        },
    }


async def _grant_monthly_safe(user_id: str) -> None:
    try:
        await vip_grants.grant_monthly(user_id)
    except Exception:
        logger.exception("vip code monthly grant failed user=%s", user_id[:8])
