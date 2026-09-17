"""VIP entitlement end computation: IAP (paid) + activation codes (deferred stack)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import db
from app.services.payments import catalog
from app.services.vip import grants as vip_grants

logger = logging.getLogger(__name__)

REDEMPTION_GRANTED = "granted"
REDEMPTION_REVOKED = "revoked"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def naive_dt(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=None) if dt is not None else None


async def compute_paid_vip_end(user_id: str, *, client: Any | None = None) -> datetime | None:
    """Subscription + consumable IAP stack (excludes activation codes)."""
    consumable_end = await _consumable_vip_floor(user_id, client=client)
    sub_end = await _subscription_vip_end(user_id, client=client)
    ends = [x for x in (consumable_end, sub_end) if x is not None]
    return max(ends) if ends else None


async def compute_activation_code_vip_end(
    user_id: str, *, client: Any | None = None, paid_end: datetime | None = None
) -> datetime | None:
    """Stack granted activation-code days after paid IAP entitlement."""
    executor = client or db
    rows = await executor.query_raw(
        """
        SELECT duration_days, redeemed_at
        FROM vip_code_redemptions
        WHERE user_id = $1 AND status = $2
        ORDER BY redeemed_at ASC, id ASC
        """,
        user_id,
        REDEMPTION_GRANTED,
    )
    if not rows:
        return None

    now = datetime.now(timezone.utc)
    if paid_end is None:
        paid_end = await compute_paid_vip_end(user_id, client=client)
    anchor = paid_end if paid_end is not None and paid_end > now else now
    running = anchor
    for row in rows:
        days = int(_field(row, "duration_days") or 0)
        if days <= 0:
            continue
        running = running + timedelta(days=days)
    return running


async def compute_vip_entitlement_end(
    user_id: str, *, client: Any | None = None
) -> datetime | None:
    """Single source of truth: max(paid IAP end, activation-code stack end)."""
    paid_end = await compute_paid_vip_end(user_id, client=client)
    code_end = await compute_activation_code_vip_end(
        user_id, client=client, paid_end=paid_end
    )
    ends = [x for x in (paid_end, code_end) if x is not None]
    return max(ends) if ends else None


async def recompute_vip_entitlements(
    user_id: str,
    *,
    client: Any | None = None,
    clear_lapse: bool = True,
) -> bool:
    """Recompute ``vip_until`` from IAP + activation codes (can raise or lower)."""
    now = datetime.now(timezone.utc)
    entitlement_end = await compute_vip_entitlement_end(user_id, client=client)
    if entitlement_end is None or entitlement_end <= now:
        new_until = now - timedelta(seconds=1)
        lapsed = True
    else:
        new_until = entitlement_end
        lapsed = False

    executor = client or db
    rows = await executor.query_raw(
        "SELECT vip_until FROM user_wallets WHERE user_id = $1 FOR UPDATE",
        user_id,
    )
    if not rows:
        return False
    current = as_utc(_field(rows[0], "vip_until"))
    if current is not None and naive_dt(current) == naive_dt(new_until):
        return False

    await executor.execute_raw(
        """
        UPDATE user_wallets
        SET vip_until = $2::timestamp, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        """,
        user_id,
        naive_dt(new_until),
    )
    logger.info(
        "vip recompute user=%s until=%s lapsed=%s",
        user_id[:8],
        new_until.isoformat(),
        lapsed,
    )
    if lapsed and clear_lapse and client is None:
        try:
            await vip_grants.clear_on_lapse(user_id)
        except Exception:
            logger.exception("clear_on_lapse after recompute failed user=%s", user_id[:8])
    return True


async def reconcile_vip_entitlements(user_id: str, *, client: Any | None = None) -> bool:
    """Backward-compatible alias."""
    return await recompute_vip_entitlements(user_id, client=client)


async def preview_code_segment_end(
    user_id: str,
    *,
    duration_days: int,
    client: Any | None = None,
) -> tuple[datetime, datetime]:
    """Predict effective [start, end) for a new redemption (audit snapshot)."""
    paid_end = await compute_paid_vip_end(user_id, client=client)
    code_end = await compute_activation_code_vip_end(
        user_id, client=client, paid_end=paid_end
    )
    now = datetime.now(timezone.utc)
    anchor = paid_end if paid_end is not None and paid_end > now else now
    if code_end is not None and code_end > anchor:
        start = code_end
    else:
        start = anchor
    end = start + timedelta(days=duration_days)
    return start, end


async def _subscription_vip_end(user_id: str, *, client: Any | None = None) -> datetime | None:
    now = datetime.now(timezone.utc)
    executor = client or db
    txn_rows = await executor.query_raw(
        """
        SELECT MAX(expires_date) AS max_expires
        FROM iap_transactions
        WHERE user_id = $1 AND kind = $2 AND status = 'granted'
          AND expires_date IS NOT NULL
        """,
        user_id,
        catalog.KIND_SUBSCRIPTION,
    )
    txn_end = as_utc(_field(txn_rows[0], "max_expires")) if txn_rows else None

    state_rows = await executor.query_raw(
        """
        SELECT expires_date
        FROM iap_subscription_state
        WHERE user_id = $1 AND status = ANY($2::text[])
          AND expires_date IS NOT NULL
        """,
        user_id,
        ["active", "in_grace"],
    )
    state_ends = [
        end
        for end in (as_utc(_field(row, "expires_date")) for row in state_rows)
        if end is not None
    ]
    state_end = max(state_ends) if state_ends else None

    candidates = [end for end in (txn_end, state_end) if end is not None and end > now]
    return max(candidates) if candidates else None


async def _consumable_vip_floor(user_id: str, *, client: Any | None = None) -> datetime | None:
    executor = client or db
    rows = await executor.query_raw(
        """
        SELECT product_id, quantity, purchase_date
        FROM iap_transactions
        WHERE user_id = $1 AND status = 'granted' AND kind = $2
        ORDER BY purchase_date ASC NULLS LAST, created_at ASC
        """,
        user_id,
        catalog.KIND_CONSUMABLE,
    )
    running: datetime | None = None
    for row in rows:
        product = catalog.product_for(_field(row, "product_id") or "")
        if product is None or product.vip_days <= 0:
            continue
        qty = int(_field(row, "quantity") or 1)
        purchased = as_utc(_field(row, "purchase_date"))
        if purchased is None:
            continue
        base = max(purchased, running) if running is not None else purchased
        running = base + timedelta(days=product.vip_days * qty)
    return running
