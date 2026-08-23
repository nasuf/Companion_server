"""Shop pack purchases: music coupons (tickets), game points (tickets), VIP trial."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from app.db import db
from app.services import game_points, wallet
from app.services.store_catalog import (
    MAKEUP_CARD_KIND,
    MUSIC_COUPON_KIND,
    MUSIC_COUPON_VALID_DAYS,
    VIP_TRIAL_DAYS,
    game_tier,
    makeup_tier,
    music_tier,
)
from app.services.store_inventory import add_inventory


async def purchase_bundle(
    user_id: str,
    bundle_kind: str,
    *,
    tier_id: str | None = None,
) -> dict[str, Any]:
    if bundle_kind == "music_coupon":
        return await _buy_music(user_id, tier_id)
    if bundle_kind == "game_points":
        return await _buy_game(user_id, tier_id)
    if bundle_kind == "makeup_card":
        return await _buy_makeup(user_id, tier_id)
    if bundle_kind == "vip_trial":
        raise ValueError("payment_required")
    raise ValueError("unknown_bundle")


async def activate_vip_trial(user_id: str) -> dict[str, Any]:
    """Grant the once-per-account 30-day VIP trial after payment clears."""
    await wallet.ensure_wallet(user_id)
    async with db.tx() as tx:
        rows = await tx.query_raw(
            """
            SELECT vip_until, vip_trial_used,
                   ticket_balance, point_balance, achievement_points_synced
            FROM user_wallets
            WHERE user_id = $1
            FOR UPDATE
            """,
            user_id,
        )
        row = rows[0]
        if not wallet.vip_trial_available_from_row(row):
            raise ValueError("vip_trial_used")
        until = datetime.now(timezone.utc) + timedelta(days=VIP_TRIAL_DAYS)
        # TIMESTAMP WITHOUT TIME ZONE stores the wall clock as-is; keep UTC
        # numbers so session TimeZone cannot shift the 30-day window.
        until_store = until.replace(tzinfo=None)
        updated = await tx.query_raw(
            """
            UPDATE user_wallets
            SET vip_until = $2,
                vip_trial_used = TRUE,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            RETURNING ticket_balance, point_balance, achievement_points_synced,
                      vip_until, vip_trial_used
            """,
            user_id,
            until_store,
        )
        balance = wallet.wallet_balances(updated[0])
    return {
        "wallet": balance,
        "vip_until": until.isoformat(),
        "inventory_item": None,
        "game_balance": None,
    }


async def _buy_music(user_id: str, tier_id: str | None) -> dict[str, Any]:
    tier = music_tier(tier_id or "")
    if tier is None:
        raise ValueError("unknown_tier")
    await wallet.ensure_wallet(user_id)
    async with db.tx() as tx:
        balance = await wallet.debit_tickets(
            user_id,
            tier.ticket_price,
            source="store_bundle",
            source_id=MUSIC_COUPON_KIND,
            metadata={
                "bundle_kind": "music_coupon",
                "tier_id": tier.tier_id,
                "grant_amount": tier.grant_amount,
                "valid_days": MUSIC_COUPON_VALID_DAYS,
            },
            client=tx,
        )
        inventory_item = await add_inventory(
            user_id,
            MUSIC_COUPON_KIND,
            quantity=tier.grant_amount,
            client=tx,
        )
    return {
        "wallet": balance,
        "inventory_item": inventory_item,
        "game_balance": None,
        "vip_until": None,
    }


async def _buy_makeup(user_id: str, tier_id: str | None) -> dict[str, Any]:
    tier = makeup_tier(tier_id or "")
    if tier is None:
        raise ValueError("unknown_tier")
    await wallet.ensure_wallet(user_id)
    async with db.tx() as tx:
        balance = await wallet.debit_tickets(
            user_id,
            tier.ticket_price,
            source="store_bundle",
            source_id=MAKEUP_CARD_KIND,
            metadata={
                "bundle_kind": "makeup_card",
                "tier_id": tier.tier_id,
                "grant_amount": tier.grant_amount,
            },
            client=tx,
        )
        inventory_item = await add_inventory(
            user_id,
            MAKEUP_CARD_KIND,
            quantity=tier.grant_amount,
            client=tx,
        )
    return {
        "wallet": balance,
        "inventory_item": inventory_item,
        "game_balance": None,
        "vip_until": None,
    }


async def _buy_game(user_id: str, tier_id: str | None) -> dict[str, Any]:
    tier = game_tier(tier_id or "")
    if tier is None:
        raise ValueError("unknown_tier")
    await wallet.ensure_wallet(user_id)
    await game_points.ensure_wallet(user_id)
    async with db.tx() as tx:
        balance = await wallet.debit_tickets(
            user_id,
            tier.ticket_price,
            source="store_bundle",
            source_id="game_points",
            metadata={
                "bundle_kind": "game_points",
                "tier_id": tier.tier_id,
                "grant_amount": tier.grant_amount,
            },
            client=tx,
        )
        credited = await game_points.credit_from_store(
            user_id,
            tier.grant_amount,
            # Partial unique index game_point_ledger_source_key on
            # (user_id, source, source_id) WHERE source_id IS NOT NULL —
            # a stable per-tier id would block buying the same pack twice.
            source_id=f"store_game:{tier.tier_id}:{uuid4()}",
            metadata={
                "bundle_kind": "game_points",
                "tier_id": tier.tier_id,
                "ticket_price": tier.ticket_price,
            },
            client=tx,
        )
    return {
        "wallet": balance,
        "inventory_item": None,
        "game_balance": credited["balance"],
        "vip_until": None,
    }
