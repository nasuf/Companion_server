"""VIP 每月发放 + 到期清零（CLAUDE.md 权益项 3/4/6）。

发放周期用 ``vip_last_grant_at + VIP_GRANT_PERIOD_DAYS`` 锚点而非自然月：
本项目不含真实订阅支付，VIP 主要来自 30 天体验或后台手动发放，用自然月
会让跨月的体验期在月初重复发放一次。定时任务见 ``jobs/scheduler.py``。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import db
from app.services import wallet
from app.services.store_catalog import MAKEUP_CARD_KIND, MUSIC_COUPON_KIND
from app.services.store_inventory import add_batch
from app.services.vip import config

SOURCE_VIP_MONTHLY_GRANT = "vip_monthly_grant"
SOURCE_VIP_EXPIRE_CLEAR = "vip_expire_clear"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


async def due_for_monthly_grant(*, limit: int = 500) -> list[str]:
    """User ids whose VIP is active and whose grant anchor is due (or unset)."""
    rows = await db.query_raw(
        """
        SELECT user_id
        FROM user_wallets
        WHERE vip_until IS NOT NULL
          AND vip_until > CURRENT_TIMESTAMP
          AND (
              vip_last_grant_at IS NULL
              OR vip_last_grant_at <= CURRENT_TIMESTAMP - ($1 || ' days')::interval
          )
        ORDER BY updated_at ASC
        LIMIT $2
        """,
        config.VIP_GRANT_PERIOD_DAYS,
        limit,
    )
    return [str(_field(row, "user_id", "")) for row in rows if _field(row, "user_id")]


async def due_for_expire_clear(*, limit: int = 500) -> list[str]:
    """User ids whose VIP lapsed but still hold gift tickets or vip_grant batches."""
    rows = await db.query_raw(
        """
        SELECT DISTINCT w.user_id
        FROM user_wallets w
        WHERE (w.vip_until IS NULL OR w.vip_until <= CURRENT_TIMESTAMP)
          AND (
              w.gift_ticket_balance > 0
              OR EXISTS (
                  SELECT 1 FROM user_consumable_batch b
                  WHERE b.user_id = w.user_id
                    AND b.source = 'vip_grant'
                    AND b.quantity > 0
              )
          )
        LIMIT $1
        """,
        limit,
    )
    return [str(_field(row, "user_id", "")) for row in rows if _field(row, "user_id")]


async def grant_monthly(user_id: str) -> dict[str, Any]:
    """Grant one VIP monthly bundle: gift tickets + music coupons + makeup cards."""
    await wallet.ensure_wallet(user_id)
    expires_at = datetime.now(timezone.utc) + timedelta(days=config.VIP_GIFT_VALID_DAYS)
    async with db.tx() as tx:
        balance = await wallet.credit_gift_tickets(
            user_id,
            config.VIP_MONTHLY_GIFT_TICKETS,
            source=SOURCE_VIP_MONTHLY_GRANT,
            metadata={"kind": "gift_ticket"},
            client=tx,
        )
        music_batch = await add_batch(
            user_id,
            MUSIC_COUPON_KIND,
            quantity=config.VIP_MONTHLY_MUSIC_COUPONS,
            source="vip_grant",
            expires_at=expires_at,
            client=tx,
        )
        makeup_batch = await add_batch(
            user_id,
            MAKEUP_CARD_KIND,
            quantity=config.VIP_MONTHLY_MAKEUP_CARDS,
            source="vip_grant",
            expires_at=expires_at,
            client=tx,
        )
        await tx.execute_raw(
            """
            UPDATE user_wallets
            SET vip_last_grant_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
            """,
            user_id,
        )
    return {
        "wallet": balance,
        "music_coupon_batch": music_batch,
        "makeup_card_batch": makeup_batch,
    }


async def clear_on_lapse(user_id: str) -> dict[str, Any]:
    """VIP 过期即清零限时钞票 + 失效所有 vip_grant 消耗品批次（不结转）。"""
    async with db.tx() as tx:
        balance = await wallet.zero_gift_tickets(
            user_id,
            source=SOURCE_VIP_EXPIRE_CLEAR,
            metadata={"reason": "vip_lapsed"},
            client=tx,
        )
        cleared = await tx.execute_raw(
            """
            UPDATE user_consumable_batch
            SET quantity = 0
            WHERE user_id = $1 AND source = 'vip_grant' AND quantity > 0
            """,
            user_id,
        )
    return {"wallet": balance, "cleared_batches": cleared}
