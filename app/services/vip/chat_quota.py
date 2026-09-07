"""对话额度计量（CLAUDE.md 权益项 1）。

计数口径：用户发送的每条消息("句")计 1，在聚合前调用一次
（对话聚合/turn 拼接是下游概念，与本计量层无关，见实施计划 §9.5）。

**取消即不发生**：产品要求"用户确认发送后则继续发送，否则消息留存在输入
框" —— 未确认/余额不足时，本模块绝不递增 `used`、绝不扣费，调用方也不得
持久化该消息。只有 `allowed=True` 的调用才会产生副作用。

免费额度耗尽后按 :mod:`vip.config` 的单价逐句扣钞票（非 VIP 0.5，VIP 0.3）；
钱包内部以 0.1 钞票子单位存储，保证奇数句也精确结算。
"""

from __future__ import annotations

from typing import Any, Literal

from app.db import db
from app.services import wallet
from app.services.vip import config

Mode = Literal["free", "paid", "blocked"]
BlockReason = Literal["paid_confirm", "no_ticket"]

SOURCE_CHAT_OVERAGE = "chat_overage"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _can_afford(spendable: float, per_msg_cost: float) -> bool:
    return spendable + 1e-9 >= per_msg_cost


async def preview(user_id: str, *, is_vip: bool) -> dict[str, Any]:
    """Read-only quota state for the client to decide whether to confirm/block
    *before* sending — mirrors what :func:`consume_one` would decide, with no
    side effects.

    Returns a superset of what the public ``/chat/quota`` response model
    declares (``mode``/``free_remaining``/``per_msg_cost``/``spendable_tickets``)
    — the extra fields (``used``/``limit``/``period_scope``/``period_key``) only
    matter to the admin quota-status endpoint, which uses this same function to
    avoid a second copy of the period/limit lookup logic.
    """
    scope, key, limit = config.message_period(is_vip)
    rows = await db.query_raw(
        """
        SELECT used FROM user_message_quota
        WHERE user_id = $1 AND period_scope = $2 AND period_key = $3
        """,
        user_id,
        scope,
        key,
    )
    used = int(_field(rows[0], "used", 0) or 0) if rows else 0
    free_remaining = max(0, limit - used)
    per_msg_cost = config.overage_per_msg(is_vip)

    wallet_snapshot = await wallet.full_wallet(user_id)
    spendable = float(wallet_snapshot["spendable_tickets"])

    if free_remaining > 0:
        mode: Mode = "free"
    elif _can_afford(spendable, per_msg_cost):
        mode = "paid"
    else:
        mode = "blocked"

    return {
        "mode": mode,
        "free_remaining": free_remaining,
        "per_msg_cost": per_msg_cost,
        "spendable_tickets": spendable,
        "used": used,
        "limit": limit,
        "period_scope": scope,
        "period_key": key,
    }


async def admin_reset(user_id: str, *, is_vip: bool) -> dict[str, Any]:
    """Zero out the user's *current* period usage (免费/VIP用户重置对话额度).

    Only resets the counter for whichever (scope, key) is active right now for
    this user's VIP status — it does not touch fractional ticket balances.
    """
    scope, key, _ = config.message_period(is_vip)
    await db.execute_raw(
        """
        UPDATE user_message_quota
        SET used = 0, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1 AND period_scope = $2 AND period_key = $3
        """,
        user_id,
        scope,
        key,
    )
    return await preview(user_id, is_vip=is_vip)


async def consume_one(
    user_id: str, *, is_vip: bool, paid_confirmed: bool = False
) -> dict[str, Any]:
    """Count one user message against quota; charge tickets if over quota.

    Returns ``{"allowed": bool, ...}``. When ``allowed`` is False the caller
    (ws intake) must reject the message outright — nothing was counted or
    charged, matching "取消发送 = 什么都没发生".
    """
    scope, key, limit = config.message_period(is_vip)
    per_msg_cost = config.overage_per_msg(is_vip)

    async with db.tx() as tx:
        await tx.execute_raw(
            """
            INSERT INTO user_message_quota (user_id, period_scope, period_key, used)
            VALUES ($1, $2, $3, 0)
            ON CONFLICT (user_id, period_scope, period_key) DO NOTHING
            """,
            user_id,
            scope,
            key,
        )
        locked = await tx.query_raw(
            """
            SELECT used FROM user_message_quota
            WHERE user_id = $1 AND period_scope = $2 AND period_key = $3
            FOR UPDATE
            """,
            user_id,
            scope,
            key,
        )
        used = int(_field(locked[0], "used", 0) or 0)

        if used < limit:
            await tx.execute_raw(
                """
                UPDATE user_message_quota
                SET used = used + 1, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = $1 AND period_scope = $2 AND period_key = $3
                """,
                user_id,
                scope,
                key,
            )
            return {"allowed": True, "mode": "free", "used": used + 1, "limit": limit, "charged": 0}

        snapshot = await wallet.full_wallet(user_id, client=tx)
        spendable = float(snapshot["spendable_tickets"])

        if not paid_confirmed:
            if _can_afford(spendable, per_msg_cost):
                reason: BlockReason = "paid_confirm"
                mode: Mode = "paid"
            else:
                reason = "no_ticket"
                mode = "blocked"
            return {
                "allowed": False,
                "mode": mode,
                "reason": reason,
                "per_msg_cost": per_msg_cost,
                "spendable_tickets": spendable,
            }

        await wallet.ensure_wallet(user_id, client=tx)
        try:
            await wallet.debit_tickets_prioritized(
                user_id,
                per_msg_cost,
                source=SOURCE_CHAT_OVERAGE,
                metadata={
                    "per_msg_cost": per_msg_cost,
                    "is_vip": is_vip,
                },
                client=tx,
            )
        except ValueError:
            return {
                "allowed": False,
                "mode": "blocked",
                "reason": "no_ticket",
                "per_msg_cost": per_msg_cost,
                "spendable_tickets": 0,
            }

        await tx.execute_raw(
            """
            UPDATE user_message_quota
            SET used = used + 1, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1 AND period_scope = $2 AND period_key = $3
            """,
            user_id,
            scope,
            key,
        )
        return {
            "allowed": True,
            "mode": "paid",
            "used": used + 1,
            "limit": limit,
            "charged": per_msg_cost,
        }
