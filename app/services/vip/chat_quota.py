"""对话额度计量（CLAUDE.md 权益项 1）。

计数口径：用户发送的每条消息("句")计 1，在聚合前调用一次
（对话聚合/turn 拼接是下游概念，与本计量层无关，见实施计划 §9.5）。

**取消即不发生**：产品要求"用户确认发送后则继续发送，否则消息留存在输入
框" —— 未确认/余额不足时，本模块绝不递增 `used`、绝不扣费，调用方也不得
持久化该消息。只有 `allowed=True` 的调用才会产生副作用。

免费额度耗尽后按 :mod:`vip.config` 的单价累加小数钞票成本；累计满 1 时
整数扣费（先扣限时赠送钞票，再扣永久钞票），账本始终是整数，累加器只
存在 `user_wallets.overage_accrued` 这一处小数状态。
"""

from __future__ import annotations

import math
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


async def preview(user_id: str, *, is_vip: bool) -> dict[str, Any]:
    """Read-only quota state for the client to decide whether to confirm/block
    *before* sending — mirrors what :func:`consume_one` would decide, with no
    side effects.
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
    spendable = wallet_snapshot["spendable_tickets"]

    mode: Mode = "free" if free_remaining > 0 else ("paid" if spendable > 0 else "blocked")

    return {
        "mode": mode,
        "free_remaining": free_remaining,
        "per_msg_cost": per_msg_cost,
        "spendable_tickets": spendable,
    }


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

        if not paid_confirmed:
            snapshot = await wallet.full_wallet(user_id)
            spendable = snapshot["spendable_tickets"]
            reason: BlockReason = "paid_confirm" if spendable > 0 else "no_ticket"
            return {
                "allowed": False,
                "mode": "paid" if reason == "paid_confirm" else "blocked",
                "reason": reason,
                "per_msg_cost": per_msg_cost,
                "spendable_tickets": spendable,
            }

        await wallet.ensure_wallet(user_id)
        wallet_locked = await tx.query_raw(
            "SELECT overage_accrued FROM user_wallets WHERE user_id = $1 FOR UPDATE",
            user_id,
        )
        accrued = float(_field(wallet_locked[0], "overage_accrued", 0) or 0) + per_msg_cost
        whole = math.floor(accrued)
        remainder = round(accrued - whole, 2)

        if whole > 0:
            try:
                await wallet.debit_tickets_prioritized(
                    user_id,
                    whole,
                    source=SOURCE_CHAT_OVERAGE,
                    metadata={"per_msg_cost": per_msg_cost, "is_vip": is_vip},
                    client=tx,
                )
            except ValueError:
                # Balance changed between preview and confirm (race/spend
                # elsewhere) — reject without writing the accrual or the count.
                return {
                    "allowed": False,
                    "mode": "blocked",
                    "reason": "no_ticket",
                    "per_msg_cost": per_msg_cost,
                    "spendable_tickets": 0,
                }

        await tx.execute_raw(
            "UPDATE user_wallets SET overage_accrued = $2, updated_at = CURRENT_TIMESTAMP WHERE user_id = $1",
            user_id,
            remainder,
        )
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
        return {"allowed": True, "mode": "paid", "used": used + 1, "limit": limit, "charged": whole}
