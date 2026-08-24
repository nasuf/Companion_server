"""音乐陪伴时长计量（CLAUDE.md 权益项 6）。

消耗优先级（客户端每 ~15s 上报一次增量秒数）：
    每日免费 0.5h → 消耗音乐畅听券（`user_consumable_batch` 最早过期先扣，
    1 张=1 小时=3600s）→ 钞票（VIP 5/非VIP 10，每 0.5h）。
    购买的券对 VIP/非 VIP 都在免费额度后生效（否则非 VIP 无法用自己买的
    券），见实施计划 §9 "待确认" 第 2 条。

优惠桶单位不同（券按小时、钞票按半小时），消耗时各自按自己的整数单位
"银行式"预支：一次报量可能购入超出本次实际所需的时长，多出的部分记入
`provisioned_seconds` 供下次上报直接扣减，不再重复购买/扣费。

返回的 ``action`` 供客户端决定弹哪个框：
    none            —— 本次上报被现有额度/已购覆盖完全吸收，继续播放
    confirm_ticket   —— 需要弹"是否消耗钞票继续听"，确认后客户端重发同一段
                        delta 并带 paid_confirmed=true
    buy_coupon       —— (VIP) 钞票也不足，弹"购买音乐畅听券"
    buy_vip          —— (非VIP) 钞票不足，弹"订阅 VIP"
"""

from __future__ import annotations

import math
from typing import Any, Literal

from app.db import db
from app.services import wallet
from app.services.store_catalog import MUSIC_COUPON_KIND
from app.services.store_inventory import consume_batch_units
from app.services.vip import config

Action = Literal["none", "confirm_ticket", "buy_coupon", "buy_vip"]

SOURCE_MUSIC_OVERAGE = "music_overage"


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


async def report(
    user_id: str,
    *,
    is_vip: bool,
    delta_seconds: int,
    paid_confirmed: bool = False,
) -> dict[str, Any]:
    if delta_seconds <= 0:
        raise ValueError("invalid_amount")
    day = config.day_key()
    # 首次触达钞票余额前必须确保钱包行存在 —— 否则一个从未碰过钱包的用户
    # (纯听歌、还没聊过天/买过东西) 第一次超额就会在 debit_tickets_prioritized
    # 里因 "SELECT ... FOR UPDATE" 查不到行而抛 wallet_not_found, 被下面
    # 宽泛的 except ValueError 悄悄吞成 "余额不足"。跟 chat_quota.consume_one
    # 的确认分支保持同样的防御, 不能只有一处有。
    await wallet.ensure_wallet(user_id)

    async with db.tx() as tx:
        rows = await tx.query_raw(
            """
            INSERT INTO user_music_quota (user_id, day_key)
            VALUES ($1, $2)
            ON CONFLICT (user_id, day_key) DO UPDATE
            SET updated_at = user_music_quota.updated_at
            RETURNING listened_seconds, provisioned_seconds, coupon_units, ticket_spent
            """,
            user_id,
            day,
        )
        row = rows[0]
        listened = int(_field(row, "listened_seconds", 0) or 0)
        provisioned = int(_field(row, "provisioned_seconds", 0) or 0)
        coupon_units = int(_field(row, "coupon_units", 0) or 0)
        ticket_spent = int(_field(row, "ticket_spent", 0) or 0)

        # 1) drain today's free allowance
        remaining = delta_seconds
        free_used = min(remaining, max(0, config.FREE_DAILY_MUSIC_SECONDS - listened))
        remaining -= free_used

        # 2) drain any already-purchased/charged seconds banked from a prior
        #    report in the same day (whole-unit purchases can overshoot need).
        covered_used = min(remaining, provisioned)
        remaining -= covered_used

        new_coupon_units = 0
        new_ticket_charge = 0
        provisioned_gain = 0

        if remaining > 0:
            coupons_needed = math.ceil(remaining / config.MUSIC_COUPON_UNIT_SECONDS)
            try:
                await consume_batch_units(
                    user_id, MUSIC_COUPON_KIND, coupons_needed, client=tx
                )
                new_coupon_units = coupons_needed
                provisioned_gain += coupons_needed * config.MUSIC_COUPON_UNIT_SECONDS
                remaining = 0
            except ValueError:
                pass  # fall through to tickets

        if remaining > 0:
            per_half_hour = config.music_ticket_per_half_hour(is_vip)
            half_hours_needed = math.ceil(remaining / config.MUSIC_HALF_HOUR_SECONDS)
            ticket_cost = half_hours_needed * per_half_hour

            if not paid_confirmed:
                snapshot = await wallet.full_wallet(user_id)
                action: Action = (
                    "confirm_ticket"
                    if snapshot["spendable_tickets"] >= ticket_cost
                    else ("buy_coupon" if is_vip else "buy_vip")
                )
                await _persist(
                    tx,
                    user_id,
                    day,
                    listened=listened + free_used + covered_used,
                    provisioned=provisioned - covered_used + provisioned_gain,
                    coupon_units=coupon_units + new_coupon_units,
                    ticket_spent=ticket_spent,
                )
                return {
                    "action": action,
                    "accepted_seconds": free_used + covered_used,
                    "pending_seconds": remaining,
                    "ticket_cost": ticket_cost,
                }

            try:
                await wallet.debit_tickets_prioritized(
                    user_id,
                    ticket_cost,
                    source=SOURCE_MUSIC_OVERAGE,
                    metadata={"per_half_hour": per_half_hour, "is_vip": is_vip},
                    client=tx,
                )
                new_ticket_charge = ticket_cost
                provisioned_gain += half_hours_needed * config.MUSIC_HALF_HOUR_SECONDS
                remaining = 0
            except ValueError:
                action = "buy_coupon" if is_vip else "buy_vip"
                await _persist(
                    tx,
                    user_id,
                    day,
                    listened=listened + free_used + covered_used,
                    provisioned=provisioned - covered_used + provisioned_gain,
                    coupon_units=coupon_units + new_coupon_units,
                    ticket_spent=ticket_spent,
                )
                return {
                    "action": action,
                    "accepted_seconds": free_used + covered_used,
                    "pending_seconds": remaining,
                    "ticket_cost": ticket_cost,
                }

        listened_final = listened + delta_seconds
        provisioned_final = provisioned - covered_used + provisioned_gain
        await _persist(
            tx,
            user_id,
            day,
            listened=listened_final,
            provisioned=provisioned_final,
            coupon_units=coupon_units + new_coupon_units,
            ticket_spent=ticket_spent + new_ticket_charge,
        )
        return {
            "action": "none",
            "accepted_seconds": delta_seconds,
            "pending_seconds": 0,
            "ticket_cost": 0,
        }


async def _persist(
    tx: Any,
    user_id: str,
    day: str,
    *,
    listened: int,
    provisioned: int,
    coupon_units: int,
    ticket_spent: int,
) -> None:
    await tx.execute_raw(
        """
        UPDATE user_music_quota
        SET listened_seconds = $2,
            provisioned_seconds = $3,
            coupon_units = $4,
            ticket_spent = $5,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1 AND day_key = $6
        """,
        user_id,
        listened,
        provisioned,
        coupon_units,
        ticket_spent,
        day,
    )
