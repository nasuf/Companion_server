from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from app.services.offline import gift_repository as repo
from app.services.relationship.intimacy import get_intimacy_data


MAX_DAILY_GIFT_PROBABILITY = 0.3
NORMAL_GIFT_START_DAY = 50
FORCED_LOGIN_DAYS = {5, 20}


@dataclass(frozen=True)
class GiftTriggerDecision:
    should_trigger: bool
    trigger_type: str = "daily_probability"
    probability: float = 0.0
    cooldown_days: int | None = None
    reason: str = ""


def gift_trigger_probability(intimacy_score: int, recharge_total_yuan: float) -> float:
    raw = MAX_DAILY_GIFT_PROBABILITY * ((2 * intimacy_score + recharge_total_yuan) / 2000)
    return max(0.0, min(MAX_DAILY_GIFT_PROBABILITY, raw))


def gift_cooldown_days(probability: float) -> int:
    return max(1, round(183 - 510 * probability))


async def decide_gift_trigger(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    day: int,
    now: datetime,
    last_gift_paid_at: datetime | None,
    random_value: float | None = None,
) -> GiftTriggerDecision:
    if day in FORCED_LOGIN_DAYS and _not_paid_today(last_gift_paid_at, now):
        return GiftTriggerDecision(True, trigger_type=f"login_day_{day}", reason="forced_login_day")

    birthday = await repo.user_birthday_mmdd(user_id, workspace_id)
    if birthday and _is_five_days_before_birthday(now.date(), birthday) and _not_paid_today(
        last_gift_paid_at,
        now,
    ):
        return GiftTriggerDecision(True, trigger_type="birthday_minus_5", reason="birthday_minus_5")

    if day < NORMAL_GIFT_START_DAY:
        return GiftTriggerDecision(False, reason="before_normal_start_day")

    recharge_yuan = (await repo.recharge_total_cents(user_id)) / 100
    intimacy = await get_intimacy_data(agent_id, user_id)
    probability = gift_trigger_probability(intimacy.growth_intimacy, recharge_yuan)
    cooldown = gift_cooldown_days(probability)

    if random_value is None:
        random_value = random.random()
    if random_value >= probability:
        return GiftTriggerDecision(
            False,
            probability=probability,
            cooldown_days=cooldown,
            reason="probability_miss",
        )

    if last_gift_paid_at and now <= last_gift_paid_at + timedelta(days=cooldown):
        return GiftTriggerDecision(
            False,
            probability=probability,
            cooldown_days=cooldown,
            reason="cooldown",
        )

    return GiftTriggerDecision(
        True,
        trigger_type="daily_probability",
        probability=probability,
        cooldown_days=cooldown,
        reason="probability_hit",
    )


def _not_paid_today(last_paid_at: datetime | None, now: datetime) -> bool:
    if not last_paid_at:
        return True
    last = last_paid_at if last_paid_at.tzinfo else last_paid_at.replace(tzinfo=UTC)
    current = now if now.tzinfo else now.replace(tzinfo=UTC)
    return last.date() < current.date()


def _is_five_days_before_birthday(today: date, birthday_mmdd: tuple[int, int]) -> bool:
    month, day = birthday_mmdd
    try:
        birthday_this_year = date(today.year, month, day)
    except ValueError:
        return False
    if birthday_this_year < today:
        try:
            birthday_this_year = date(today.year + 1, month, day)
        except ValueError:
            return False
    return birthday_this_year - today == timedelta(days=5)
