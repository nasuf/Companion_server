from __future__ import annotations

from app.services.offline import gift_repository as repo


async def available_gift_budget_cents(user_id: str) -> int:
    recharge_total = await repo.recharge_total_cents(user_id)
    historical_spend = await repo.historical_gift_spend_cents(user_id)
    return max(0, int(recharge_total * 0.8) - historical_spend)
