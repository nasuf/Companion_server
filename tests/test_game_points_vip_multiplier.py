from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.services import game_points


class _FakeSettleDb:
    def __init__(self, *, rules: dict[str, Any], balance: int = 0, lifetime_earned: int = 0):
        self.rules = rules
        self.balance = balance
        self.lifetime_earned = lifetime_earned
        self.ledger_calls: list[tuple] = []

    async def query_raw(self, query: str, *args):
        if "SELECT rules FROM game_point_rules" in query:
            return [{"rules": self.rules}]
        if "SELECT balance FROM user_game_wallets" in query:
            return [{"balance": self.balance}]
        raise AssertionError(f"unexpected query_raw: {query}")

    async def execute_raw(self, query: str, *args):
        if "INSERT INTO user_game_wallets" in query:
            return 1
        if "UPDATE user_game_wallets" in query:
            self.balance = args[1]
            self.lifetime_earned += args[2]
            return 1
        if "INSERT INTO game_point_ledger" in query:
            self.ledger_calls.append(args)
            return 1
        raise AssertionError(f"unexpected execute_raw: {query}")


def _session(*, outcome: str, final_payload: dict | None = None) -> SimpleNamespace:
    result: dict[str, Any] = {"user_outcome": outcome}
    if final_payload is not None:
        result["final_payload"] = final_payload
    return SimpleNamespace(game_key="tetris", user_id="u1", id="session-1", result=result)


@pytest.mark.asyncio
async def test_settle_session_applies_vip_multiplier_to_positive_outcome():
    db = _FakeSettleDb(rules={"type": "outcome", "win": 4, "lose": -3, "draw": 0, "quit": -3})
    session = _session(outcome="win")

    await game_points.settle_session(session, database=db, is_vip=True)

    # 4 * 1.5 = 6
    assert db.balance == 6
    assert db.lifetime_earned == 6
    metadata = db.ledger_calls[0][-1]
    import json

    payload = json.loads(metadata)
    assert payload["earned"] == 6
    assert payload["intended_delta"] == 4
    assert payload["vip_multiplier"] == 1.5


@pytest.mark.asyncio
async def test_settle_session_does_not_amplify_a_loss_for_vip():
    db = _FakeSettleDb(
        rules={"type": "outcome", "win": 4, "lose": -3, "draw": 0, "quit": -3},
        balance=10,
    )
    session = _session(outcome="lose")

    await game_points.settle_session(session, database=db, is_vip=True)

    # Losing is never made worse by VIP status -- multiplier only applies
    # to positive settlements.
    assert db.balance == 7
    assert db.lifetime_earned == 0
    metadata = db.ledger_calls[0][-1]
    import json

    payload = json.loads(metadata)
    assert "vip_multiplier" not in payload
    assert payload["intended_delta"] == -3


@pytest.mark.asyncio
async def test_settle_session_non_vip_earns_base_amount():
    db = _FakeSettleDb(rules={"type": "outcome", "win": 4, "lose": -3, "draw": 0, "quit": -3})
    session = _session(outcome="win")

    await game_points.settle_session(session, database=db, is_vip=False)

    assert db.balance == 4
    assert db.lifetime_earned == 4


@pytest.mark.asyncio
async def test_settle_session_milestone_multiplier_rounds_to_nearest_int():
    rules = {
        "type": "milestone",
        "milestones": [{"tile": 128, "points": 3}],
        "quit_below_threshold": {"threshold": 128, "below": -2, "at_or_above": 0},
    }
    db = _FakeSettleDb(rules=rules)
    session = _session(outcome="win", final_payload={"max_tile": 128})

    await game_points.settle_session(session, database=db, is_vip=True)

    # 3 * 1.5 = 4.5 -> round() to 4 (banker's rounding on .5 ties to even)
    assert db.balance == 4
