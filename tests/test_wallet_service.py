from __future__ import annotations

import pytest

from app.services import wallet


class _FakeDb:
    def __init__(self, rows_by_query: list[list[dict]]):
        self.rows_by_query = rows_by_query
        self.query_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []

    async def query_raw(self, query: str, *args):
        self.query_calls.append((query, args))
        return self.rows_by_query.pop(0)

    async def execute_raw(self, query: str, *args):
        self.execute_calls.append((query, args))
        return 1


@pytest.mark.asyncio
async def test_sync_achievement_points_only_adds_new_delta(monkeypatch):
    fake_db = _FakeDb(
        [
            [
                {
                    "ticket_balance": 3,
                    "point_balance": 20,
                    "achievement_points_synced": 10,
                }
            ],
            [{"synced": 10}],
            [{"point_balance": 70, "achievement_points_synced": 60}],
        ]
    )
    monkeypatch.setattr(wallet, "db", fake_db)

    async def fake_list_achievements(*, user_id: str, agent_id: str):
        assert user_id == "u1"
        assert agent_id == "a1"
        return {"score": 60}

    monkeypatch.setattr(wallet, "list_achievements", fake_list_achievements)

    balance = await wallet.sync_achievement_points("u1", "a1")

    assert balance == {
        "ticket_balance": 3,
        "point_balance": 70,
        "achievement_points_synced": 60,
    }
    assert len(fake_db.execute_calls) == 1
    _, args = fake_db.execute_calls[0]
    assert args[1:5] == ("point", 50, 70, "achievement_sync")


@pytest.mark.asyncio
async def test_exchange_ticket_to_points_updates_both_balances_and_ledgers(monkeypatch):
    fake_db = _FakeDb(
        [
            [
                {
                    "ticket_balance": 12,
                    "point_balance": 40,
                    "achievement_points_synced": 30,
                }
            ],
            [{"ticket_balance": 7, "point_balance": 90}],
            [
                {
                    "ticket_balance": 7,
                    "point_balance": 90,
                    "achievement_points_synced": 30,
                }
            ],
        ]
    )
    monkeypatch.setattr(wallet, "db", fake_db)

    balance = await wallet.exchange_ticket_to_points("u1", ticket_amount=5)

    assert balance == {
        "ticket_balance": 7,
        "point_balance": 90,
        "achievement_points_synced": 30,
    }
    assert len(fake_db.execute_calls) == 2
    assert fake_db.execute_calls[0][1][1:5] == (
        "ticket",
        -5,
        7,
        "ticket_to_point_exchange",
    )
    assert fake_db.execute_calls[1][1][1:5] == (
        "point",
        50,
        90,
        "ticket_to_point_exchange",
    )


@pytest.mark.asyncio
async def test_exchange_ticket_to_points_rejects_insufficient_balance(monkeypatch):
    fake_db = _FakeDb(
        [
            [
                {
                    "ticket_balance": 2,
                    "point_balance": 40,
                    "achievement_points_synced": 0,
                }
            ],
            [],
        ]
    )
    monkeypatch.setattr(wallet, "db", fake_db)

    with pytest.raises(ValueError, match="insufficient_ticket_balance"):
        await wallet.exchange_ticket_to_points("u1", ticket_amount=5)

    assert fake_db.execute_calls == []
