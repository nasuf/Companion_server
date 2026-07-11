from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

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

    def tx(self):
        return _FakeTransaction(self)


class _FakeTransaction:
    def __init__(self, database: _FakeDb):
        self.database = database

    async def __aenter__(self):
        return self.database

    async def __aexit__(self, exc_type, exc, traceback):
        return False


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
    assert "FOR UPDATE" in fake_db.query_calls[1][0]
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


class _TransactionalWalletDb:
    def __init__(self, *, fail_ledger: bool = False):
        self.lock = asyncio.Lock()
        self.point_balance = 0
        self.achievement_points_synced = 0
        self.ledger_synced = 0
        self.fail_ledger = fail_ledger

    async def query_raw(self, query: str, *args):
        if "INSERT INTO user_wallets" in query:
            return [self._wallet_row()]
        raise AssertionError(f"Unexpected outer query: {query}")

    def tx(self):
        return _TransactionalWalletTx(self)

    def _wallet_row(self) -> dict:
        return {
            "ticket_balance": 0,
            "point_balance": self.point_balance,
            "achievement_points_synced": self.achievement_points_synced,
        }


class _TransactionalWalletTx:
    def __init__(self, owner: _TransactionalWalletDb):
        self.owner = owner
        self.snapshot: tuple[int, int, int] | None = None

    async def __aenter__(self):
        await self.owner.lock.acquire()
        self.snapshot = (
            self.owner.point_balance,
            self.owner.achievement_points_synced,
            self.owner.ledger_synced,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None and self.snapshot is not None:
            (
                self.owner.point_balance,
                self.owner.achievement_points_synced,
                self.owner.ledger_synced,
            ) = self.snapshot
        self.owner.lock.release()
        return False

    async def query_raw(self, query: str, *args):
        if "FOR UPDATE" in query:
            return [self.owner._wallet_row()]
        if "COALESCE(SUM(delta)" in query:
            return [{"synced": self.owner.ledger_synced}]
        if "UPDATE user_wallets" in query:
            delta = int(args[1])
            self.owner.point_balance += delta
            self.owner.achievement_points_synced += delta
            return [self.owner._wallet_row()]
        raise AssertionError(f"Unexpected transaction query: {query}")

    async def execute_raw(self, query: str, *args):
        if self.owner.fail_ledger:
            raise RuntimeError("ledger write failed")
        self.owner.ledger_synced += int(args[2])
        return 1


@pytest.mark.asyncio
async def test_concurrent_achievement_sync_credits_points_once(monkeypatch):
    fake_db = _TransactionalWalletDb()
    monkeypatch.setattr(wallet, "db", fake_db)
    monkeypatch.setattr(
        wallet,
        "list_achievements",
        AsyncMock(return_value={"score": 60}),
    )

    await asyncio.gather(
        wallet.sync_achievement_points("u1", "a1"),
        wallet.sync_achievement_points("u1", "a1"),
    )

    assert fake_db.point_balance == 60
    assert fake_db.achievement_points_synced == 60
    assert fake_db.ledger_synced == 60


@pytest.mark.asyncio
async def test_achievement_sync_rolls_back_wallet_when_ledger_write_fails(
    monkeypatch,
):
    fake_db = _TransactionalWalletDb(fail_ledger=True)
    monkeypatch.setattr(wallet, "db", fake_db)
    monkeypatch.setattr(
        wallet,
        "list_achievements",
        AsyncMock(return_value={"score": 60}),
    )

    with pytest.raises(RuntimeError, match="ledger write failed"):
        await wallet.sync_achievement_points("u1", "a1")

    assert fake_db.point_balance == 0
    assert fake_db.achievement_points_synced == 0
    assert fake_db.ledger_synced == 0
