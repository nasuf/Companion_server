from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.services import store_inventory


class _FakeTx:
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


class _TxContext:
    def __init__(self, tx: _FakeTx):
        self.tx = tx

    async def __aenter__(self):
        return self.tx

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeDb:
    def __init__(self, *, list_rows=None, tx_rows=None):
        self.list_rows = list_rows or []
        self.fake_tx = _FakeTx(tx_rows or [])
        self.query_calls: list[tuple[str, tuple]] = []

    async def query_raw(self, query: str, *args):
        self.query_calls.append((query, args))
        return self.list_rows

    def tx(self):
        return _TxContext(self.fake_tx)


@pytest.mark.asyncio
async def test_list_inventory_returns_owned_items(monkeypatch):
    acquired_at = datetime(2026, 7, 1, tzinfo=UTC)
    fake_db = _FakeDb(
        list_rows=[
            {
                "product_kind": "tea",
                "quantity": 2,
                "acquired_at": acquired_at,
                "updated_at": acquired_at,
            }
        ]
    )
    monkeypatch.setattr(store_inventory, "db", fake_db)

    result = await store_inventory.list_inventory("user-1")

    assert result["items"] == [
        {
            "product_kind": "tea",
            "quantity": 2,
            "acquired_at": acquired_at.isoformat(),
            "updated_at": acquired_at.isoformat(),
        }
    ]
    assert fake_db.query_calls[0][1] == ("user-1",)


@pytest.mark.asyncio
async def test_exchange_product_spends_points_and_upserts_inventory(monkeypatch):
    updated_at = datetime(2026, 7, 1, tzinfo=UTC)
    fake_db = _FakeDb(
        tx_rows=[
            [
                {
                    "ticket_balance": 0,
                    "point_balance": 901,
                    "achievement_points_synced": 0,
                }
            ],
            [
                {
                    "product_kind": "tea",
                    "quantity": 1,
                    "acquired_at": updated_at,
                    "updated_at": updated_at,
                }
            ],
        ]
    )
    monkeypatch.setattr(store_inventory, "db", fake_db)

    async def fake_ensure_wallet(user_id: str):
        assert user_id == "user-1"
        return {
            "ticket_balance": 0,
            "point_balance": 1000,
            "achievement_points_synced": 0,
        }

    monkeypatch.setattr(store_inventory.wallet, "ensure_wallet", fake_ensure_wallet)

    result = await store_inventory.exchange_product("user-1", "tea")

    assert result["wallet"]["point_balance"] == 901
    assert result["inventory_item"]["product_kind"] == "tea"
    assert result["inventory_item"]["quantity"] == 1
    assert fake_db.fake_tx.query_calls[0][1] == ("user-1", 99)
    assert fake_db.fake_tx.query_calls[1][1] == ("user-1", "tea")
    ledger_args = fake_db.fake_tx.execute_calls[0][1]
    assert ledger_args[1:4] == (-99, 901, "tea")
    assert json.loads(ledger_args[4]) == {"product_kind": "tea", "price": 99}


@pytest.mark.asyncio
async def test_exchange_product_rejects_unknown_product():
    with pytest.raises(ValueError, match="unknown_product"):
        await store_inventory.exchange_product("user-1", "not-real")
