from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.services import store_inventory
from app.services.store_catalog import EXCHANGE_PRODUCTS, _PRODUCTS


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


def test_catalog_covers_exchange_tabs_and_gift_subcategories():
    gifts = [p for p in EXCHANGE_PRODUCTS.values() if p.category == "gift"]
    blinds = [p for p in EXCHANGE_PRODUCTS.values() if p.category == "blind"]
    outfits = [p for p in EXCHANGE_PRODUCTS.values() if p.category == "outfit"]
    assert len(gifts) == 70
    assert len(blinds) == 7
    assert len(outfits) == 7
    assert {p.subcategory for p in gifts} == {
        "奢享",
        "数码",
        "生活",
        "美食",
        "配饰",
        "饮品",
        "饰品",
        "鲜花",
    }
    assert len(_PRODUCTS) == len(EXCHANGE_PRODUCTS) == 84
    for product in EXCHANGE_PRODUCTS.values():
        assert product.member_price > 0
        assert product.list_price >= product.member_price
    coffee = EXCHANGE_PRODUCTS["gift_1"]
    assert coffee.title == "美式咖啡"
    assert coffee.price_for(False) == 25
    assert coffee.price_for(True) == 18
    assert EXCHANGE_PRODUCTS["outfit_theme"].title == "主题皮肤"


@pytest.mark.asyncio
async def test_list_inventory_returns_owned_items(monkeypatch):
    acquired_at = datetime(2026, 7, 1, tzinfo=UTC)
    fake_db = _FakeDb(
        list_rows=[
            {
                "product_kind": "gift_1",
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
            "product_kind": "gift_1",
            "quantity": 2,
            "acquired_at": acquired_at.isoformat(),
            "updated_at": acquired_at.isoformat(),
        }
    ]
    assert fake_db.query_calls[0][1] == ("user-1",)


@pytest.mark.asyncio
async def test_exchange_product_charges_list_price_when_not_vip(monkeypatch):
    updated_at = datetime(2026, 7, 1, tzinfo=UTC)
    fake_db = _FakeDb(
        tx_rows=[
            [{"point_balance": 1000, "vip_until": None, "vip_trial_used": False}],
            [
                {
                    "ticket_balance": 0,
                    "point_balance": 975,
                    "achievement_points_synced": 0,
                }
            ],
            [
                {
                    "product_kind": "gift_1",
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

    result = await store_inventory.exchange_product("user-1", "gift_1")

    assert result["wallet"]["point_balance"] == 975
    assert result["inventory_item"]["product_kind"] == "gift_1"
    assert fake_db.fake_tx.query_calls[1][1] == ("user-1", 25)
    assert fake_db.fake_tx.query_calls[2][1] == ("user-1", "gift_1", 1)
    ledger_args = fake_db.fake_tx.execute_calls[0][1]
    assert ledger_args[1:4] == (-25, 975, "gift_1")
    assert json.loads(ledger_args[4]) == {
        "product_kind": "gift_1",
        "price": 25,
        "member_price": 18,
        "list_price": 25,
        "is_vip": False,
    }


@pytest.mark.asyncio
async def test_exchange_product_charges_member_price_when_vip(monkeypatch):
    updated_at = datetime(2026, 7, 1, tzinfo=UTC)
    fake_db = _FakeDb(
        tx_rows=[
            [
                {
                    "point_balance": 1000,
                    "vip_until": datetime.now(UTC) + timedelta(days=10),
                    "vip_trial_used": True,
                }
            ],
            [
                {
                    "ticket_balance": 0,
                    "point_balance": 982,
                    "achievement_points_synced": 0,
                }
            ],
            [
                {
                    "product_kind": "gift_1",
                    "quantity": 1,
                    "acquired_at": updated_at,
                    "updated_at": updated_at,
                }
            ],
        ]
    )
    monkeypatch.setattr(store_inventory, "db", fake_db)

    async def fake_ensure_wallet(user_id: str):
        return {
            "ticket_balance": 0,
            "point_balance": 1000,
            "achievement_points_synced": 0,
        }

    monkeypatch.setattr(store_inventory.wallet, "ensure_wallet", fake_ensure_wallet)

    result = await store_inventory.exchange_product("user-1", "gift_1")

    assert result["wallet"]["point_balance"] == 982
    assert fake_db.fake_tx.query_calls[1][1] == ("user-1", 18)
    payload = json.loads(fake_db.fake_tx.execute_calls[0][1][4])
    assert payload["price"] == 18
    assert payload["is_vip"] is True


@pytest.mark.asyncio
async def test_exchange_product_rejects_unknown_product():
    with pytest.raises(ValueError, match="unknown_product"):
        await store_inventory.exchange_product("user-1", "tea")


@pytest.mark.asyncio
async def test_get_catalog_marks_vip_status(monkeypatch):
    fake_db = _FakeDb(
        list_rows=[{"vip_until": None, "vip_trial_used": False}]
    )
    monkeypatch.setattr(store_inventory, "db", fake_db)

    async def fake_ensure_wallet(user_id: str):
        return {
            "ticket_balance": 0,
            "point_balance": 1000,
            "achievement_points_synced": 0,
        }

    monkeypatch.setattr(store_inventory.wallet, "ensure_wallet", fake_ensure_wallet)

    result = await store_inventory.get_catalog("user-1")

    assert result["is_vip"] is False
    assert result["vip_trial_available"] is True
    by_kind = {item["product_kind"]: item for item in result["products"]}
    assert by_kind["gift_1"]["price"] == 25
    assert by_kind["gift_1"]["member_price"] == 18
    assert result["bundles"]["vip_trial"]["available"] is True
    assert result["bundles"]["music"]["tiers"][0]["ticket_price"] == 10
