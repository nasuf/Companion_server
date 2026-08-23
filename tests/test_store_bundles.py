from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services import store_bundles


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
    def __init__(self, *, tx_rows=None):
        self.fake_tx = _FakeTx(tx_rows or [])

    def tx(self):
        return _TxContext(self.fake_tx)


async def _ok_wallet(_user_id: str):
    return {
        "ticket_balance": 100,
        "point_balance": 0,
        "achievement_points_synced": 0,
    }


@pytest.mark.asyncio
async def test_buy_music_coupon_spends_tickets_and_grants_inventory(monkeypatch):
    updated_at = datetime(2026, 8, 14, tzinfo=UTC)
    fake_db = _FakeDb(
        tx_rows=[
            [
                {
                    "ticket_balance": 90,
                    "point_balance": 0,
                    "achievement_points_synced": 0,
                }
            ],
            [
                {
                    "product_kind": "music_hour_coupon",
                    "quantity": 1,
                    "acquired_at": updated_at,
                    "updated_at": updated_at,
                }
            ],
        ]
    )
    monkeypatch.setattr(store_bundles, "db", fake_db)
    monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)

    result = await store_bundles.purchase_bundle(
        "user-1", "music_coupon", tier_id="1"
    )

    assert result["wallet"]["ticket_balance"] == 90
    assert result["inventory_item"]["product_kind"] == "music_hour_coupon"
    assert result["inventory_item"]["quantity"] == 1
    assert fake_db.fake_tx.query_calls[0][1] == ("user-1", 10)
    assert fake_db.fake_tx.query_calls[1][1] == (
        "user-1",
        "music_hour_coupon",
        1,
    )


@pytest.mark.asyncio
async def test_buy_makeup_card_spends_tickets_and_grants_inventory(monkeypatch):
    updated_at = datetime(2026, 8, 14, tzinfo=UTC)
    fake_db = _FakeDb(
        tx_rows=[
            [
                {
                    "ticket_balance": 70,
                    "point_balance": 0,
                    "achievement_points_synced": 0,
                }
            ],
            [
                {
                    "product_kind": "makeup_card",
                    "quantity": 1,
                    "acquired_at": updated_at,
                    "updated_at": updated_at,
                }
            ],
        ]
    )
    monkeypatch.setattr(store_bundles, "db", fake_db)
    monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)

    result = await store_bundles.purchase_bundle(
        "user-1", "makeup_card", tier_id="makeup_1"
    )

    assert result["wallet"]["ticket_balance"] == 70
    assert result["inventory_item"]["product_kind"] == "makeup_card"
    assert result["inventory_item"]["quantity"] == 1
    assert fake_db.fake_tx.query_calls[0][1] == ("user-1", 30)
    assert fake_db.fake_tx.query_calls[1][1] == (
        "user-1",
        "makeup_card",
        1,
    )


@pytest.mark.asyncio
async def test_buy_makeup_card_rejects_unknown_tier(monkeypatch):
    monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)
    with pytest.raises(ValueError, match="unknown_tier"):
        await store_bundles.purchase_bundle(
            "user-1", "makeup_card", tier_id="bogus"
        )


@pytest.mark.asyncio
async def test_buy_game_points_credits_game_wallet_not_shop_points(monkeypatch):
    fake_db = _FakeDb(
        tx_rows=[
            [
                {
                    "ticket_balance": 80,
                    "point_balance": 40,
                    "achievement_points_synced": 0,
                }
            ],
            [{"balance": 12}],
        ]
    )
    monkeypatch.setattr(store_bundles, "db", fake_db)
    monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)
    monkeypatch.setattr(store_bundles.game_points, "ensure_wallet", _ok_wallet)

    result = await store_bundles.purchase_bundle(
        "user-1", "game_points", tier_id="100"
    )

    assert result["wallet"]["ticket_balance"] == 80
    assert result["wallet"]["point_balance"] == 40
    assert result["game_balance"] == 112
    assert result["inventory_item"] is None
    assert fake_db.fake_tx.query_calls[0][1] == ("user-1", 20)
    game_update = fake_db.fake_tx.execute_calls[1][1]
    assert game_update == ("user-1", 112)
    ledger_args = fake_db.fake_tx.execute_calls[2][1]
    source_id = ledger_args[4]
    assert source_id.startswith("store_game:100:")
    assert source_id != "game_points:100"


@pytest.mark.asyncio
async def test_buy_game_points_can_be_purchased_twice(monkeypatch):
    source_ids: list[str] = []
    for _ in range(2):
        fake_db = _FakeDb(
            tx_rows=[
                [
                    {
                        "ticket_balance": 80,
                        "point_balance": 40,
                        "achievement_points_synced": 0,
                    }
                ],
                [{"balance": 12}],
            ]
        )
        monkeypatch.setattr(store_bundles, "db", fake_db)
        monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)
        monkeypatch.setattr(store_bundles.game_points, "ensure_wallet", _ok_wallet)
        await store_bundles.purchase_bundle(
            "user-1", "game_points", tier_id="100"
        )
        source_ids.append(fake_db.fake_tx.execute_calls[2][1][4])
    assert source_ids[0] != source_ids[1]
    assert all(sid.startswith("store_game:100:") for sid in source_ids)


@pytest.mark.asyncio
async def test_vip_trial_purchase_requires_payment():
    with pytest.raises(ValueError, match="payment_required"):
        await store_bundles.purchase_bundle("user-1", "vip_trial")


@pytest.mark.asyncio
async def test_activate_vip_trial_is_once_per_account(monkeypatch):
    until = datetime.now(UTC) + timedelta(days=30)
    fake_db = _FakeDb(
        tx_rows=[
            [
                {
                    "vip_until": None,
                    "vip_trial_used": False,
                    "ticket_balance": 0,
                    "point_balance": 0,
                    "achievement_points_synced": 0,
                }
            ],
            [
                {
                    "ticket_balance": 0,
                    "point_balance": 0,
                    "achievement_points_synced": 0,
                    "vip_until": until,
                    "vip_trial_used": True,
                }
            ],
        ]
    )
    monkeypatch.setattr(store_bundles, "db", fake_db)
    monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)

    result = await store_bundles.activate_vip_trial("user-1")
    assert result["wallet"]["ticket_balance"] == 0
    assert result["vip_until"]
    stored_until = fake_db.fake_tx.query_calls[1][1][1]
    assert stored_until.tzinfo is None


@pytest.mark.asyncio
async def test_activate_vip_trial_rejects_second_purchase(monkeypatch):
    fake_db = _FakeDb(
        tx_rows=[
            [
                {
                    "vip_until": None,
                    "vip_trial_used": True,
                    "ticket_balance": 0,
                    "point_balance": 0,
                    "achievement_points_synced": 0,
                }
            ]
        ]
    )
    monkeypatch.setattr(store_bundles, "db", fake_db)
    monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)

    with pytest.raises(ValueError, match="vip_trial_used"):
        await store_bundles.activate_vip_trial("user-1")
