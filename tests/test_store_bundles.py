from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

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


def _find_call(calls: list[tuple[str, tuple]], needle: str) -> tuple[str, tuple]:
    for query, args in calls:
        if needle in query:
            return query, args
    raise AssertionError(f"no call matched {needle!r} in {[q for q, _ in calls]}")


@pytest.mark.asyncio
async def test_buy_music_coupon_spends_tickets_and_grants_inventory(monkeypatch):
    updated_at = datetime(2026, 8, 14, tzinfo=UTC)
    fake_db = _FakeDb(
        tx_rows=[
            # debit_tickets_prioritized: lock-read, then update...returning
            [{"gift_ticket_balance": 0, "ticket_balance": 100}],
            [
                {
                    "gift_ticket_balance": 0,
                    "ticket_balance": 90,
                    "point_balance": 0,
                    "achievement_points_synced": 0,
                }
            ],
            # add_batch: insert...returning
            [
                {
                    "id": "batch-1",
                    "product_kind": "music_hour_coupon",
                    "quantity": 1,
                    "source": "purchase",
                    "expires_at": updated_at,
                    "created_at": updated_at,
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
    _, update_args = _find_call(fake_db.fake_tx.query_calls, "UPDATE user_wallets")
    assert update_args == ("user-1", 0, 10)  # (user_id, from_gift, from_perm)
    _, batch_args = _find_call(fake_db.fake_tx.query_calls, "INSERT INTO user_consumable_batch")
    assert batch_args[:4] == ("user-1", "music_hour_coupon", 1, "purchase")


@pytest.mark.asyncio
async def test_buy_makeup_card_spends_tickets_and_grants_inventory(monkeypatch):
    updated_at = datetime(2026, 8, 14, tzinfo=UTC)
    fake_db = _FakeDb(
        tx_rows=[
            [{"gift_ticket_balance": 0, "ticket_balance": 100}],
            [
                {
                    "gift_ticket_balance": 0,
                    "ticket_balance": 70,
                    "point_balance": 0,
                    "achievement_points_synced": 0,
                }
            ],
            [
                {
                    "id": "batch-2",
                    "product_kind": "makeup_card",
                    "quantity": 1,
                    "source": "purchase",
                    "expires_at": None,
                    "created_at": updated_at,
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
    _, update_args = _find_call(fake_db.fake_tx.query_calls, "UPDATE user_wallets")
    assert update_args == ("user-1", 0, 30)
    _, batch_args = _find_call(fake_db.fake_tx.query_calls, "INSERT INTO user_consumable_batch")
    assert batch_args[:4] == ("user-1", "makeup_card", 1, "purchase")


@pytest.mark.asyncio
async def test_buy_makeup_card_rejects_unknown_tier(monkeypatch):
    monkeypatch.setattr(store_bundles.wallet, "ensure_wallet", _ok_wallet)
    with pytest.raises(ValueError, match="unknown_tier"):
        await store_bundles.purchase_bundle(
            "user-1", "makeup_card", tier_id="bogus"
        )


def _game_points_tx_rows() -> list[list[dict]]:
    return [
        [{"gift_ticket_balance": 0, "ticket_balance": 100}],  # debit lock-select
        [
            {
                "gift_ticket_balance": 0,
                "ticket_balance": 80,
                "point_balance": 40,
                "achievement_points_synced": 0,
            }
        ],  # debit update...returning
        [{"balance": 12}],  # game wallet balance select
    ]


@pytest.mark.asyncio
async def test_buy_game_points_credits_game_wallet_not_shop_points(monkeypatch):
    fake_db = _FakeDb(tx_rows=_game_points_tx_rows())
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
    _, update_args = _find_call(fake_db.fake_tx.query_calls, "UPDATE user_wallets")
    assert update_args == ("user-1", 0, 20)  # (user_id, from_gift, from_perm)
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
        fake_db = _FakeDb(tx_rows=_game_points_tx_rows())
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
async def test_activate_vip_trial_immediately_grants_monthly_benefits(monkeypatch):
    """CLAUDE.md 权益项 3/4/6: 新 VIP 不该等到次日 02:45 的 cron 才拿到东西
    —— 付了钱却要等近 24h 才看到限时钞票/音乐券/补签卡, 体感是"钱白付了"。
    """
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
    grant_mock = AsyncMock(
        return_value={
            "wallet": {
                "gift_ticket_balance": 40,
                "ticket_balance": 0,
                "point_balance": 0,
                "achievement_points_synced": 0,
            },
            "music_coupon_batch": {"quantity": 20},
            "makeup_card_batch": {"quantity": 2},
        }
    )
    monkeypatch.setattr(store_bundles.grants, "grant_monthly", grant_mock)

    result = await store_bundles.activate_vip_trial("user-1")

    grant_mock.assert_awaited_once_with("user-1")
    # The response must reflect the just-granted gift tickets, not the
    # pre-grant (all-zero) balance from the VIP-activation transaction alone.
    assert result["wallet"]["gift_ticket_balance"] == 40


@pytest.mark.asyncio
async def test_activate_vip_trial_still_succeeds_if_immediate_grant_fails(monkeypatch):
    """The nightly vip_monthly_grant cron is the safety net: a failure in the
    best-effort immediate grant must not turn a successful VIP purchase into
    an error response for the paying user.
    """
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
    monkeypatch.setattr(
        store_bundles.grants, "grant_monthly", AsyncMock(side_effect=RuntimeError("db hiccup"))
    )

    result = await store_bundles.activate_vip_trial("user-1")

    assert result["vip_until"]
    assert result["wallet"]["ticket_balance"] == 0


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
