from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.vip import music_quota


@pytest.fixture(autouse=True)
def _stub_ensure_wallet(monkeypatch):
    """report() now calls wallet.ensure_wallet() before touching the wallet
    (mirrors chat_quota's defensive pattern — see code review finding).
    Real ensure_wallet hits the DB via app.services.wallet's module-level
    `db`, which these tests don't fake; stub it out so it's a no-op here.
    """
    monkeypatch.setattr(music_quota.wallet, "ensure_wallet", AsyncMock())


class _FakeMusicDb:
    def __init__(
        self,
        *,
        listened_seconds: int = 0,
        provisioned_seconds: int = 0,
        coupon_units: int = 0,
        ticket_spent: int = 0,
    ):
        self.listened_seconds = listened_seconds
        self.provisioned_seconds = provisioned_seconds
        self.coupon_units = coupon_units
        self.ticket_spent = ticket_spent
        self.persisted: dict | None = None

    def tx(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def query_raw(self, query: str, *args):
        if "INSERT INTO user_music_quota" in query:
            return [
                {
                    "listened_seconds": self.listened_seconds,
                    "provisioned_seconds": self.provisioned_seconds,
                    "coupon_units": self.coupon_units,
                    "ticket_spent": self.ticket_spent,
                }
            ]
        raise AssertionError(f"unexpected query_raw: {query}")

    async def execute_raw(self, query: str, *args):
        if "UPDATE user_music_quota" in query:
            self.persisted = {
                "listened_seconds": args[1],
                "provisioned_seconds": args[2],
                "coupon_units": args[3],
                "ticket_spent": args[4],
            }
            return 1
        raise AssertionError(f"unexpected execute_raw: {query}")


@pytest.mark.asyncio
async def test_report_fully_covered_by_free_allowance(monkeypatch):
    fake_db = _FakeMusicDb(listened_seconds=0)
    monkeypatch.setattr(music_quota, "db", fake_db)

    result = await music_quota.report("u1", is_vip=False, delta_seconds=600)

    assert result == {"action": "none", "accepted_seconds": 600, "pending_seconds": 0, "ticket_cost": 0}
    assert fake_db.persisted["listened_seconds"] == 600


@pytest.mark.asyncio
async def test_report_drains_banked_provisioned_seconds_before_buying_more(monkeypatch):
    # Free allowance already exhausted (1800s), but a prior report over-bought
    # 600s of coverage that hasn't been "spent" by listening yet.
    fake_db = _FakeMusicDb(listened_seconds=1800, provisioned_seconds=600)
    monkeypatch.setattr(music_quota, "db", fake_db)
    consume_mock = AsyncMock()
    monkeypatch.setattr(music_quota, "consume_batch_units", consume_mock)

    result = await music_quota.report("u1", is_vip=False, delta_seconds=600)

    assert result["action"] == "none"
    consume_mock.assert_not_called()
    assert fake_db.persisted["listened_seconds"] == 2400
    assert fake_db.persisted["provisioned_seconds"] == 0


@pytest.mark.asyncio
async def test_report_consumes_one_coupon_when_free_and_bank_are_exhausted(monkeypatch):
    fake_db = _FakeMusicDb(listened_seconds=1800, provisioned_seconds=0)
    monkeypatch.setattr(music_quota, "db", fake_db)
    consume_mock = AsyncMock()
    monkeypatch.setattr(music_quota, "consume_batch_units", consume_mock)

    result = await music_quota.report("u1", is_vip=False, delta_seconds=1000)

    assert result == {"action": "none", "accepted_seconds": 1000, "pending_seconds": 0, "ticket_cost": 0}
    consume_mock.assert_awaited_once()
    assert consume_mock.call_args.args[2] == 1  # ceil(1000/3600) coupon
    # 1 coupon banks a full hour; only 1000s of it was needed this report.
    assert fake_db.persisted["provisioned_seconds"] == 3600
    assert fake_db.persisted["coupon_units"] == 1


@pytest.mark.asyncio
async def test_report_no_coupon_unconfirmed_with_enough_tickets_asks_to_confirm(monkeypatch):
    fake_db = _FakeMusicDb(listened_seconds=1800)
    monkeypatch.setattr(music_quota, "db", fake_db)
    monkeypatch.setattr(
        music_quota, "consume_batch_units", AsyncMock(side_effect=ValueError("insufficient_inventory"))
    )
    monkeypatch.setattr(
        music_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 100})
    )

    result = await music_quota.report("u1", is_vip=False, delta_seconds=900)

    assert result["action"] == "confirm_ticket"
    assert result["ticket_cost"] == 10  # 1 half-hour block * 10 (non-VIP rate)
    assert result["pending_seconds"] == 900
    # Nothing charged yet -- only the free portion (0 here) would be persisted.
    assert fake_db.persisted["listened_seconds"] == 1800


@pytest.mark.asyncio
async def test_report_vip_rate_is_cheaper_than_non_vip(monkeypatch):
    fake_db = _FakeMusicDb(listened_seconds=1800)
    monkeypatch.setattr(music_quota, "db", fake_db)
    monkeypatch.setattr(
        music_quota, "consume_batch_units", AsyncMock(side_effect=ValueError("insufficient_inventory"))
    )
    monkeypatch.setattr(
        music_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 100})
    )

    result = await music_quota.report("u1", is_vip=True, delta_seconds=900)

    assert result["ticket_cost"] == 5  # VIP half-hour rate


@pytest.mark.asyncio
async def test_report_confirmed_charge_succeeds_and_records_ticket_spend(monkeypatch):
    fake_db = _FakeMusicDb(listened_seconds=1800)
    monkeypatch.setattr(music_quota, "db", fake_db)
    monkeypatch.setattr(
        music_quota, "consume_batch_units", AsyncMock(side_effect=ValueError("insufficient_inventory"))
    )
    debit_mock = AsyncMock()
    monkeypatch.setattr(music_quota.wallet, "debit_tickets_prioritized", debit_mock)

    result = await music_quota.report(
        "u1", is_vip=False, delta_seconds=900, paid_confirmed=True
    )

    assert result == {"action": "none", "accepted_seconds": 900, "pending_seconds": 0, "ticket_cost": 0}
    debit_mock.assert_awaited_once()
    assert debit_mock.call_args.args[:2] == ("u1", 10)
    assert fake_db.persisted["ticket_spent"] == 10


@pytest.mark.asyncio
async def test_report_confirmed_charge_fails_falls_back_to_buy_prompt(monkeypatch):
    fake_db = _FakeMusicDb(listened_seconds=1800)
    monkeypatch.setattr(music_quota, "db", fake_db)
    monkeypatch.setattr(
        music_quota, "consume_batch_units", AsyncMock(side_effect=ValueError("insufficient_inventory"))
    )
    monkeypatch.setattr(
        music_quota.wallet,
        "debit_tickets_prioritized",
        AsyncMock(side_effect=ValueError("insufficient_ticket_balance")),
    )

    result = await music_quota.report(
        "u1", is_vip=True, delta_seconds=900, paid_confirmed=True
    )

    assert result["action"] == "buy_coupon"
    assert fake_db.persisted["ticket_spent"] == 0


@pytest.mark.asyncio
async def test_report_rejects_non_positive_delta():
    with pytest.raises(ValueError, match="invalid_amount"):
        await music_quota.report("u1", is_vip=False, delta_seconds=0)


@pytest.mark.asyncio
async def test_report_ensures_wallet_exists_before_touching_balance(monkeypatch):
    # Regression guard: a user who has never touched their wallet (pure
    # listening, no chat/store activity yet) must not hit wallet_not_found
    # the first time report() needs to read/charge tickets.
    fake_db = _FakeMusicDb(listened_seconds=1800)
    monkeypatch.setattr(music_quota, "db", fake_db)
    monkeypatch.setattr(
        music_quota, "consume_batch_units", AsyncMock(side_effect=ValueError("insufficient_inventory"))
    )
    monkeypatch.setattr(
        music_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 100})
    )
    ensure_wallet_mock = AsyncMock()
    monkeypatch.setattr(music_quota.wallet, "ensure_wallet", ensure_wallet_mock)

    await music_quota.report("u1", is_vip=False, delta_seconds=900)

    ensure_wallet_mock.assert_awaited_once_with("u1")
