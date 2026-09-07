from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.vip import chat_quota


class _FakeQuotaDb:
    """Minimal fake mirroring chat_quota's SQL shapes."""

    def __init__(self, *, used: int = 0):
        self.used = used
        self.execute_calls: list[tuple[str, tuple]] = []

    def tx(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute_raw(self, query: str, *args):
        self.execute_calls.append((query, args))
        if "INSERT INTO user_message_quota" in query:
            return 1
        if "UPDATE user_message_quota" in query and "used = used + 1" in query:
            self.used += 1
            return 1
        if "UPDATE user_message_quota" in query and "used = 0" in query:
            self.used = 0
            return 1
        raise AssertionError(f"unexpected execute_raw: {query}")

    async def query_raw(self, query: str, *args):
        if "SELECT used FROM user_message_quota" in query:
            return [{"used": self.used}]
        raise AssertionError(f"unexpected query_raw: {query}")


@pytest.mark.asyncio
async def test_consume_one_free_within_daily_quota(monkeypatch):
    fake_db = _FakeQuotaDb(used=5)
    monkeypatch.setattr(chat_quota, "db", fake_db)

    result = await chat_quota.consume_one("u1", is_vip=False)

    assert result == {"allowed": True, "mode": "free", "used": 6, "limit": 20, "charged": 0}


@pytest.mark.asyncio
async def test_consume_one_over_quota_unconfirmed_blocks_without_side_effects(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    full_wallet_mock = AsyncMock(return_value={"spendable_tickets": 3.0})
    monkeypatch.setattr(chat_quota.wallet, "full_wallet", full_wallet_mock)

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=False)

    assert result["allowed"] is False
    assert result["reason"] == "paid_confirm"
    assert result["per_msg_cost"] == 0.5
    assert fake_db.used == 20
    full_wallet_mock.assert_awaited_once_with("u1", client=fake_db)


@pytest.mark.asyncio
async def test_consume_one_over_quota_no_tickets_blocks(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 0.0})
    )

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=False)

    assert result["allowed"] is False
    assert result["mode"] == "blocked"
    assert result["reason"] == "no_ticket"


@pytest.mark.asyncio
async def test_consume_one_confirmed_debits_half_ticket_per_message(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    ensure_wallet_mock = AsyncMock()
    monkeypatch.setattr(chat_quota.wallet, "ensure_wallet", ensure_wallet_mock)
    debit_mock = AsyncMock()
    monkeypatch.setattr(chat_quota.wallet, "debit_tickets_prioritized", debit_mock)
    monkeypatch.setattr(
        chat_quota.wallet,
        "full_wallet",
        AsyncMock(return_value={"spendable_tickets": 2.0}),
    )

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=True)

    assert result == {
        "allowed": True,
        "mode": "paid",
        "used": 21,
        "limit": 20,
        "charged": 0.5,
    }
    debit_mock.assert_awaited_once()
    assert debit_mock.call_args.args[:2] == ("u1", 0.5)
    ensure_wallet_mock.assert_awaited_once_with("u1", client=fake_db)


@pytest.mark.asyncio
async def test_consume_one_confirmed_three_messages_charges_one_point_five_total(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(chat_quota.wallet, "ensure_wallet", AsyncMock())
    debit_mock = AsyncMock()
    monkeypatch.setattr(chat_quota.wallet, "debit_tickets_prioritized", debit_mock)
    monkeypatch.setattr(
        chat_quota.wallet,
        "full_wallet",
        AsyncMock(return_value={"spendable_tickets": 5.0}),
    )

    total_charged = 0.0
    for _ in range(3):
        result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=True)
        total_charged += float(result["charged"])

    assert total_charged == 1.5
    assert debit_mock.await_count == 3
    assert all(call.args[1] == 0.5 for call in debit_mock.call_args_list)


@pytest.mark.asyncio
async def test_consume_one_confirmed_insufficient_balance_blocks_without_writes(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(chat_quota.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(
        chat_quota.wallet,
        "full_wallet",
        AsyncMock(return_value={"spendable_tickets": 1.0}),
    )
    monkeypatch.setattr(
        chat_quota.wallet,
        "debit_tickets_prioritized",
        AsyncMock(side_effect=ValueError("insufficient_ticket_balance")),
    )

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=True)

    assert result["allowed"] is False
    assert result["reason"] == "no_ticket"
    assert fake_db.used == 20


@pytest.mark.asyncio
async def test_consume_one_blocks_when_balance_below_per_message_cost(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 0.4})
    )

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=False)

    assert result["allowed"] is False
    assert result["reason"] == "no_ticket"


@pytest.mark.asyncio
async def test_consume_one_vip_uses_monthly_bucket_and_cheaper_overage(monkeypatch):
    fake_db = _FakeQuotaDb(used=5199)
    monkeypatch.setattr(chat_quota, "db", fake_db)

    result = await chat_quota.consume_one("u1", is_vip=True)
    assert result == {"allowed": True, "mode": "free", "used": 5200, "limit": 5200, "charged": 0}

    fake_db.used = 5200
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 1.0})
    )
    blocked = await chat_quota.consume_one("u1", is_vip=True, paid_confirmed=False)
    assert blocked["per_msg_cost"] == 0.3


@pytest.mark.asyncio
async def test_preview_includes_admin_fields(monkeypatch):
    fake_db = _FakeQuotaDb(used=7)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 3.5})
    )

    result = await chat_quota.preview("u1", is_vip=False)

    assert result["used"] == 7
    assert result["limit"] == 20
    assert result["period_scope"] == "day"
    assert result["free_remaining"] == 13


@pytest.mark.asyncio
async def test_admin_reset_zeroes_used_and_returns_fresh_preview(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 5.0})
    )

    result = await chat_quota.admin_reset("u1", is_vip=False)

    assert fake_db.used == 0
    assert result["used"] == 0
    assert result["free_remaining"] == 20
    assert result["mode"] == "free"
