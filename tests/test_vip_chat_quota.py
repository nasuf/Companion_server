from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.vip import chat_quota


class _FakeQuotaDb:
    """Minimal fake mirroring chat_quota's exact SQL shapes, tracking the two
    pieces of state it mutates: the message-quota counter and the fractional
    ticket accrual (+ its owning period) on the wallet row.
    """

    def __init__(
        self,
        *,
        used: int = 0,
        overage_accrued: float = 0.0,
        overage_period: str | None = None,
    ):
        self.used = used
        self.overage_accrued = overage_accrued
        self.overage_period = overage_period
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
        if "UPDATE user_wallets" in query and "overage_accrued" in query:
            self.overage_accrued = args[1]
            self.overage_period = args[2]
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
        if "overage_accrued" in query and "user_wallets" in query:
            return [
                {
                    "overage_accrued": self.overage_accrued,
                    "overage_period": self.overage_period,
                }
            ]
        raise AssertionError(f"unexpected query_raw: {query}")


def _current_period(is_vip: bool) -> str:
    scope, key, _ = chat_quota.config.message_period(is_vip)
    return f"{scope}:{key}"


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
    full_wallet_mock = AsyncMock(return_value={"spendable_tickets": 3})
    monkeypatch.setattr(chat_quota.wallet, "full_wallet", full_wallet_mock)

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=False)

    assert result["allowed"] is False
    assert result["reason"] == "paid_confirm"
    assert result["per_msg_cost"] == 0.5
    # Cancelling must be a true no-op: no message counted, no ticket charged.
    assert fake_db.used == 20
    assert fake_db.overage_accrued == 0.0
    # Must read through the already-open tx (here `fake_db` itself, since
    # _FakeQuotaDb.tx() returns self) rather than borrowing a second pool
    # connection while still holding the row lock on user_message_quota.
    full_wallet_mock.assert_awaited_once_with("u1", client=fake_db)


@pytest.mark.asyncio
async def test_consume_one_over_quota_no_tickets_blocks(monkeypatch):
    fake_db = _FakeQuotaDb(used=20)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 0})
    )

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=False)

    assert result["allowed"] is False
    assert result["mode"] == "blocked"
    assert result["reason"] == "no_ticket"


@pytest.mark.asyncio
async def test_consume_one_confirmed_accrues_fraction_without_charging_yet(monkeypatch):
    fake_db = _FakeQuotaDb(used=20, overage_accrued=0.0)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    ensure_wallet_mock = AsyncMock()
    monkeypatch.setattr(chat_quota.wallet, "ensure_wallet", ensure_wallet_mock)
    debit_mock = AsyncMock()
    monkeypatch.setattr(chat_quota.wallet, "debit_tickets_prioritized", debit_mock)

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=True)

    # 0.5/句 accrued once is still < 1, so no whole ticket is charged yet.
    assert result == {"allowed": True, "mode": "paid", "used": 21, "limit": 20, "charged": 0}
    debit_mock.assert_not_called()
    assert fake_db.overage_accrued == 0.5
    assert fake_db.used == 21
    # Same pool-starvation concern as full_wallet above: must read through
    # the open tx, not borrow a second connection mid-transaction.
    ensure_wallet_mock.assert_awaited_once_with("u1", client=fake_db)


@pytest.mark.asyncio
async def test_consume_one_confirmed_charges_whole_ticket_once_accrual_crosses_one(monkeypatch):
    # 0.7 累积在*当前*周期内 (overage_period 与本次算出的周期一致) —— 同一
    # 周期内继续累积才该被结算, 见下面 test_consume_one_stale_period_accrual_
    # is_dropped_not_charged 里"上个周期"的对照场景。
    fake_db = _FakeQuotaDb(
        used=20, overage_accrued=0.7, overage_period=_current_period(False)
    )
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(chat_quota.wallet, "ensure_wallet", AsyncMock())
    debit_mock = AsyncMock()
    monkeypatch.setattr(chat_quota.wallet, "debit_tickets_prioritized", debit_mock)

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=True)

    # 0.7 + 0.5 = 1.2 -> charge 1 whole ticket, keep 0.2 remainder accrued.
    assert result["charged"] == 1
    debit_mock.assert_awaited_once()
    assert debit_mock.call_args.args[:2] == ("u1", 1)
    assert fake_db.overage_accrued == 0.2
    assert fake_db.overage_period == _current_period(False)
    assert fake_db.used == 21


@pytest.mark.asyncio
async def test_consume_one_confirmed_insufficient_balance_blocks_without_writes(monkeypatch):
    fake_db = _FakeQuotaDb(
        used=20, overage_accrued=0.7, overage_period=_current_period(False)
    )
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(chat_quota.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(
        chat_quota.wallet,
        "debit_tickets_prioritized",
        AsyncMock(side_effect=ValueError("insufficient_ticket_balance")),
    )

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=True)

    assert result["allowed"] is False
    assert result["reason"] == "no_ticket"
    # A race between preview and confirm must not silently count/charge.
    assert fake_db.used == 20
    assert fake_db.overage_accrued == 0.7


@pytest.mark.asyncio
async def test_consume_one_stale_period_accrual_is_dropped_not_charged(monkeypatch):
    """跨周期的历史欠账不会在新周期第一次超额时被悄悄一起结算扣款——这正是
    用户报告的场景: 剩 1 钞票, 确认发送第一句就直接把余额扣光, 根因是上个
    周期残留的 0.5 和这次的 0.5 一起凑成了 1。
    """
    fake_db = _FakeQuotaDb(used=20, overage_accrued=0.7, overage_period="day:2020-01-01")
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(chat_quota.wallet, "ensure_wallet", AsyncMock())
    debit_mock = AsyncMock()
    monkeypatch.setattr(chat_quota.wallet, "debit_tickets_prioritized", debit_mock)

    result = await chat_quota.consume_one("u1", is_vip=False, paid_confirmed=True)

    # 上个周期的 0.7 被放弃; 这次从 0 开始只累积这一句的 0.5, 不该凑够 1。
    assert result == {"allowed": True, "mode": "paid", "used": 21, "limit": 20, "charged": 0}
    debit_mock.assert_not_called()
    assert fake_db.overage_accrued == 0.5
    assert fake_db.overage_period == _current_period(False)


@pytest.mark.asyncio
async def test_consume_one_vip_uses_monthly_bucket_and_cheaper_overage(monkeypatch):
    fake_db = _FakeQuotaDb(used=5199)
    monkeypatch.setattr(chat_quota, "db", fake_db)

    result = await chat_quota.consume_one("u1", is_vip=True)
    assert result == {"allowed": True, "mode": "free", "used": 5200, "limit": 5200, "charged": 0}

    fake_db.used = 5200
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 1})
    )
    blocked = await chat_quota.consume_one("u1", is_vip=True, paid_confirmed=False)
    assert blocked["per_msg_cost"] == 0.3


@pytest.mark.asyncio
async def test_preview_includes_admin_fields(monkeypatch):
    fake_db = _FakeQuotaDb(used=7)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 3})
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
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 5})
    )

    result = await chat_quota.admin_reset("u1", is_vip=False)

    assert fake_db.used == 0
    assert result["used"] == 0
    assert result["free_remaining"] == 20
    assert result["mode"] == "free"
    # Resetting usage must not touch the fractional overage accrual — that's
    # money already earmarked to be charged, a separate concern from "give
    # back unused free messages".
    assert not any(
        "overage_accrued" in query for query, _ in fake_db.execute_calls
    )


@pytest.mark.asyncio
async def test_admin_reset_uses_vip_monthly_period(monkeypatch):
    fake_db = _FakeQuotaDb(used=100)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 0})
    )

    result = await chat_quota.admin_reset("u1", is_vip=True)

    assert result["period_scope"] == "month"
    assert result["limit"] == 5200
    assert fake_db.used == 0
    reset_call = next(
        args for query, args in fake_db.execute_calls if "used = 0" in query
    )
    assert reset_call[1] == "month"


@pytest.mark.asyncio
async def test_admin_reset_is_a_no_op_when_no_usage_yet(monkeypatch):
    fake_db = _FakeQuotaDb(used=0)
    monkeypatch.setattr(chat_quota, "db", fake_db)
    monkeypatch.setattr(
        chat_quota.wallet, "full_wallet", AsyncMock(return_value={"spendable_tickets": 0})
    )

    result = await chat_quota.admin_reset("u1", is_vip=False)

    assert result["used"] == 0
    assert result["mode"] == "free"
