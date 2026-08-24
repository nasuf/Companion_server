from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.vip import grants


class _FakeTx:
    def __init__(self):
        self.execute_calls: list[tuple[str, tuple]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute_raw(self, query: str, *args):
        self.execute_calls.append((query, args))
        return 1


class _FakeDb:
    def __init__(self):
        self.tx_obj = _FakeTx()

    def tx(self):
        return self.tx_obj


@pytest.mark.asyncio
async def test_grant_monthly_credits_gift_tickets_and_grants_batches(monkeypatch):
    fake_db = _FakeDb()
    monkeypatch.setattr(grants, "db", fake_db)
    monkeypatch.setattr(grants.wallet, "ensure_wallet", AsyncMock())
    credit_mock = AsyncMock(return_value={"gift_ticket_balance": 40})
    monkeypatch.setattr(grants.wallet, "credit_gift_tickets", credit_mock)
    add_batch_mock = AsyncMock(
        side_effect=[
            {"product_kind": "music_hour_coupon", "quantity": 20},
            {"product_kind": "makeup_card", "quantity": 2},
        ]
    )
    monkeypatch.setattr(grants, "add_batch", add_batch_mock)

    result = await grants.grant_monthly("u1")

    credit_mock.assert_awaited_once()
    assert credit_mock.call_args.args[:2] == ("u1", 40)
    assert credit_mock.call_args.kwargs["source"] == grants.SOURCE_VIP_MONTHLY_GRANT

    assert add_batch_mock.await_count == 2
    music_call = add_batch_mock.await_args_list[0]
    assert music_call.args[:2] == ("u1", "music_hour_coupon")
    assert music_call.kwargs["quantity"] == 20
    assert music_call.kwargs["source"] == "vip_grant"
    assert music_call.kwargs["expires_at"] is not None

    makeup_call = add_batch_mock.await_args_list[1]
    assert makeup_call.args[:2] == ("u1", "makeup_card")
    assert makeup_call.kwargs["quantity"] == 2

    # vip_last_grant_at anchor must be stamped so the next scan isn't re-due.
    assert any(
        "vip_last_grant_at" in query for query, _ in fake_db.tx_obj.execute_calls
    )
    assert result["wallet"]["gift_ticket_balance"] == 40


@pytest.mark.asyncio
async def test_clear_on_lapse_zeroes_gift_tickets_and_vip_grant_batches(monkeypatch):
    fake_db = _FakeDb()
    monkeypatch.setattr(grants, "db", fake_db)
    zero_mock = AsyncMock(return_value={"gift_ticket_balance": 0})
    monkeypatch.setattr(grants.wallet, "zero_gift_tickets", zero_mock)

    result = await grants.clear_on_lapse("u1")

    zero_mock.assert_awaited_once()
    assert zero_mock.call_args.kwargs["source"] == grants.SOURCE_VIP_EXPIRE_CLEAR
    # Only vip_grant batches are invalidated -- purchased (permanent/30-day)
    # batches must survive a VIP lapse.
    query, args = fake_db.tx_obj.execute_calls[0]
    assert "source = 'vip_grant'" in query
    assert args == ("u1",)
    assert result["wallet"]["gift_ticket_balance"] == 0
