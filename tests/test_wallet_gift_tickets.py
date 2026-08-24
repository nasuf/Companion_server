from __future__ import annotations

from typing import Any

import pytest

from app.services import wallet


class _FakeGiftTx:
    """Fake transactional client for the gift-ticket wallet helpers.

    Tracks gift_ticket_balance/ticket_balance/point_balance/achievement_points_synced
    and dispatches by SQL substring, mirroring the exact statements in wallet.py.
    """

    def __init__(
        self,
        *,
        gift_ticket_balance: int = 0,
        ticket_balance: int = 0,
        point_balance: int = 0,
        achievement_points_synced: int = 0,
    ):
        self.gift_ticket_balance = gift_ticket_balance
        self.ticket_balance = ticket_balance
        self.point_balance = point_balance
        self.achievement_points_synced = achievement_points_synced
        self.ledger_calls: list[tuple] = []

    def _row(self) -> dict[str, Any]:
        return {
            "gift_ticket_balance": self.gift_ticket_balance,
            "ticket_balance": self.ticket_balance,
            "point_balance": self.point_balance,
            "achievement_points_synced": self.achievement_points_synced,
        }

    async def query_raw(self, query: str, *args):
        if "SELECT gift_ticket_balance, ticket_balance" in query and "FOR UPDATE" in query:
            return [self._row()]
        if "UPDATE user_wallets" in query and "RETURNING" in query:
            if "gift_ticket_balance = gift_ticket_balance -" in query:
                self.gift_ticket_balance -= args[1]
                self.ticket_balance -= args[2]
            elif "gift_ticket_balance = gift_ticket_balance +" in query:
                self.gift_ticket_balance += args[1]
            elif "gift_ticket_balance = 0" in query:
                self.gift_ticket_balance = 0
            return [self._row()]
        raise AssertionError(f"unexpected query_raw: {query}")

    async def execute_raw(self, query: str, *args):
        if "INSERT INTO wallet_ledger" in query:
            self.ledger_calls.append(args)
            return 1
        raise AssertionError(f"unexpected execute_raw: {query}")


@pytest.mark.asyncio
async def test_debit_tickets_prioritized_drains_gift_before_permanent():
    tx = _FakeGiftTx(gift_ticket_balance=5, ticket_balance=10)

    balance = await wallet.debit_tickets_prioritized(
        "u1", 8, source="chat_overage", client=tx
    )

    assert balance["gift_ticket_balance"] == 0
    assert balance["ticket_balance"] == 7
    assert len(tx.ledger_calls) == 2
    assert tx.ledger_calls[0][1:3] == ("gift_ticket", -5)
    assert tx.ledger_calls[1][1:3] == ("ticket", -3)


@pytest.mark.asyncio
async def test_debit_tickets_prioritized_skips_ledger_row_for_untouched_bucket():
    tx = _FakeGiftTx(gift_ticket_balance=10, ticket_balance=10)

    balance = await wallet.debit_tickets_prioritized(
        "u1", 4, source="chat_overage", client=tx
    )

    assert balance["gift_ticket_balance"] == 6
    assert balance["ticket_balance"] == 10
    # Only the gift bucket moved -> exactly one ledger row, not two.
    assert len(tx.ledger_calls) == 1
    assert tx.ledger_calls[0][1:3] == ("gift_ticket", -4)


@pytest.mark.asyncio
async def test_debit_tickets_prioritized_raises_when_combined_balance_short():
    tx = _FakeGiftTx(gift_ticket_balance=1, ticket_balance=2)

    with pytest.raises(ValueError, match="insufficient_ticket_balance"):
        await wallet.debit_tickets_prioritized("u1", 5, source="chat_overage", client=tx)

    assert tx.ledger_calls == []


@pytest.mark.asyncio
async def test_credit_gift_tickets_adds_to_gift_bucket_only():
    tx = _FakeGiftTx(gift_ticket_balance=0, ticket_balance=3)

    balance = await wallet.credit_gift_tickets(
        "u1", 40, source="vip_monthly_grant", client=tx
    )

    assert balance["gift_ticket_balance"] == 40
    assert balance["ticket_balance"] == 3
    assert tx.ledger_calls[0][1:3] == ("gift_ticket", 40)


@pytest.mark.asyncio
async def test_zero_gift_tickets_clears_balance_and_records_negative_ledger():
    tx = _FakeGiftTx(gift_ticket_balance=25, ticket_balance=3)

    balance = await wallet.zero_gift_tickets(
        "u1", source="vip_expire_clear", client=tx
    )

    assert balance["gift_ticket_balance"] == 0
    assert tx.ledger_calls[0][1:3] == ("gift_ticket", -25)


@pytest.mark.asyncio
async def test_zero_gift_tickets_is_a_no_op_ledger_when_already_zero():
    tx = _FakeGiftTx(gift_ticket_balance=0, ticket_balance=3)

    balance = await wallet.zero_gift_tickets(
        "u1", source="vip_expire_clear", client=tx
    )

    assert balance["gift_ticket_balance"] == 0
    assert tx.ledger_calls == []
