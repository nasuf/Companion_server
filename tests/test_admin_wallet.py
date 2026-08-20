from __future__ import annotations

import json

import pytest

from app.services import wallet


class _FakeTx:
    """Minimal transaction stub: canned query rows + captured execute calls."""

    def __init__(self, rows_by_query: list[list[dict]]):
        self.rows_by_query = list(rows_by_query)
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
    def __init__(self, *, query_rows=None, tx_rows=None):
        self.query_rows = list(query_rows or [])
        self.query_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self.fake_tx = _FakeTx(tx_rows or [])

    async def query_raw(self, query: str, *args):
        self.query_calls.append((query, args))
        return self.query_rows.pop(0)

    async def execute_raw(self, query: str, *args):
        self.execute_calls.append((query, args))
        return 1

    def tx(self):
        return _TxContext(self.fake_tx)


def _balance_row(ticket: int, point: int = 0, synced: int = 0) -> dict:
    return {
        "ticket_balance": ticket,
        "point_balance": point,
        "achievement_points_synced": synced,
    }


@pytest.mark.asyncio
async def test_admin_adjust_credits_and_writes_audit_ledger(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[
            [{"1": 1}],          # _ensure_user_exists
            [_balance_row(10)],  # ensure_wallet
        ],
        tx_rows=[
            [_balance_row(10)],  # SELECT ... FOR UPDATE
            [_balance_row(28)],  # UPDATE ... RETURNING (10 + 18)
        ],
    )
    monkeypatch.setattr(wallet, "db", fake_db)

    result = await wallet.admin_adjust_tickets(
        "u1", 18, admin_id="admin-9", note="补偿"
    )

    assert result["delta"] == 18
    assert result["ticket_balance"] == 28
    # Ledger row is the audit record.
    ledger_sql, ledger_args = fake_db.fake_tx.execute_calls[0]
    assert "INSERT INTO wallet_ledger" in ledger_sql
    assert wallet.SOURCE_ADMIN_GRANT in ledger_args
    metadata = json.loads(ledger_args[-1])
    assert metadata["admin_id"] == "admin-9"
    assert metadata["requested"] == 18
    assert metadata["applied"] == 18
    assert metadata["note"] == "补偿"


@pytest.mark.asyncio
async def test_admin_adjust_floors_at_zero(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[
            [{"1": 1}],         # _ensure_user_exists
            [_balance_row(5)],  # ensure_wallet
        ],
        tx_rows=[
            [_balance_row(5)],  # FOR UPDATE
            [_balance_row(0)],  # UPDATE -> floored to 0 (5 - 20 => 0)
        ],
    )
    monkeypatch.setattr(wallet, "db", fake_db)

    result = await wallet.admin_adjust_tickets("u1", -20, admin_id="admin-9")

    # Applied delta reflects the floor, not the requested -20.
    assert result["delta"] == -5
    assert result["ticket_balance"] == 0
    metadata = json.loads(fake_db.fake_tx.execute_calls[0][1][-1])
    assert metadata["requested"] == -20
    assert metadata["applied"] == -5


@pytest.mark.asyncio
async def test_admin_adjust_zero_amount_rejected(monkeypatch):
    fake_db = _FakeDb()
    monkeypatch.setattr(wallet, "db", fake_db)
    with pytest.raises(ValueError, match="invalid_amount"):
        await wallet.admin_adjust_tickets("u1", 0, admin_id="a1")


@pytest.mark.asyncio
async def test_admin_adjust_rejects_amount_over_cap(monkeypatch):
    fake_db = _FakeDb()
    monkeypatch.setattr(wallet, "db", fake_db)
    with pytest.raises(ValueError, match="invalid_amount"):
        await wallet.admin_adjust_tickets(
            "u1", wallet.MAX_TICKET_ADJUST + 1, admin_id="a1"
        )


@pytest.mark.asyncio
async def test_admin_adjust_unknown_user_rejected(monkeypatch):
    fake_db = _FakeDb(query_rows=[[]])  # _ensure_user_exists -> empty
    monkeypatch.setattr(wallet, "db", fake_db)
    with pytest.raises(ValueError, match="user_not_found"):
        await wallet.admin_adjust_tickets("ghost", 5, admin_id="a1")


@pytest.mark.asyncio
async def test_admin_adjust_deduct_from_empty_is_no_change(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[
            [{"1": 1}],         # _ensure_user_exists
            [_balance_row(0)],  # ensure_wallet
        ],
        tx_rows=[
            [_balance_row(0)],  # FOR UPDATE -> already 0
        ],
    )
    monkeypatch.setattr(wallet, "db", fake_db)
    with pytest.raises(ValueError, match="no_change"):
        await wallet.admin_adjust_tickets("u1", -10, admin_id="a1")


@pytest.mark.asyncio
async def test_credit_tickets_updates_balance_and_ledger():
    tx = _FakeTx([[_balance_row(30)]])  # UPDATE ... RETURNING
    balance = await wallet.credit_tickets(
        "u1", 12, source="admin_grant", client=tx
    )
    assert balance["ticket_balance"] == 30
    assert any("INSERT INTO wallet_ledger" in sql for sql, _ in tx.execute_calls)
    # Ticket delta is positive.
    _, ledger_args = tx.execute_calls[0]
    assert 12 in ledger_args


@pytest.mark.asyncio
async def test_list_admin_balances_shape(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[
            [{"n": 2}],  # count
            [
                {
                    "id": "u1",
                    "username": "alice",
                    "display_name": "Alice",
                    "ticket_balance": 100,
                    "point_balance": 0,
                    "updated_at": "2026-08-20T08:00:00+00:00",
                    "nickname": None,
                },
                {
                    "id": "u2",
                    "username": "bob",
                    "display_name": None,
                    "ticket_balance": 0,
                    "point_balance": 5,
                    "updated_at": None,
                    "nickname": "小明",
                },
            ],
        ],
    )
    monkeypatch.setattr(wallet, "db", fake_db)

    result = await wallet.list_admin_balances(search="a", limit=20, offset=0)
    assert result["total"] == 2
    assert result["items"][0]["user_id"] == "u1"
    assert result["items"][0]["ticket_balance"] == 100
    assert result["items"][1]["nickname"] == "小明"


@pytest.mark.asyncio
async def test_list_admin_ticket_ledger_filters_ticket_currency(monkeypatch):
    fake_db = _FakeDb(
        query_rows=[
            [
                {
                    "id": "l1",
                    "user_id": "u1",
                    "username": "alice",
                    "display_name": "Alice",
                    "nickname": None,
                    "currency": "ticket",
                    "delta": 18,
                    "balance_after": 28,
                    "source": "admin_grant",
                    "source_id": None,
                    "metadata": json.dumps({"admin_id": "a1", "note": "补偿"}),
                    "created_at": "2026-08-20T08:00:00+00:00",
                }
            ]
        ]
    )
    monkeypatch.setattr(wallet, "db", fake_db)

    rows = await wallet.list_admin_ticket_ledger(user_id="u1", limit=20, offset=0)
    assert len(rows) == 1
    assert rows[0]["source"] == "admin_grant"
    assert rows[0]["metadata"]["admin_id"] == "a1"
    # user_id filter appends the currency + user_id predicate.
    ledger_sql = fake_db.query_calls[0][0]
    assert "l.currency = 'ticket'" in ledger_sql
    assert "l.user_id = $3" in ledger_sql
