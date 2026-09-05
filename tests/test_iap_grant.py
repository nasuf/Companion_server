"""IAP 到账服务（grant.py）：幂等、消耗型到账、订阅激活。

DB 用最小 fake（tx()/query_raw/execute_raw），Apple 校验/钱包/VIP 发放在
seam 处 monkeypatch，不碰真实网络与库。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.payments import grant


class _FakeTx:
    def __init__(self, db: "_FakeDb"):
        self._db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query_raw(self, query: str, *args):
        return self._db._query(query, args)

    async def execute_raw(self, query: str, *args):
        return self._db._execute(query, args)


class _FakeDb:
    def __init__(self, *, existing=None, insert_returns=None, locked_status="granted"):
        self.existing = existing  # _find_transaction 结果行 or None
        self.insert_returns = [{"id": "row1"}] if insert_returns is None else insert_returns
        self.locked_status = locked_status
        self.vip_until_written = "unset"
        self.calls: list[tuple[str, str]] = []

    def tx(self):
        return _FakeTx(self)

    async def query_raw(self, query: str, *args):
        return self._query(query, args)

    async def execute_raw(self, query: str, *args):
        return self._execute(query, args)

    def _query(self, query: str, args):
        self.calls.append(("q", query))
        if "SELECT status, kind, product_id, user_id" in query:
            return [self.existing] if self.existing else []
        if "INSERT INTO iap_transactions" in query:
            return list(self.insert_returns)
        if "SELECT status FROM iap_transactions" in query and "FOR UPDATE" in query:
            return [{"status": self.locked_status}]
        if "SELECT vip_until FROM user_wallets" in query:
            return [{"vip_until": None}]
        return []

    def _execute(self, query: str, args):
        self.calls.append(("e", query))
        if "UPDATE user_wallets" in query and "vip_until" in query:
            self.vip_until_written = args[1]
        return 1


def _payload(product_id: str, *, txn="t1", otxn="t1", expires_ms=None, quantity=1):
    return SimpleNamespace(
        productId=product_id,
        transactionId=txn,
        originalTransactionId=otxn,
        webOrderLineItemId=None,
        expiresDate=expires_ms,
        purchaseDate=1_700_000_000_000,
        quantity=quantity,
        type=SimpleNamespace(value="Consumable"),
        environment="Sandbox",
    )


def _wire(monkeypatch, fake_db, *, fetch_payload):
    monkeypatch.setattr(grant, "db", fake_db)
    monkeypatch.setattr(
        grant.apple_env,
        "fetch_and_verify_transaction",
        AsyncMock(return_value=(fetch_payload, "Sandbox")),
    )
    monkeypatch.setattr(grant.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(
        grant.wallet,
        "get_balance",
        AsyncMock(return_value={"ticket_balance": 10, "point_balance": 0, "achievement_points_synced": 0}),
    )
    monkeypatch.setattr(
        grant.wallet,
        "full_wallet",
        AsyncMock(
            return_value={
                "is_vip": True,
                "vip_until": None,
                "vip_trial_available": False,
                "gift_ticket_balance": 0,
                "ticket_balance": 10,
                "point_balance": 0,
                "spendable_tickets": 10,
            }
        ),
    )
    monkeypatch.setattr(grant.vip_grants, "grant_monthly", AsyncMock())


@pytest.mark.asyncio
async def test_consumable_grant_credits_tickets(monkeypatch):
    fake = _FakeDb(existing=None)
    payload = _payload("com.bansheng.companion.ticket.80", txn="txn-abc")
    _wire(monkeypatch, fake, fetch_payload=payload)
    credit = AsyncMock(return_value={"ticket_balance": 90, "point_balance": 0, "achievement_points_synced": 0})
    monkeypatch.setattr(grant.wallet, "credit_tickets", credit)

    result = await grant.verify_and_grant("u1", "txn-abc")

    assert result["status"] == "granted"
    assert result["kind"] == "consumable"
    credit.assert_awaited_once()
    args, kwargs = credit.call_args
    assert args[0] == "u1" and args[1] == 80  # ticket_amount * quantity
    assert kwargs["source"] == "iap_apple"
    assert kwargs["source_id"] == "txn-abc"
    # VIP 月度发放不该在纯充值时触发
    grant.vip_grants.grant_monthly.assert_not_awaited()


@pytest.mark.asyncio
async def test_grant_is_idempotent_fast_path(monkeypatch):
    fake = _FakeDb(existing={"status": "granted", "kind": "consumable", "product_id": "x", "user_id": "u1"})
    # fetch 不该被调用（已 granted 直接回放）
    _wire(monkeypatch, fake, fetch_payload=_payload("com.bansheng.companion.ticket.10"))
    credit = AsyncMock()
    monkeypatch.setattr(grant.wallet, "credit_tickets", credit)

    result = await grant.verify_and_grant("u1", "txn-dup")

    assert result["status"] == "granted"
    grant.apple_env.fetch_and_verify_transaction.assert_not_awaited()
    credit.assert_not_awaited()


@pytest.mark.asyncio
async def test_grant_conflict_replays_without_double_credit(monkeypatch):
    # INSERT 命中 ON CONFLICT（并发/重放）→ 锁行读到 granted → 回放，不重复到账
    fake = _FakeDb(existing=None, insert_returns=[], locked_status="granted")
    _wire(monkeypatch, fake, fetch_payload=_payload("com.bansheng.companion.ticket.10", txn="txn-x"))
    credit = AsyncMock()
    monkeypatch.setattr(grant.wallet, "credit_tickets", credit)

    result = await grant.verify_and_grant("u1", "txn-x")

    assert result["status"] == "granted"
    credit.assert_not_awaited()


@pytest.mark.asyncio
async def test_refunded_transaction_is_not_regranted(monkeypatch):
    # 已退款的交易被再次 verify（INSERT 命中冲突，锁行读到 refunded）→ 绝不二次到账
    fake = _FakeDb(existing=None, insert_returns=[], locked_status="refunded")
    _wire(monkeypatch, fake, fetch_payload=_payload("com.bansheng.companion.ticket.10", txn="txn-r"))
    credit = AsyncMock()
    monkeypatch.setattr(grant.wallet, "credit_tickets", credit)

    result = await grant.verify_and_grant("u1", "txn-r")

    assert result["status"] == "refunded"
    credit.assert_not_awaited()


@pytest.mark.asyncio
async def test_subscription_activation_sets_vip_until_from_apple(monkeypatch):
    expires = datetime(2027, 1, 1, tzinfo=timezone.utc)
    expires_ms = int(expires.timestamp() * 1000)
    fake = _FakeDb(existing=None)
    payload = _payload("com.bansheng.companion.vip.monthly.auto", txn="sub-1", expires_ms=expires_ms)
    _wire(monkeypatch, fake, fetch_payload=payload)
    monkeypatch.setattr(grant.wallet, "credit_tickets", AsyncMock())

    result = await grant.verify_and_grant("u1", "sub-1")

    assert result["kind"] == "subscription"
    # vip_until 以 Apple expires_date 为准（存 naive UTC）
    assert fake.vip_until_written == expires.replace(tzinfo=None)
    grant.vip_grants.grant_monthly.assert_awaited_once()


@pytest.mark.asyncio
async def test_unknown_product_raises(monkeypatch):
    from app.services.payments.errors import UnknownProductError

    fake = _FakeDb(existing=None)
    _wire(monkeypatch, fake, fetch_payload=_payload("com.bansheng.companion.NOPE"))

    with pytest.raises(UnknownProductError):
        await grant.verify_and_grant("u1", "txn-unknown")
