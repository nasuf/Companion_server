"""App Store Server Notifications V2 处理：续期到账、退款清算、幂等短路。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from appstoreserverlibrary.models.NotificationTypeV2 import NotificationTypeV2

from app.services.payments import notifications


class _FakeDb:
    def __init__(self, *, insert_new=True, find_txn=None, user_row=None):
        self.insert_new = insert_new
        self.find_txn = find_txn
        self.user_row = user_row
        self.executed: list[str] = []

    def tx(self):
        db = self

        class _Tx:
            async def __aenter__(self_):
                return self_

            async def __aexit__(self_, *exc):
                return False

            async def query_raw(self_, q, *a):
                return db._query(q, a)

            async def execute_raw(self_, q, *a):
                db.executed.append(q)
                return 1

        return _Tx()

    async def query_raw(self, q, *a):
        return self._query(q, a)

    async def execute_raw(self, q, *a):
        self.executed.append(q)
        return 1

    def _query(self, q, a):
        if "INSERT INTO iap_notifications" in q:
            return [{"id": "n1"}] if self.insert_new else []
        if "SELECT status, kind, product_id, user_id" in q:
            return [self.find_txn] if self.find_txn else []
        if "SELECT user_id FROM iap_subscription_state" in q:
            return [self.user_row] if self.user_row else []
        if "SELECT user_id FROM iap_transactions" in q:
            return [self.user_row] if self.user_row else []
        if "SELECT ticket_balance FROM user_wallets" in q:
            return [{"ticket_balance": 100}]
        if q.strip().startswith("UPDATE user_wallets"):
            return [{"ticket_balance": 20, "point_balance": 0, "achievement_points_synced": 0}]
        return []


def _decoded(ntype, *, subtype=None, txn_jws="txn", renewal_jws=None):
    data = SimpleNamespace(signedTransactionInfo=txn_jws, signedRenewalInfo=renewal_jws)
    return SimpleNamespace(
        notificationUUID="uuid-1",
        notificationType=ntype,
        subtype=subtype,
        data=data,
    )


def _txn(product_id, *, txn="t1", otxn="t1"):
    return SimpleNamespace(
        productId=product_id, transactionId=txn, originalTransactionId=otxn
    )


def _wire(monkeypatch, fake_db, decoded, txn_payload, renewal=None):
    monkeypatch.setattr(notifications, "db", fake_db)
    # _handle_refund 复用 grant._find_transaction，其内部用 grant.db。
    monkeypatch.setattr(notifications.grant, "db", fake_db)
    monkeypatch.setattr(
        notifications.apple_env, "verify_notification", AsyncMock(return_value=(decoded, "Sandbox"))
    )
    monkeypatch.setattr(
        notifications.apple_env, "verify_signed_transaction", lambda jws, env: txn_payload
    )
    monkeypatch.setattr(
        notifications.apple_env, "verify_renewal_info", lambda jws, env: renewal
    )


@pytest.mark.asyncio
async def test_did_renew_grants_via_record_and_grant(monkeypatch):
    fake = _FakeDb(insert_new=True, user_row={"user_id": "u1"})
    decoded = _decoded(NotificationTypeV2.DID_RENEW)
    _wire(monkeypatch, fake, decoded, _txn("com.bansheng.vip.monthly.auto"))
    rg = AsyncMock()
    monkeypatch.setattr(notifications.grant, "record_and_grant", rg)

    await notifications.apply_notification("signed")

    rg.assert_awaited_once()
    assert rg.call_args.args[0] == "u1"


@pytest.mark.asyncio
async def test_duplicate_notification_short_circuits(monkeypatch):
    fake = _FakeDb(insert_new=False, user_row={"user_id": "u1"})
    decoded = _decoded(NotificationTypeV2.DID_RENEW)
    _wire(monkeypatch, fake, decoded, _txn("com.bansheng.vip.monthly.auto"))
    rg = AsyncMock()
    monkeypatch.setattr(notifications.grant, "record_and_grant", rg)

    await notifications.apply_notification("signed")

    # 已收到过该 notificationUUID → 不再分派
    rg.assert_not_awaited()


@pytest.mark.asyncio
async def test_refund_consumable_reverses_and_marks(monkeypatch):
    fake = _FakeDb(
        insert_new=True,
        find_txn={"status": "granted", "kind": "consumable", "product_id": "com.bansheng.ticket.80", "user_id": "u1"},
    )
    decoded = _decoded(NotificationTypeV2.REFUND)
    _wire(monkeypatch, fake, decoded, _txn("com.bansheng.ticket.80", txn="t-refund"))
    monkeypatch.setattr(notifications.wallet, "_record_ledger", AsyncMock())
    monkeypatch.setattr(
        notifications.wallet, "wallet_balances",
        lambda row: {"ticket_balance": 20, "point_balance": 0, "achievement_points_synced": 0},
    )

    await notifications.apply_notification("signed")

    assert any("status = 'refunded'" in q for q in fake.executed)


@pytest.mark.asyncio
async def test_expired_sets_state_without_touching_vip(monkeypatch):
    fake = _FakeDb(insert_new=True, user_row={"user_id": "u1"})
    decoded = _decoded(NotificationTypeV2.EXPIRED, txn_jws="txn")
    _wire(monkeypatch, fake, decoded, _txn("com.bansheng.vip.monthly.auto"))

    await notifications.apply_notification("signed")

    # 只更新订阅状态表，不 UPDATE user_wallets 的 vip_until
    assert any("iap_subscription_state" in q for q in fake.executed)
    assert not any("UPDATE user_wallets" in q and "vip_until" in q for q in fake.executed)
