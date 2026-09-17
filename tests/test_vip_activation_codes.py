"""VIP activation codes: entitlement stacking, redeem, revoke, IAP coexistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.services.payments import catalog, grant
from app.services.vip import entitlements
from app.services.vip.activation_codes.errors import VipActivationError
from app.services.vip.activation_codes import redeem as redeem_mod
from app.services.vip.activation_codes import admin as admin_mod


class _FakeTx:
    def __init__(self, db: "_EntitlementFakeDb"):
        self._db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query_raw(self, query: str, *args):
        return self._db._query(query, args)

    async def execute_raw(self, query: str, *args):
        return self._db._execute(query, args)


class _EntitlementFakeDb:
    def __init__(self):
        self.vip_until: datetime | None = None
        self.iap_rows: list[dict[str, Any]] = []
        self.code_rows: list[dict[str, Any]] = []
        self.redemptions: list[dict[str, Any]] = []

    def tx(self):
        return _FakeTx(self)

    async def query_raw(self, query: str, *args):
        return self._query(query, args)

    async def execute_raw(self, query: str, *args):
        return self._execute(query, args)

    def _query(self, query: str, args):
        if "FROM vip_activation_codes" in query and "FOR UPDATE" in query:
            cid = str(args[0])
            row = next((r for r in self.code_rows if r["id"] == cid), None)
            return [row] if row else []
        if "FROM vip_activation_codes" in query and "WHERE code =" in query:
            code = str(args[0])
            row = next((r for r in self.code_rows if r["code"] == code), None)
            return [row] if row else []
        if "FROM vip_code_redemptions" in query and "code_id" in query and "user_id" in query:
            if len(args) >= 3 and args[2] == entitlements.REDEMPTION_GRANTED:
                cid, uid = str(args[0]), str(args[1])
                hit = [
                    r
                    for r in self.redemptions
                    if r["code_id"] == cid
                    and r["user_id"] == uid
                    and r["status"] == entitlements.REDEMPTION_GRANTED
                ]
                return hit[:1]
        if "FROM vip_code_redemptions" in query and "ORDER BY redeemed_at" in query:
            uid = str(args[0])
            rows = [
                r
                for r in self.redemptions
                if r["user_id"] == uid and r["status"] == entitlements.REDEMPTION_GRANTED
            ]
            rows.sort(key=lambda r: r["redeemed_at"])
            return rows
        if "FROM iap_transactions" in query and catalog.KIND_CONSUMABLE in str(args):
            return [
                {
                    "product_id": r["product_id"],
                    "quantity": r["quantity"],
                    "purchase_date": r["purchase_date"],
                }
                for r in self.iap_rows
                if r["status"] == "granted" and r["kind"] == catalog.KIND_CONSUMABLE
            ]
        if "MAX(expires_date)" in query and catalog.KIND_SUBSCRIPTION in str(args):
            expires = [
                r["expires_date"]
                for r in self.iap_rows
                if r["status"] == "granted" and r["kind"] == catalog.KIND_SUBSCRIPTION
            ]
            return [{"max_expires": max(expires) if expires else None}]
        if "FROM iap_subscription_state" in query:
            return []
        if "SELECT vip_until FROM user_wallets" in query:
            return [{"vip_until": self.vip_until}]
        if "INSERT INTO vip_code_redemptions" in query:
            rid = f"red-{len(self.redemptions)+1}"
            self.redemptions.append(
                {
                    "id": rid,
                    "code_id": str(args[0]),
                    "user_id": str(args[1]),
                    "duration_days": int(args[2]),
                    "status": entitlements.REDEMPTION_GRANTED,
                    "redeemed_at": datetime.now(timezone.utc),
                }
            )
            return [{"id": rid}]
        if "SELECT r.id, r.user_id, r.status, r.code_id" in query and "FOR UPDATE" in query:
            rid = str(args[0])
            row = next((r for r in self.redemptions if r["id"] == rid), None)
            return [row] if row else []
        return []

    def _execute(self, query: str, args):
        if "UPDATE user_wallets" in query and "vip_until" in query:
            self.vip_until = entitlements.as_utc(args[1])
        if "UPDATE vip_activation_codes" in query and "redemption_count = redemption_count + 1" in query:
            cid = str(args[0])
            for row in self.code_rows:
                if row["id"] == cid:
                    row["redemption_count"] = int(row.get("redemption_count", 0)) + 1
        if "UPDATE vip_activation_codes" in query and "GREATEST(redemption_count - 1" in query:
            cid = str(args[0])
            for row in self.code_rows:
                if row["id"] == cid:
                    row["redemption_count"] = max(int(row.get("redemption_count", 0)) - 1, 0)
        if "UPDATE vip_code_redemptions" in query and "revoked" in query.lower():
            rid = str(args[0])
            for row in self.redemptions:
                if row["id"] == rid:
                    row["status"] = entitlements.REDEMPTION_REVOKED
        return 1


def _add_code(fake: _EntitlementFakeDb, *, days: int = 7, max_red: int | None = 1) -> dict:
    row = {
        "id": f"code-{len(fake.code_rows)+1}",
        "code": f"VIPTEST{len(fake.code_rows)+1}",
        "duration_days": days,
        "max_redemptions": max_red,
        "redemption_count": 0,
        "enabled": True,
        "valid_from": None,
        "valid_until": None,
        "note": None,
        "created_by": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    fake.code_rows.append(row)
    return row


@pytest.mark.asyncio
async def test_code_vip_stacks_after_paid_subscription(monkeypatch):
    fake = _EntitlementFakeDb()
    monkeypatch.setattr(entitlements, "db", fake)
    now = datetime.now(timezone.utc)
    sub_end = now + timedelta(days=30)
    fake.iap_rows.append(
        {
            "product_id": "com.bansheng.vip.monthly.auto",
            "kind": catalog.KIND_SUBSCRIPTION,
            "status": "granted",
            "expires_date": sub_end,
            "quantity": 1,
            "purchase_date": now,
        }
    )
    fake.redemptions.append(
        {
            "id": "r1",
            "code_id": "c1",
            "user_id": "u1",
            "duration_days": 14,
            "status": entitlements.REDEMPTION_GRANTED,
            "redeemed_at": now,
        }
    )
    end = await entitlements.compute_vip_entitlement_end("u1")
    assert end is not None
    assert end == sub_end + timedelta(days=14)


@pytest.mark.asyncio
async def test_redeem_code_grants_vip(monkeypatch):
    fake = _EntitlementFakeDb()
    monkeypatch.setattr(entitlements, "db", fake)
    monkeypatch.setattr(redeem_mod, "db", fake)
    code = _add_code(fake, days=7)
    monkeypatch.setattr(redeem_mod.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(redeem_mod.wallet, "is_vip", AsyncMock(return_value=False))
    monkeypatch.setattr(
        redeem_mod.wallet,
        "full_wallet",
        AsyncMock(
            return_value={
                "is_vip": True,
                "vip_until": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                "vip_trial_available": False,
                "gift_ticket_balance": 0,
                "ticket_balance": 0,
                "point_balance": 0,
                "spendable_tickets": 0,
            }
        ),
    )
    monkeypatch.setattr(redeem_mod, "fire_background", lambda c: None)

    result = await redeem_mod.redeem_code("u1", code["code"])
    assert result["redemption"]["duration_days"] == 7
    assert fake.vip_until is not None
    assert fake.vip_until > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_redeem_rejects_duplicate_user(monkeypatch):
    fake = _EntitlementFakeDb()
    code = _add_code(fake)
    fake.redemptions.append(
        {
            "id": "existing",
            "code_id": code["id"],
            "user_id": "u1",
            "duration_days": 7,
            "status": entitlements.REDEMPTION_GRANTED,
            "redeemed_at": datetime.now(timezone.utc),
        }
    )
    monkeypatch.setattr(entitlements, "db", fake)
    monkeypatch.setattr(redeem_mod, "db", fake)
    monkeypatch.setattr(redeem_mod.wallet, "ensure_wallet", AsyncMock())
    monkeypatch.setattr(redeem_mod.wallet, "is_vip", AsyncMock(return_value=True))

    with pytest.raises(VipActivationError) as exc:
        await redeem_mod.redeem_code("u1", code["code"])
    assert exc.value.code == "already_redeemed"


@pytest.mark.asyncio
async def test_revoke_redemption_recomputes_vip(monkeypatch):
    fake = _EntitlementFakeDb()
    now = datetime.now(timezone.utc)
    fake.redemptions.append(
        {
            "id": "red-1",
            "code_id": "c1",
            "user_id": "u1",
            "duration_days": 30,
            "status": entitlements.REDEMPTION_GRANTED,
            "redeemed_at": now,
        }
    )
    fake.vip_until = now + timedelta(days=30)
    monkeypatch.setattr(entitlements, "db", fake)
    monkeypatch.setattr(admin_mod, "db", fake)

    result = await admin_mod.revoke_redemption("red-1", admin_user_id="admin1")
    assert result["status"] == entitlements.REDEMPTION_REVOKED
    assert fake.redemptions[0]["status"] == entitlements.REDEMPTION_REVOKED
    assert fake.vip_until is not None
    assert fake.vip_until < datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_iap_grant_still_recomputes_with_codes_present(monkeypatch):
    """Regression: IAP grant path must include activation codes in recompute."""
    fake = _EntitlementFakeDb()
    now = datetime.now(timezone.utc)
    fake.redemptions.append(
        {
            "id": "r1",
            "code_id": "c1",
            "user_id": "u1",
            "duration_days": 10,
            "status": entitlements.REDEMPTION_GRANTED,
            "redeemed_at": now - timedelta(days=1),
        }
    )
    monkeypatch.setattr(grant, "db", fake)
    monkeypatch.setattr(entitlements, "db", fake)
    monkeypatch.setattr(grant, "fire_background", lambda c: None)

    end = await grant.compute_vip_entitlement_end("u1")
    assert end is not None
    assert end > now + timedelta(days=9)
