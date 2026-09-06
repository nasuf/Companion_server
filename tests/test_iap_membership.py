"""GET /me/iap/membership — user VIP + subscription + purchase history."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.payments import membership


def _vip_snapshot(*, is_vip: bool = True):
    return {
        "is_vip": is_vip,
        "vip_until": "2027-01-08T10:07:40+00:00" if is_vip else None,
        "vip_trial_available": False,
        "gift_ticket_balance": 0,
        "ticket_balance": 100,
        "point_balance": 0,
        "spendable_tickets": 100,
    }


@pytest.mark.asyncio
async def test_get_membership_composes_subscription_and_history(monkeypatch):
    sub_row = {
        "product_id": "com.bansheng.vip.monthly.auto",
        "status": "active",
        "auto_renew_status": True,
        "auto_renew_product_id": "com.bansheng.vip.monthly.auto",
        "expires_date": datetime(2027, 2, 1, tzinfo=timezone.utc),
        "grace_period_expires_date": None,
        "updated_at": datetime(2026, 9, 6, tzinfo=timezone.utc),
    }
    history_row = {
        "transaction_id": "t-month",
        "product_id": "com.bansheng.vip.month",
        "kind": "consumable",
        "status": "granted",
        "purchase_date": datetime(2026, 9, 6, 10, 7, 40, tzinfo=timezone.utc),
        "expires_date": None,
    }

    async def fake_query(query: str, *args):
        if "iap_subscription_state" in query:
            return [sub_row]
        if "iap_transactions" in query:
            return [history_row]
        return []

    monkeypatch.setattr(membership.grant, "reconcile_vip_entitlements", AsyncMock(return_value=False))
    monkeypatch.setattr(membership.wallet, "full_wallet", AsyncMock(return_value=_vip_snapshot()))
    monkeypatch.setattr(membership.db, "query_raw", fake_query)

    result = await membership.get_membership("u1")

    assert result["auto_renew_active"] is True
    assert result["subscription"]["product_label"] == "连续包月"
    assert len(result["history"]) == 1
    assert result["history"][0]["product_label"] == "月卡"


@pytest.mark.asyncio
async def test_get_membership_no_subscription(monkeypatch):
    monkeypatch.setattr(membership.grant, "reconcile_vip_entitlements", AsyncMock(return_value=False))
    monkeypatch.setattr(membership.wallet, "full_wallet", AsyncMock(return_value=_vip_snapshot(is_vip=False)))
    monkeypatch.setattr(membership.db, "query_raw", AsyncMock(return_value=[]))

    result = await membership.get_membership("u1")

    assert result["subscription"] is None
    assert result["auto_renew_active"] is False
    assert result["history"] == []


def test_membership_endpoint_requires_auth(api_client):
    resp = api_client.get("/me/iap/membership")
    assert resp.status_code == 401


def test_membership_endpoint_ok(api_client, auth_header):
    payload = {
        "vip": _vip_snapshot(),
        "subscription": None,
        "auto_renew_active": False,
        "history": [],
    }
    with patch(
        "app.api.public.iap_membership.membership.get_membership",
        new=AsyncMock(return_value=payload),
    ):
        resp = api_client.get("/me/iap/membership", headers=auth_header("u1"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_renew_active"] is False
    assert body["vip"]["is_vip"] is True
