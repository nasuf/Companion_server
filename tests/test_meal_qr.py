from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import meal_qr


@pytest.mark.asyncio
async def test_issue_creates_prefixed_short_lived_grant(monkeypatch):
    redis = SimpleNamespace(eval=AsyncMock(return_value=1))
    monkeypatch.setattr(meal_qr, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(meal_qr.settings, "meal_qr_ttl_seconds", 60)

    result = await meal_qr.issue("voucher-1", "user-1", "activate")

    assert result["value"].startswith(meal_qr.QR_PREFIX)
    assert result["expires_in"] == 60
    assert result["action"] == "activate"
    args = redis.eval.await_args.args
    assert args[1] == 2
    assert "voucher-1" in args[2]


@pytest.mark.asyncio
async def test_consume_is_atomic_and_returns_bound_identity(monkeypatch):
    value = f"{meal_qr.QR_PREFIX}{'a' * 43}"
    raw = json.dumps(
        {"voucher_id": "voucher-1", "user_id": "user-1", "action": "redeem"}
    )
    redis = SimpleNamespace(
        get=AsyncMock(return_value=raw),
        eval=AsyncMock(return_value=raw),
    )
    monkeypatch.setattr(meal_qr, "get_redis", AsyncMock(return_value=redis))

    result = await meal_qr.consume(value, "redeem")

    assert result == {
        "voucher_id": "voucher-1",
        "user_id": "user-1",
        "action": "redeem",
    }
    redis.eval.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_rejects_foreign_and_expired_qr(monkeypatch):
    with pytest.raises(meal_qr.MealQRError) as invalid:
        await meal_qr.consume("https://example.com/not-a-meal-code", "redeem")
    assert invalid.value.reason == "invalid_qr"

    redis = SimpleNamespace(get=AsyncMock(return_value=None))
    monkeypatch.setattr(meal_qr, "get_redis", AsyncMock(return_value=redis))
    value = f"{meal_qr.QR_PREFIX}{'b' * 43}"
    with pytest.raises(meal_qr.MealQRError) as expired:
        await meal_qr.consume(value, "redeem")
    assert expired.value.reason == "expired_qr"


@pytest.mark.asyncio
async def test_public_qr_issue_uses_voucher_stage(monkeypatch):
    from app.api.public import meal as meal_api

    voucher = SimpleNamespace(id="v-1", userId="user-1", status="inactive")
    monkeypatch.setattr(
        meal_api.mv, "get_or_create_voucher", AsyncMock(return_value=voucher)
    )
    monkeypatch.setattr(meal_api.mv, "is_code_enabled", AsyncMock(return_value=True))
    issue = AsyncMock(return_value={"action": "activate"})
    monkeypatch.setattr(meal_api.meal_qr, "issue", issue)

    result = await meal_api.voucher_qr_token(payload={"sub": "user-1"})
    issue.assert_awaited_once_with("v-1", "user-1", "activate")
    assert result["action"] == "activate"


@pytest.mark.asyncio
async def test_public_inactive_qr_respects_validation_toggle(monkeypatch):
    from fastapi import HTTPException

    from app.api.public import meal as meal_api

    voucher = SimpleNamespace(id="v-1", userId="user-1", status="inactive")
    monkeypatch.setattr(
        meal_api.mv, "get_or_create_voucher", AsyncMock(return_value=voucher)
    )
    monkeypatch.setattr(meal_api.mv, "is_code_enabled", AsyncMock(return_value=False))

    with pytest.raises(HTTPException) as exc:
        await meal_api.voucher_qr_token(payload={"sub": "user-1"})
    assert exc.value.status_code == 400
    assert "暂未开放" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_consume_wrong_stage_does_not_burn_token(monkeypatch):
    value = f"{meal_qr.QR_PREFIX}{'c' * 43}"
    raw = json.dumps(
        {"voucher_id": "voucher-1", "user_id": "user-1", "action": "activate"}
    )
    redis = SimpleNamespace(get=AsyncMock(return_value=raw), eval=AsyncMock())
    monkeypatch.setattr(meal_qr, "get_redis", AsyncMock(return_value=redis))

    with pytest.raises(meal_qr.MealQRError) as exc:
        await meal_qr.consume(value, "redeem")
    assert exc.value.reason == "wrong_qr_stage"
    redis.eval.assert_not_awaited()


@pytest.mark.asyncio
async def test_merchant_scan_endpoint_uses_merchant_session(monkeypatch):
    from datetime import UTC, datetime

    from app.api.public import meal as meal_api

    merchant = SimpleNamespace(id="m-1", name="伴生宴", codeActive=True)
    voucher = SimpleNamespace(
        id="v-1",
        userId="user-1",
        redeemedAt=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    monkeypatch.setattr(meal_api, "_require_merchant", lambda _: "m-1")
    monkeypatch.setattr(
        meal_api,
        "db",
        SimpleNamespace(
            mealmerchant=SimpleNamespace(find_unique=AsyncMock(return_value=merchant))
        ),
    )
    monkeypatch.setattr(
        meal_api.meal_qr,
        "consume",
        AsyncMock(return_value={"voucher_id": "v-1", "user_id": "user-1"}),
    )
    redeem = AsyncMock(return_value=voucher)
    monkeypatch.setattr(meal_api.mv, "redeem_voucher_by_merchant", redeem)
    monkeypatch.setattr(
        meal_api.mv, "resolve_user_display", AsyncMock(return_value="小明")
    )

    result = await meal_api.merchant_redeem_scan(
        meal_api.MealScanRequest(value=f"{meal_qr.QR_PREFIX}{'a' * 43}"),
        SimpleNamespace(),
    )

    redeem.assert_awaited_once_with("v-1", "user-1", "m-1")
    meal_api.meal_qr.consume.assert_awaited_once_with(
        f"{meal_qr.QR_PREFIX}{'a' * 43}", "redeem"
    )
    assert result["user_display"] == "小明"
