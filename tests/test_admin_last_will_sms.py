"""Admin last-will SMS smoke test."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api.admin import last_wills as admin_last_wills_api
from app.api.admin.last_wills import LastWillSmsTestRequest


@pytest.mark.asyncio
async def test_admin_last_will_sms_test_happy_path(monkeypatch):
    monkeypatch.setattr(
        admin_last_wills_api,
        "send_test_last_will_sms",
        AsyncMock(return_value={"phone_tail": "5678", "mock": False}),
    )

    response = await admin_last_wills_api.test_last_will_sms(
        LastWillSmsTestRequest(phone="13812345678")
    )

    assert response.ok is True
    assert response.phone_tail == "5678"
    assert response.mock is False


@pytest.mark.asyncio
async def test_admin_last_will_sms_test_invalid_phone(monkeypatch):
    monkeypatch.setattr(
        admin_last_wills_api,
        "send_test_last_will_sms",
        AsyncMock(side_effect=ValueError("invalid_phone")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_last_wills_api.test_last_will_sms(
            LastWillSmsTestRequest(phone="00000000000")
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_admin_last_will_sms_test_not_configured(monkeypatch):
    monkeypatch.setattr(
        admin_last_wills_api,
        "send_test_last_will_sms",
        AsyncMock(side_effect=RuntimeError("sms_not_configured")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_last_wills_api.test_last_will_sms(
            LastWillSmsTestRequest(phone="13812345678")
        )

    assert exc_info.value.status_code == 503
