from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.public import auth as auth_api
from app.services import wechat_auth
from app.services.wechat_auth import WeChatTokenPayload


class FakeRequest:
    headers = {}
    client = SimpleNamespace(host="1.2.3.4")


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class FakeAsyncClient:
    payload = {}

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, *, params):
        self.url = url
        self.params = params
        return FakeResponse(self.payload)


def test_wechat_login_request_strips_code():
    payload = auth_api.WeChatMobileLoginRequest(code="  abc  ", platform="ios")

    assert payload.code == "abc"


@pytest.mark.asyncio
async def test_exchange_wechat_code_requires_enabled_config(monkeypatch):
    monkeypatch.setattr(wechat_auth.settings, "wechat_login_enabled", False)

    with pytest.raises(HTTPException) as exc:
        await wechat_auth.exchange_wechat_code("code")

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_exchange_wechat_code_returns_identity_without_leaking_secret(monkeypatch):
    monkeypatch.setattr(wechat_auth.settings, "wechat_login_enabled", True)
    monkeypatch.setattr(wechat_auth.settings, "wechat_mobile_app_id", "wx-app")
    monkeypatch.setattr(wechat_auth.settings, "wechat_mobile_app_secret", "secret")
    FakeAsyncClient.payload = {
        "access_token": "token",
        "openid": "open-1",
        "unionid": "union-1",
        "scope": "snsapi_userinfo",
    }

    with patch.object(wechat_auth.httpx, "AsyncClient", FakeAsyncClient):
        payload = await wechat_auth.exchange_wechat_code("one-time-code")

    assert payload.openid == "open-1"
    assert payload.unionid == "union-1"
    assert payload.provider_account_id == "union-1"
    assert "access_token" not in payload.raw


@pytest.mark.asyncio
async def test_find_or_create_wechat_user_updates_openid_identity_to_unionid(monkeypatch):
    existing_identity = SimpleNamespace(id="identity-1", userId="user-1")
    existing_user = SimpleNamespace(id="user-1", username="wx_old", role="user")
    fake_db = SimpleNamespace(
        authidentity=SimpleNamespace(
            find_first=AsyncMock(return_value=existing_identity),
            update=AsyncMock(),
        ),
        user=SimpleNamespace(find_unique=AsyncMock(return_value=existing_user)),
    )
    monkeypatch.setattr(wechat_auth, "db", fake_db)

    user = await wechat_auth.find_or_create_wechat_user(
        WeChatTokenPayload(
            openid="open-1",
            unionid="union-1",
            scope="snsapi_userinfo",
            raw={"openid": "open-1", "unionid": "union-1"},
        )
    )

    assert user is existing_user
    fake_db.authidentity.find_first.assert_awaited_once()
    where = fake_db.authidentity.find_first.await_args.kwargs["where"]
    assert {"openid": "open-1"} in where["OR"]
    assert {"unionid": "union-1"} in where["OR"]
    update_data = fake_db.authidentity.update.await_args.kwargs["data"]
    assert update_data["providerAccountId"] == "union-1"


@pytest.mark.asyncio
async def test_wechat_mobile_login_returns_existing_auth_response(monkeypatch):
    request = auth_api.WeChatMobileLoginRequest(code="code", platform="ios")
    user = SimpleNamespace(id="user-1", username="wx_user", role="user")
    expected = auth_api.AuthResponse(
        token="jwt",
        user_id="user-1",
        username="wx_user",
        role="user",
        has_agent=False,
    )

    monkeypatch.setattr(auth_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(auth_api, "clear_login_failures", AsyncMock())
    monkeypatch.setattr(auth_api, "exchange_wechat_code", AsyncMock(return_value="payload"))
    monkeypatch.setattr(auth_api, "find_or_create_wechat_user", AsyncMock(return_value=user))
    monkeypatch.setattr(auth_api, "create_jwt", lambda user_id, role: "jwt")
    monkeypatch.setattr(auth_api, "_record_auth_activity", AsyncMock())
    monkeypatch.setattr(auth_api, "_build_auth_response", AsyncMock(return_value=expected))

    response = await auth_api.wechat_mobile_login(request, FakeRequest())

    assert response == expected
    auth_api.enforce_login_rate_limit.assert_awaited_once()
    auth_api.clear_login_failures.assert_awaited_once()
    auth_api._record_auth_activity.assert_awaited_once_with("user-1", source="wechat_login")


@pytest.mark.asyncio
async def test_password_login_rejects_social_user_without_hash(monkeypatch):
    social_user = SimpleNamespace(
        id="user-1",
        username="wx_user",
        role="user",
        hashedPassword=None,
    )
    monkeypatch.setattr(
        auth_api.db,
        "user",
        SimpleNamespace(find_unique=AsyncMock(return_value=social_user)),
    )
    monkeypatch.setattr(auth_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(auth_api, "record_login_failure", AsyncMock())

    with pytest.raises(HTTPException) as exc:
        await auth_api.login(auth_api.LoginRequest(username="wx_user", password="x"), FakeRequest())

    assert exc.value.status_code == 401
    auth_api.record_login_failure.assert_awaited_once()
