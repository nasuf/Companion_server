from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from prisma import Json

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


class FakeTransaction:
    def __init__(self):
        self.created_identity_data = None
        self.user = SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(id="user-new", username="wx_new", role="user")
            )
        )
        self.authidentity = SimpleNamespace(create=AsyncMock(side_effect=self._create_identity))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def _create_identity(self, *, data):
        self.created_identity_data = data
        return SimpleNamespace(id="identity-new")


@pytest.mark.asyncio
async def test_wechat_profile_avatar_upgraded_to_https(monkeypatch):
    """qlogo.cn 常回 http:// — 读取侧必须升级, 否则 https 页面 mixed-content 拦截."""
    identity = SimpleNamespace(
        rawProfile={"nickname": "小明", "headimgurl": "http://thirdwx.qlogo.cn/mmopen/x/132"}
    )
    monkeypatch.setattr(
        auth_api,
        "db",
        SimpleNamespace(
            authidentity=SimpleNamespace(find_first=AsyncMock(return_value=identity))
        ),
    )

    name, avatar = await auth_api._wechat_profile_for_user("user-1")

    assert name == "小明"
    assert avatar == "https://thirdwx.qlogo.cn/mmopen/x/132"


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
        "nickname": "小伴",
        "headimgurl": "https://avatar",
        "sex": 1,
        "province": "Guangdong",
        "city": "Shenzhen",
        "country": "CN",
        "privilege": ["tester"],
    }

    with patch.object(wechat_auth.httpx, "AsyncClient", FakeAsyncClient):
        payload = await wechat_auth.exchange_wechat_code("one-time-code")

    assert payload.openid == "open-1"
    assert payload.unionid == "union-1"
    assert payload.provider_account_id == "union-1"
    assert payload.raw["nickname"] == "小伴"
    assert payload.raw["headimgurl"] == "https://avatar"
    assert payload.raw["sex"] == 1
    assert payload.raw["province"] == "Guangdong"
    assert payload.raw["city"] == "Shenzhen"
    assert payload.raw["country"] == "CN"
    assert payload.raw["privilege"] == ["tester"]
    assert payload.raw["scope"] == "snsapi_userinfo"
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
    # Priority lookup: the unionid match is checked first and wins immediately.
    fake_db.authidentity.find_first.assert_awaited_once()
    where = fake_db.authidentity.find_first.await_args.kwargs["where"]
    assert where == {"provider": "wechat", "unionid": "union-1"}
    update_data = fake_db.authidentity.update.await_args.kwargs["data"]
    assert update_data["providerAccountId"] == "union-1"
    assert isinstance(update_data["rawProfile"], Json)


@pytest.mark.asyncio
async def test_identity_lookup_priority_falls_back_union_account_openid(monkeypatch):
    """unionid > providerAccountId > openid — the openid-only duplicate is the
    last resort, so post-binding logins deterministically land on the
    canonical (unionid) account when it exists."""
    duplicate_identity = SimpleNamespace(id="identity-dup", userId="user-dup")
    fake_db = SimpleNamespace(
        authidentity=SimpleNamespace(
            find_first=AsyncMock(side_effect=[None, None, duplicate_identity]),
            update=AsyncMock(),
        ),
        user=SimpleNamespace(
            find_unique=AsyncMock(
                return_value=SimpleNamespace(id="user-dup", username="wx_dup", role="user")
            )
        ),
    )
    monkeypatch.setattr(wechat_auth, "db", fake_db)

    user = await wechat_auth.find_or_create_wechat_user(
        WeChatTokenPayload(
            openid="open-mini",
            unionid="union-1",
            scope="miniprogram",
            raw={"openid": "open-mini", "unionid": "union-1", "source": "miniprogram"},
        )
    )

    assert user.id == "user-dup"
    wheres = [c.kwargs["where"] for c in fake_db.authidentity.find_first.await_args_list]
    assert wheres == [
        {"provider": "wechat", "unionid": "union-1"},
        {"provider": "wechat", "providerAccountId": "union-1"},
        {"provider": "wechat", "openid": "open-mini"},
    ]


@pytest.mark.asyncio
async def test_find_or_create_wechat_user_creates_identity_with_relation_connect(monkeypatch):
    tx = FakeTransaction()
    fake_db = SimpleNamespace(
        authidentity=SimpleNamespace(find_first=AsyncMock(return_value=None)),
        tx=lambda: tx,
    )
    monkeypatch.setattr(wechat_auth, "db", fake_db)

    user = await wechat_auth.find_or_create_wechat_user(
        WeChatTokenPayload(
            openid="open-1",
            unionid=None,
            scope="snsapi_userinfo",
            raw={"openid": "open-1", "nickname": "小伴", "headimgurl": "https://avatar"},
        )
    )

    assert user.id == "user-new"
    tx.user.create.assert_awaited_once()
    tx.authidentity.create.assert_awaited_once()
    data = tx.created_identity_data
    assert data["user"] == {"connect": {"id": "user-new"}}
    assert "userId" not in data
    assert isinstance(data["rawProfile"], Json)


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


class TestMergedRawProfile:
    """Regression: re-login must not wipe the saved 头像昵称 profile fields."""

    def _token(self, raw: dict) -> WeChatTokenPayload:
        return WeChatTokenPayload(
            openid="openid-1", unionid="union-1", scope="miniprogram", raw=raw,
        )

    def test_miniprogram_relogin_preserves_saved_nickname_and_avatar(self):
        existing = {
            "openid": "openid-1",
            "nickname": "小明",
            "headimgurl": "/chat/media/u_abc.jpg",
        }
        token = self._token({"openid": "openid-1", "unionid": None, "source": "miniprogram"})
        merged = wechat_auth._merged_raw_profile(existing, token)
        assert merged["nickname"] == "小明"
        assert merged["headimgurl"] == "/chat/media/u_abc.jpg"
        assert merged["source"] == "miniprogram"

    def test_fresh_values_from_token_win(self):
        existing = {"nickname": "旧名", "headimgurl": "old.jpg"}
        token = self._token({
            "openid": "openid-1",
            "nickname": "新名",
            "headimgurl": "new.jpg",
        })
        merged = wechat_auth._merged_raw_profile(existing, token)
        assert merged["nickname"] == "新名"
        assert merged["headimgurl"] == "new.jpg"

    def test_handles_missing_existing_profile(self):
        token = self._token({"openid": "openid-1", "source": "miniprogram"})
        merged = wechat_auth._merged_raw_profile(None, token)
        assert merged == {"openid": "openid-1", "source": "miniprogram"}
