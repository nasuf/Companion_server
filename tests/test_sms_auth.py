"""SMS login/binding: phone normalization, code lifecycle, identity binding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.phone_auth import (
    IdentityConflict,
    bind_phone_to_user,
    find_or_create_phone_user,
)
from app.services.sms import service as sms_service
from app.services.sms.service import (
    SmsRateLimited,
    normalize_cn_phone,
    send_login_code,
    verify_code,
)
from app.services.sms.tencent import SmsSendError, _tc3_authorization


class FakeRedis:
    """In-memory stand-in covering the subset of redis used by sms.service."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = str(value)
        return True

    async def get(self, key):
        return self.store.get(key)

    async def incr(self, key):
        value = int(self.store.get(key, "0")) + 1
        self.store[key] = str(value)
        return value

    async def expire(self, key, ttl):
        return True

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
        return len(keys)


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()

    async def _get_redis():
        return redis

    monkeypatch.setattr(sms_service, "get_redis", _get_redis)
    return redis


# ── normalize_cn_phone ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("13812345678", "13812345678"),
        ("+8613812345678", "13812345678"),
        ("8613812345678", "13812345678"),
        ("138 1234 5678", "13812345678"),
        ("138-1234-5678", "13812345678"),
        ("12812345678", None),  # 1[3-9] second digit rule
        ("1381234567", None),  # too short
        ("138123456789", None),  # too long
        ("abcdefghijk", None),
        ("", None),
    ],
)
def test_normalize_cn_phone(raw, expected):
    assert normalize_cn_phone(raw) == expected


# ── verify_code ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_code_success_is_single_use(fake_redis):
    fake_redis.store["sms:code:13812345678"] = "123456"

    assert await verify_code("13812345678", "123456") is True
    # burned after success
    assert "sms:code:13812345678" not in fake_redis.store
    assert await verify_code("13812345678", "123456") is False


@pytest.mark.asyncio
async def test_verify_code_wrong_then_right(fake_redis):
    fake_redis.store["sms:code:13812345678"] = "123456"

    assert await verify_code("13812345678", "000000") is False
    assert await verify_code("13812345678", "123456") is True


@pytest.mark.asyncio
async def test_verify_code_burns_after_max_attempts(fake_redis):
    fake_redis.store["sms:code:13812345678"] = "123456"

    for _ in range(5):
        assert await verify_code("13812345678", "000000") is False
    # code deleted -> even the right code no longer works
    assert "sms:code:13812345678" not in fake_redis.store
    assert await verify_code("13812345678", "123456") is False


@pytest.mark.asyncio
async def test_verify_code_rejects_malformed_without_redis_hit(fake_redis):
    fake_redis.store["sms:code:13812345678"] = "123456"

    assert await verify_code("13812345678", "12345") is False
    assert await verify_code("13812345678", "abcdef") is False
    # malformed input must not consume attempts
    assert "sms:tries:13812345678" not in fake_redis.store


# ── send_login_code ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_login_code_mock_mode_stores_code(fake_redis, monkeypatch):
    monkeypatch.setattr(sms_service.settings, "sms_mock_enabled", True)
    monkeypatch.setattr(sms_service.settings, "app_env", "development")

    await send_login_code("13812345678")

    code = fake_redis.store["sms:code:13812345678"]
    assert len(code) == 6 and code.isdigit()
    assert "sms:cooldown:13812345678" in fake_redis.store


@pytest.mark.asyncio
async def test_send_login_code_cooldown(fake_redis, monkeypatch):
    monkeypatch.setattr(sms_service.settings, "sms_mock_enabled", True)
    monkeypatch.setattr(sms_service.settings, "app_env", "development")

    await send_login_code("13812345678")
    with pytest.raises(SmsRateLimited) as exc:
        await send_login_code("13812345678")
    assert exc.value.reason == "cooldown"


@pytest.mark.asyncio
async def test_send_login_code_daily_limit(fake_redis, monkeypatch):
    monkeypatch.setattr(sms_service.settings, "sms_mock_enabled", True)
    monkeypatch.setattr(sms_service.settings, "app_env", "development")
    fake_redis.store["sms:daily:13812345678"] = "10"

    with pytest.raises(SmsRateLimited) as exc:
        await send_login_code("13812345678")
    assert exc.value.reason == "daily_limit"


@pytest.mark.asyncio
async def test_send_login_code_rolls_back_on_send_failure(fake_redis, monkeypatch):
    monkeypatch.setattr(sms_service.settings, "sms_mock_enabled", False)
    monkeypatch.setattr(
        sms_service, "send_sms_code", AsyncMock(side_effect=SmsSendError("boom"))
    )

    with pytest.raises(SmsSendError):
        await send_login_code("13812345678")
    # rollback lets the user retry immediately with a fresh code
    assert "sms:code:13812345678" not in fake_redis.store
    assert "sms:cooldown:13812345678" not in fake_redis.store


@pytest.mark.asyncio
async def test_send_login_code_ip_hourly_limit(fake_redis, monkeypatch):
    monkeypatch.setattr(sms_service.settings, "sms_mock_enabled", True)
    monkeypatch.setattr(sms_service.settings, "app_env", "development")

    # 15 sends from the same IP across different phones is fine...
    for i in range(15):
        await send_login_code(f"138123456{i:02d}", client_ip="1.2.3.4")
    # ...the 16th is rejected at the IP dimension
    with pytest.raises(SmsRateLimited) as exc:
        await send_login_code("13912345678", client_ip="1.2.3.4")
    assert exc.value.reason == "ip_limit"


@pytest.mark.asyncio
async def test_cooldown_hit_does_not_consume_ip_quota(fake_redis, monkeypatch):
    monkeypatch.setattr(sms_service.settings, "sms_mock_enabled", True)
    monkeypatch.setattr(sms_service.settings, "app_env", "development")

    await send_login_code("13812345678", client_ip="1.2.3.4")
    ip_key = next(k for k in fake_redis.store if k.startswith("sms:iph:"))
    before = fake_redis.store[ip_key]

    with pytest.raises(SmsRateLimited):
        await send_login_code("13812345678", client_ip="1.2.3.4")

    assert fake_redis.store[ip_key] == before


@pytest.mark.asyncio
async def test_mock_mode_never_active_in_production(monkeypatch):
    monkeypatch.setattr(sms_service.settings, "sms_mock_enabled", True)
    monkeypatch.setattr(sms_service.settings, "app_env", "production")

    assert sms_service._mock_mode() is False


# ── TC3 signature structure ───────────────────────────────────────


def test_tc3_authorization_shape(monkeypatch):
    monkeypatch.setattr(sms_service.settings, "tencent_sms_secret_id", "AKIDtest")
    monkeypatch.setattr(sms_service.settings, "tencent_sms_secret_key", "secret")

    auth1 = _tc3_authorization('{"a":1}', 1_700_000_000)
    auth2 = _tc3_authorization('{"a":2}', 1_700_000_000)

    assert auth1.startswith("TC3-HMAC-SHA256 Credential=AKIDtest/")
    assert "/sms/tc3_request" in auth1
    assert "SignedHeaders=content-type;host" in auth1
    # different payloads must produce different signatures
    assert auth1 != auth2


# ── API endpoints (validation-layer; service internals mocked) ────


def test_sms_send_503_when_disabled(api_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "sms_enabled", False)

    res = api_client.post("/auth/sms/send", json={"phone": "13812345678"})
    assert res.status_code == 503


def test_sms_send_400_invalid_phone(api_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "sms_enabled", True)
    monkeypatch.setattr(settings, "sms_mock_enabled", True)

    res = api_client.post("/auth/sms/send", json={"phone": "12345678901"})
    assert res.status_code == 400


def test_sms_login_400_invalid_phone(api_client):
    res = api_client.post(
        "/auth/sms/login", json={"phone": "00000000000", "code": "123456"}
    )
    assert res.status_code == 400


def test_h5_config_exposes_sms_flag(api_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "sms_enabled", False)

    res = api_client.get("/auth/wechat/h5/config")
    assert res.status_code == 200
    body = res.json()
    assert "sms_enabled" in body
    assert body["sms_enabled"] is False


# ── endpoint logic (direct call, deps monkeypatched) ──────────────


class FakeRequest:
    headers = {}
    client = SimpleNamespace(host="1.2.3.4")


@pytest.mark.asyncio
async def test_sms_login_endpoint_happy_path(monkeypatch):
    from app.api.public import auth as auth_api
    from app.models.auth import SmsLoginRequest

    user = SimpleNamespace(id="user-1", username="ph_user", role="user")
    expected = auth_api.AuthResponse(
        token="jwt", user_id="user-1", username="ph_user", role="user", has_agent=True
    )
    monkeypatch.setattr(auth_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(auth_api, "clear_login_failures", AsyncMock())
    import app.services.sms as sms_pkg
    verify_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(sms_pkg, "verify_code", verify_mock)
    import app.services.phone_auth as phone_auth_mod

    focp_mock = AsyncMock(return_value=user)
    monkeypatch.setattr(phone_auth_mod, "find_or_create_phone_user", focp_mock)
    monkeypatch.setattr(auth_api, "ensure_default_agent_for_user", AsyncMock())
    monkeypatch.setattr(auth_api, "create_jwt", lambda user_id, role: "jwt")
    monkeypatch.setattr(auth_api, "_record_auth_activity", AsyncMock())
    monkeypatch.setattr(
        auth_api, "_build_auth_response", AsyncMock(return_value=expected)
    )

    response = await auth_api.sms_login(
        SmsLoginRequest(
            phone="+86 138 1234 5678", code="123456", channel="h5", platform="ios"
        ),
        FakeRequest(),
    )

    assert response == expected
    # normalized phone reaches the service layer
    verify_mock.assert_awaited_once_with("13812345678", "123456")
    auth_api.ensure_default_agent_for_user.assert_awaited_once_with("user-1")
    # signup-origin analytics recorded on the create path
    kwargs = focp_mock.await_args.kwargs
    assert focp_mock.await_args.args == ("13812345678",)
    assert kwargs["signup_fields"]["signupSource"] == "sms_h5"
    assert kwargs["signup_fields"]["signupPlatform"] == "ios"


@pytest.mark.asyncio
async def test_build_auth_response_never_falls_back_to_the_login_hash(monkeypatch):
    """展示名为空就是为空 —— 绝不能拿 username 兜底。

    username 对真实用户是 `ph_deadbeef` / `wx_89b939bc004` 这种内部 hash。历史上
    这里有一句 `or user.username`, 结果每个客户端都得再写一遍正则把它过滤掉。
    优先级链本身归 resolve_display_identity 管 (见 test_user_avatars.py)。
    """
    from app.api.public import auth as auth_api

    user = SimpleNamespace(id="user-1", username="ph_deadbeef", role="user")
    monkeypatch.setattr(auth_api, "get_active_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(
        auth_api, "resolve_display_identity", AsyncMock(return_value=(None, None))
    )
    import app.services.phone_auth as phone_auth_mod

    monkeypatch.setattr(
        phone_auth_mod,
        "get_identity_summary",
        AsyncMock(return_value=("13812345678", False)),
    )

    response = await auth_api._build_auth_response(user, "jwt")

    assert response.user_display_name is None
    assert response.phone == "138****5678"
    assert response.wechat_bound is False

    # 解析出名字时原样透出, 不做二次加工。
    monkeypatch.setattr(
        auth_api, "resolve_display_identity", AsyncMock(return_value=("小明", None))
    )
    response = await auth_api._build_auth_response(user, "jwt")
    assert response.user_display_name == "小明"


@pytest.mark.asyncio
async def test_sms_login_endpoint_channel_defaults_to_plain_sms(monkeypatch):
    """Legacy clients without channel land on the neutral "sms" source."""
    from app.api.public import auth as auth_api
    from app.models.auth import SmsLoginRequest

    user = SimpleNamespace(id="user-1", username="ph_user", role="user")
    monkeypatch.setattr(auth_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(auth_api, "clear_login_failures", AsyncMock())
    import app.services.sms as sms_pkg

    monkeypatch.setattr(sms_pkg, "verify_code", AsyncMock(return_value=True))
    import app.services.phone_auth as phone_auth_mod

    focp_mock = AsyncMock(return_value=user)
    monkeypatch.setattr(phone_auth_mod, "find_or_create_phone_user", focp_mock)
    monkeypatch.setattr(auth_api, "ensure_default_agent_for_user", AsyncMock())
    monkeypatch.setattr(auth_api, "create_jwt", lambda user_id, role: "jwt")
    monkeypatch.setattr(auth_api, "_record_auth_activity", AsyncMock())
    monkeypatch.setattr(auth_api, "_build_auth_response", AsyncMock())

    await auth_api.sms_login(
        SmsLoginRequest(phone="13812345678", code="123456"), FakeRequest()
    )

    fields = focp_mock.await_args.kwargs["signup_fields"]
    assert fields["signupSource"] == "sms"


@pytest.mark.asyncio
async def test_sms_login_endpoint_rejects_bad_code(monkeypatch):
    from fastapi import HTTPException

    from app.api.public import auth as auth_api
    from app.models.auth import SmsLoginRequest

    monkeypatch.setattr(auth_api, "enforce_login_rate_limit", AsyncMock())
    monkeypatch.setattr(auth_api, "record_login_failure", AsyncMock())
    import app.services.sms as sms_pkg
    monkeypatch.setattr(sms_pkg, "verify_code", AsyncMock(return_value=False))

    with pytest.raises(HTTPException) as exc:
        await auth_api.sms_login(
            SmsLoginRequest(phone="13812345678", code="000000"), FakeRequest()
        )
    assert exc.value.status_code == 401
    auth_api.record_login_failure.assert_awaited_once()


@pytest.mark.asyncio
async def test_phone_bind_endpoint_conflict_maps_409(monkeypatch):
    from fastapi import HTTPException

    from app.api.public import auth as auth_api
    from app.models.auth import PhoneBindRequest
    from app.services.phone_auth import IdentityConflict

    user = SimpleNamespace(id="user-1", username="wx_user", role="user")
    monkeypatch.setattr(
        auth_api,
        "db",
        SimpleNamespace(user=SimpleNamespace(find_unique=AsyncMock(return_value=user))),
    )
    monkeypatch.setattr(auth_api, "enforce_login_rate_limit", AsyncMock())
    import app.services.sms as sms_pkg
    monkeypatch.setattr(sms_pkg, "verify_code", AsyncMock(return_value=True))
    import app.services.phone_auth as phone_auth_mod

    monkeypatch.setattr(
        phone_auth_mod,
        "bind_phone_to_user",
        AsyncMock(side_effect=IdentityConflict("phone_taken")),
    )

    with pytest.raises(HTTPException) as exc:
        await auth_api.phone_bind(
            PhoneBindRequest(phone="13812345678", code="123456"),
            FakeRequest(),
            payload={"sub": "user-1"},
        )
    assert exc.value.status_code == 409


# ── phone_auth: find_or_create / bind ─────────────────────────────


def _mock_db(monkeypatch, module, **tables):
    db = SimpleNamespace(**tables)
    monkeypatch.setattr(module, "db", db)
    return db


@pytest.mark.asyncio
async def test_find_or_create_phone_user_existing(monkeypatch):
    from app.services import phone_auth

    identity = SimpleNamespace(id="iden-1", userId="user-1")
    user = SimpleNamespace(id="user-1", username="ph_x", role="user")
    _mock_db(
        monkeypatch,
        phone_auth,
        authidentity=SimpleNamespace(
            find_first=AsyncMock(return_value=identity),
            update=AsyncMock(),
        ),
        user=SimpleNamespace(find_unique=AsyncMock(return_value=user)),
    )

    result = await find_or_create_phone_user("13812345678")

    assert result.id == "user-1"


@pytest.mark.asyncio
async def test_bind_phone_conflict_other_user(monkeypatch):
    from app.services import phone_auth

    other = SimpleNamespace(id="iden-2", userId="user-other")
    _mock_db(
        monkeypatch,
        phone_auth,
        authidentity=SimpleNamespace(find_first=AsyncMock(return_value=other)),
    )

    with pytest.raises(IdentityConflict) as exc:
        await bind_phone_to_user("user-1", "13812345678")
    assert exc.value.reason == "phone_taken"


@pytest.mark.asyncio
async def test_bind_phone_idempotent_same_user(monkeypatch):
    from app.services import phone_auth

    mine = SimpleNamespace(id="iden-1", userId="user-1")
    db = _mock_db(
        monkeypatch,
        phone_auth,
        authidentity=SimpleNamespace(
            find_first=AsyncMock(return_value=mine),
            create=AsyncMock(),
            update=AsyncMock(),
        ),
    )

    await bind_phone_to_user("user-1", "13812345678")

    db.authidentity.create.assert_not_awaited()
    db.authidentity.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_bind_phone_rebind_repoints_existing_row(monkeypatch):
    from app.services import phone_auth

    old_mine = SimpleNamespace(id="iden-1", userId="user-1")
    # first find_first: nobody owns the new phone; second: my existing phone row
    db = _mock_db(
        monkeypatch,
        phone_auth,
        authidentity=SimpleNamespace(
            find_first=AsyncMock(side_effect=[None, old_mine]),
            create=AsyncMock(),
            update=AsyncMock(),
        ),
    )

    await bind_phone_to_user("user-1", "13900001111")

    db.authidentity.update.assert_awaited_once()
    kwargs = db.authidentity.update.await_args.kwargs
    assert kwargs["where"] == {"id": "iden-1"}
    assert kwargs["data"]["providerAccountId"] == "13900001111"


@pytest.mark.asyncio
async def test_bind_phone_creates_row_when_none(monkeypatch):
    from app.services import phone_auth

    db = _mock_db(
        monkeypatch,
        phone_auth,
        authidentity=SimpleNamespace(
            find_first=AsyncMock(side_effect=[None, None]),
            create=AsyncMock(),
            update=AsyncMock(),
        ),
    )

    await bind_phone_to_user("user-1", "13900001111")

    db.authidentity.create.assert_awaited_once()
    data = db.authidentity.create.await_args.kwargs["data"]
    assert data["provider"] == "phone"
    assert data["providerAccountId"] == "13900001111"
