from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.admin.users import (
    _serialize_admin_user,
    _serialize_phone_identity,
    _serialize_wechat_identity,
)
from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app, require_admin_jwt


def test_serialize_wechat_identity_exposes_admin_profile_fields():
    identity = SimpleNamespace(
        provider="wechat",
        providerAccountId="union-1",
        openid="open-1",
        unionid="union-1",
        scope="snsapi_userinfo",
        rawProfile={
            "nickname": "七七",
            "headimgurl": "https://wx.example/avatar.png",
            "sex": 2,
            "province": "Guangdong",
            "city": "Shenzhen",
            "country": "CN",
            "privilege": ["tester"],
        },
        lastLoginAt=datetime(2026, 6, 1, 8, 30, tzinfo=UTC),
        createdAt=datetime(2026, 5, 20, 10, 0, tzinfo=UTC),
        updatedAt=datetime(2026, 6, 1, 8, 31, tzinfo=UTC),
    )

    payload = _serialize_wechat_identity(identity)

    assert payload == {
        "provider": "wechat",
        "provider_account_id": "union-1",
        "openid": "open-1",
        "unionid": "union-1",
        "scope": "snsapi_userinfo",
        "nickname": "七七",
        "avatar_url": "https://wx.example/avatar.png",
        "sex": 2,
        "province": "Guangdong",
        "city": "Shenzhen",
        "country": "CN",
        "privilege": ["tester"],
        "last_login_at": "2026-06-01 08:30:00+00:00",
        "created_at": "2026-05-20 10:00:00+00:00",
        "updated_at": "2026-06-01 08:31:00+00:00",
    }


def test_serialize_phone_identity_exposes_admin_phone_fields():
    identity = SimpleNamespace(
        provider="phone",
        providerAccountId="13812345678",
        rawProfile={"phone": "13812345678"},
        lastLoginAt=datetime(2026, 6, 2, 8, 30, tzinfo=UTC),
        createdAt=datetime(2026, 5, 21, 10, 0, tzinfo=UTC),
        updatedAt=datetime(2026, 6, 2, 8, 31, tzinfo=UTC),
    )

    payload = _serialize_phone_identity(identity)

    assert payload == {
        "provider": "phone",
        "provider_account_id": "13812345678",
        "phone": "13812345678",
        "phone_masked": "138****5678",
        "last_login_at": "2026-06-02 08:30:00+00:00",
        "created_at": "2026-05-21 10:00:00+00:00",
        "updated_at": "2026-06-02 08:31:00+00:00",
    }


def test_serialize_admin_user_includes_login_methods_and_contact_fields():
    user = SimpleNamespace(
        id="user-1",
        username="alice",
        email="alice@example.com",
        hashedPassword="hashed",
        role="user",
        createdAt=datetime(2026, 6, 1, tzinfo=UTC),
        status="active",
        archivedAt=None,
        signupSource="password_web",
        displayName="爱丽丝",
        agents=[SimpleNamespace(id="agent-1")],
    )
    identities = [
        SimpleNamespace(
            provider="phone",
            providerAccountId="13812345678",
            rawProfile={"phone": "13812345678"},
            lastLoginAt=datetime(2026, 6, 2, 8, 30, tzinfo=UTC),
            createdAt=datetime(2026, 5, 21, 10, 0, tzinfo=UTC),
            updatedAt=datetime(2026, 6, 2, 8, 31, tzinfo=UTC),
        )
    ]

    payload = _serialize_admin_user(user, identities)

    # 后台的显示链要跟客户端一致 (自设 → 微信昵称), 少了这个字段就只能显示旧的
    # 微信昵称, 客服照着念会跟用户自己看到的不一样。
    assert payload["display_name"] == "爱丽丝"
    assert payload["email"] == "alice@example.com"
    assert payload["phone"]["phone"] == "13812345678"
    assert payload["phone"]["phone_masked"] == "138****5678"
    assert payload["auth_methods"] == [
        {
            "type": "password",
            "label": "邮箱密码",
            "identifier": "alice@example.com",
            "email": "alice@example.com",
            "username": "alice",
            "signup_source": "password_web",
        },
        {
            "type": "phone",
            "label": "手机号",
            "identifier": "138****5678",
            "phone": "13812345678",
            "phone_masked": "138****5678",
            "last_login_at": "2026-06-02 08:30:00+00:00",
        },
    ]


def _fake_user_row(user_id: str = "user-1", username: str = "alice"):
    return SimpleNamespace(
        id=user_id,
        username=username,
        role="user",
        createdAt=datetime(2026, 6, 1, tzinfo=UTC),
        status="active",
        archivedAt=None,
        agents=[],
    )


def test_list_users_excludes_template_system_user(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.user.count = AsyncMock(return_value=1)
    fake_db.user.find_many = AsyncMock(return_value=[_fake_user_row()])
    fake_db.authidentity.find_many = AsyncMock(return_value=[])
    fake_db.query_raw = AsyncMock(return_value=[])

    try:
        with patch("app.api.admin.users.db", fake_db):
            response = api_client.get("/admin-api/users")

        assert response.status_code == 200
        where = fake_db.user.find_many.await_args.kwargs["where"]
        assert {"username": {"not": TEMPLATE_SYSTEM_USERNAME}} in where["AND"]
        # The count query must use the same exclusion so pagination totals match.
        assert fake_db.user.count.await_args.kwargs["where"] == where
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


@pytest.mark.parametrize("query", ["alice", "user-1", "七七", "1381234", "alice@example.com", "爱丽丝", "x' OR 1=1 --"])
def test_list_users_searches_all_fields_with_pagination(api_client, query):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[[{"total": 41}], [{"id": "user-1"}], []])
    fake_db.user.find_many = AsyncMock(return_value=[_fake_user_row()])
    fake_db.authidentity.find_many = AsyncMock(return_value=[])
    try:
        with patch("app.api.admin.users.db", fake_db):
            response = api_client.get("/admin-api/users", params={"search": f" {query} ", "limit": 30, "offset": 30})
        assert response.status_code == 200
        assert response.json()["total"] == 41
        assert response.json()["users"][0]["id"] == "user-1"
        count_call, page_call, _last_chat = fake_db.query_raw.await_args_list
        count_sql, page_sql = count_call.args[0], page_call.args[0]
        assert count_call.args[1:] == (f"%{query}%", TEMPLATE_SYSTEM_USERNAME)
        assert page_call.args[1:] == (f"%{query}%", TEMPLATE_SYSTEM_USERNAME, 30, 30)
        predicate = count_sql[count_sql.index("FROM users"):]
        assert predicate in page_sql
        for field in ["u.username", "u.id", "u.display_name", "u.email", "ai.raw_profile->>'nickname'", "ai.provider_account_id"]:
            assert f"{field} ILIKE $1" in predicate
        assert "ai.provider = 'wechat'" in predicate
        assert "ai.provider = 'phone'" in predicate
        assert "EXISTS" in predicate
        assert "LIMIT $3 OFFSET $4" in page_sql
        kwargs = fake_db.user.find_many.await_args.kwargs
        assert kwargs["where"]["AND"] == [
            {"username": {"not": TEMPLATE_SYSTEM_USERNAME}}, {"id": {"in": ["user-1"]}}
        ]
        assert kwargs["skip"] == 0
        assert kwargs["take"] == 30
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_list_users_empty_search_page_preserves_total(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[[{"total": 1}], [], []])
    fake_db.user.find_many = AsyncMock(return_value=[])
    fake_db.authidentity.find_many = AsyncMock(return_value=[])
    try:
        with patch("app.api.admin.users.db", fake_db):
            response = api_client.get("/admin-api/users", params={"search": "七七", "offset": 30})
        assert response.json() == {"users": [], "total": 1}
        assert fake_db.user.find_many.await_args.kwargs["where"]["AND"][-1] == {"id": {"in": []}}
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_list_users_search_requires_admin(api_client, auth_header):
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock()
    fake_db.user.find_unique = AsyncMock(return_value=SimpleNamespace(role="user", status="active", archivedAt=None))
    with patch("app.api.admin.users.db", fake_db), patch("app.db.db", fake_db):
        response = api_client.get("/admin-api/users", params={"search": "1381234"}, headers=auth_header("user-1"))
    assert response.status_code == 403
    fake_db.query_raw.assert_not_awaited()


def test_delete_user_refuses_template_system_user(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.user.find_unique = AsyncMock(
        return_value=SimpleNamespace(
            id="tpl-owner-1",
            username=TEMPLATE_SYSTEM_USERNAME,
            role="user",
        )
    )
    hard_delete = AsyncMock()

    try:
        with (
            patch("app.api.admin.users.db", fake_db),
            patch("app.services.runtime.data_reset.hard_delete_user_data", hard_delete),
        ):
            response = api_client.delete("/admin-api/users/tpl-owner-1")

        assert response.status_code == 400
        assert "template system user" in response.json()["detail"]
        hard_delete.assert_not_awaited()
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_delete_user_cleanup_failure_is_not_reported_as_success(api_client):
    app, admin = _admin_override()
    fake = MagicMock()
    fake.user.find_unique = AsyncMock(return_value=_fake_user_row())
    try:
        with patch("app.api.admin.users.db",fake), patch(
            "app.services.runtime.data_reset.hard_delete_user_data",AsyncMock(side_effect=RuntimeError("PRIVATE_DISK_PATH"))):
            response=api_client.delete("/admin-api/users/user-1")
        assert response.status_code==503
        assert "删除未完成" in response.json()["detail"]
        assert "PRIVATE_DISK_PATH" not in response.text
    finally:app.dependency_overrides.pop(admin,None)


def test_serialize_admin_user_exposes_platform_and_activity_times():
    user = SimpleNamespace(
        id="user-1", username="alice", role="user", createdAt=datetime(2026, 6, 1, tzinfo=UTC),
        status="active", archivedAt=None, signupSource="wechat_h5", signupPlatform="ios",
        lastSeenAt=datetime(2026, 10, 28, 12, 0), agents=[],
    )
    payload = _serialize_admin_user(user, [], last_chat_at="2026-10-28T11:30:00")
    assert payload["signup_platform"] == "ios"
    # Naive UTC columns are emitted with an explicit offset so the browser does not read them as local time.
    assert payload["last_seen_at"] == "2026-10-28T12:00:00+00:00"
    assert payload["last_chat_at"] == "2026-10-28T11:30:00+00:00"
    assert _serialize_admin_user(user, [], last_chat_at=None)["last_chat_at"] is None


def test_list_users_filters_by_login_platform_and_activity_windows(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[[{"total": 2}], [{"id": "user-2"}, {"id": "user-1"}], []])
    fake_db.user.find_many = AsyncMock(return_value=[_fake_user_row("user-1"), _fake_user_row("user-2", "bob")])
    fake_db.authidentity.find_many = AsyncMock(return_value=[])
    try:
        with patch("app.api.admin.users.db", fake_db):
            response = api_client.get("/admin-api/users", params={
                "login": "wechat", "platform": "ios", "registered": "30d",
                "last_login": "over7d", "last_chat": "never", "sort": "last_chat_at", "limit": 30,
            })
        assert response.status_code == 200
        # The SQL page order wins (most recent chat first), not the ORM's created_at order.
        assert [u["id"] for u in response.json()["users"]] == ["user-2", "user-1"]
        count_call, page_call, _ = fake_db.query_raw.await_args_list
        sql, params = count_call.args[0], count_call.args[1:]
        assert params[0] == TEMPLATE_SYSTEM_USERNAME and "u.username <> $1" in sql
        assert params[1] == "wechat" and "li.provider = $2" in sql
        assert params[2] == "ios" and "u.signup_platform = $3" in sql
        assert "u.created_at >= $4::timestamp" in sql
        assert "u.last_seen_at < $5::timestamp" in sql
        assert "lc.last_chat_at IS NULL" in sql and "LEFT JOIN LATERAL" in sql
        assert "lc.last_chat_at DESC NULLS LAST" in page_call.args[0]
        assert page_call.args[-2:] == (30, 0)
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


@pytest.mark.parametrize("params", [{"platform": "symbian"}, {"last_chat": "2d"}, {"sort": "username"}, {"login": "qq"}])
def test_list_users_rejects_unknown_filter_values(api_client, params):
    app, require_admin_jwt = _admin_override()
    try:
        assert api_client.get("/admin-api/users", params=params).status_code == 422
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_list_users_unknown_platform_and_password_login(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(side_effect=[[{"total": 0}], [], []])
    fake_db.user.find_many = AsyncMock(return_value=[])
    fake_db.authidentity.find_many = AsyncMock(return_value=[])
    try:
        with patch("app.api.admin.users.db", fake_db):
            api_client.get("/admin-api/users", params={"platform": "unknown", "login": "password"})
        sql = fake_db.query_raw.await_args_list[0].args[0]
        assert "u.hashed_password IS NOT NULL" in sql
        assert "u.signup_platform IS NULL OR NOT (u.signup_platform = ANY($2::text[]))" in sql
        assert "LEFT JOIN LATERAL" not in sql  # only joined when the last-chat filter or sort needs it
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
