from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.api.admin.users import _serialize_wechat_identity
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


def test_list_users_keeps_exclusion_when_searching(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.user.count = AsyncMock(return_value=0)
    fake_db.user.find_many = AsyncMock(return_value=[])
    fake_db.authidentity.find_many = AsyncMock(return_value=[])

    try:
        with patch("app.api.admin.users.db", fake_db):
            response = api_client.get("/admin-api/users", params={"search": "template"})

        assert response.status_code == 200
        where = fake_db.user.find_many.await_args.kwargs["where"]
        assert {"username": {"not": TEMPLATE_SYSTEM_USERNAME}} in where["AND"]
        assert {"username": {"contains": "template", "mode": "insensitive"}} in where["AND"]
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


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
