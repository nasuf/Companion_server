from __future__ import annotations

from unittest.mock import AsyncMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"sub": "admin-1", "role": "admin"}
    return app, require_admin_jwt


def test_get_status_returns_current_quota(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        with (
            patch(
                "app.api.admin.chat_quota.wallet.is_vip",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "app.api.admin.chat_quota.chat_quota.preview",
                new_callable=AsyncMock,
                return_value={
                    "mode": "paid",
                    "free_remaining": 0,
                    "per_msg_cost": 0.5,
                    "spendable_tickets": 3,
                    "used": 20,
                    "limit": 20,
                    "period_scope": "day",
                    "period_key": "2026-08-24",
                },
            ),
        ):
            response = api_client.get(
                "/admin-api/chat-quota/status", params={"user_id": "u1"}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["used"] == 20
        assert body["is_vip"] is False
        assert body["period_key"] == "2026-08-24"
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_reset_resets_and_logs(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        with (
            patch(
                "app.api.admin.chat_quota.wallet.is_vip",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "app.api.admin.chat_quota.chat_quota.admin_reset",
                new_callable=AsyncMock,
                return_value={
                    "mode": "free",
                    "free_remaining": 5200,
                    "per_msg_cost": 0.3,
                    "spendable_tickets": 0,
                    "used": 0,
                    "limit": 5200,
                    "period_scope": "month",
                    "period_key": "2026-08",
                },
            ) as reset_mock,
        ):
            response = api_client.post(
                "/admin-api/chat-quota/reset",
                json={"user_id": "u1", "note": "客诉补偿"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["used"] == 0
        assert body["is_vip"] is True
        reset_mock.assert_awaited_once_with("u1", is_vip=True)
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_reset_requires_user_id(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        response = api_client.post("/admin-api/chat-quota/reset", json={})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_reset_rejects_unknown_fields(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        response = api_client.post(
            "/admin-api/chat-quota/reset",
            json={"user_id": "u1", "unexpected": "nope"},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
