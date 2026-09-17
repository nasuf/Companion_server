from __future__ import annotations

from unittest.mock import AsyncMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"sub": "admin-1", "role": "admin"}
    return app, require_admin_jwt


def test_admin_payments_vip_members_route(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        with patch(
            "app.api.admin.payments.payments_admin.list_vip_members",
            new_callable=AsyncMock,
            return_value={"items": [], "total": 0, "active_count": 0},
        ):
            response = api_client.get("/admin-api/payments/vip-members?limit=20&offset=0")
        assert response.status_code == 200
        assert response.json()["total"] == 0
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_admin_payments_recharges_route(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        with patch(
            "app.api.admin.payments.payments_admin.list_recharges",
            new_callable=AsyncMock,
            return_value={
                "items": [],
                "total": 0,
                "total_tickets": 0,
                "distinct_users": 0,
            },
        ):
            response = api_client.get("/admin-api/payments/recharges?limit=20&offset=0")
        assert response.status_code == 200
        assert response.json()["total"] == 0
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
