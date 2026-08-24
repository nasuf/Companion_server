from __future__ import annotations

from unittest.mock import AsyncMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"sub": "admin-1", "role": "admin"}
    return app, require_admin_jwt


def test_vip_set_grants_vip_and_does_not_clear_gift_tickets(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        with (
            patch(
                "app.api.admin.wallet.wallet.admin_set_vip_until",
                new_callable=AsyncMock,
                return_value={
                    "user_id": "u1",
                    "is_vip": True,
                    "vip_until": "2026-09-23T00:00:00+00:00",
                },
            ) as set_vip,
            patch(
                "app.api.admin.wallet.grants.clear_on_lapse",
                new_callable=AsyncMock,
            ) as clear_on_lapse,
        ):
            response = api_client.post(
                "/admin-api/wallet/vip-set",
                json={"user_id": "u1", "vip_until": "2026-09-23T00:00:00+00:00"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["is_vip"] is True
        set_vip.assert_awaited_once()
        # Granting/extending VIP must never trigger the lapse-clear path.
        clear_on_lapse.assert_not_called()
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_vip_set_to_null_ends_vip_and_synchronously_clears_gift_tickets(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        with (
            patch(
                "app.api.admin.wallet.wallet.admin_set_vip_until",
                new_callable=AsyncMock,
                return_value={"user_id": "u1", "is_vip": False, "vip_until": None},
            ),
            patch(
                "app.api.admin.wallet.grants.clear_on_lapse",
                new_callable=AsyncMock,
            ) as clear_on_lapse,
        ):
            response = api_client.post(
                "/admin-api/wallet/vip-set",
                json={"user_id": "u1", "vip_until": None},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["is_vip"] is False
        # Ending VIP from the admin console must be immediate, not wait for
        # the nightly vip_expire_clear cron.
        clear_on_lapse.assert_awaited_once_with("u1")
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_vip_set_rejects_malformed_vip_until(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        response = api_client.post(
            "/admin-api/wallet/vip-set",
            json={"user_id": "u1", "vip_until": "not-a-date"},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
