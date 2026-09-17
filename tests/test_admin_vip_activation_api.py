from __future__ import annotations

from unittest.mock import AsyncMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"sub": "admin-1", "role": "admin"}
    return app, require_admin_jwt


def _user_override(user_id: str = "user-1"):
    from app.api.jwt_auth import require_user
    from app.main import app

    app.dependency_overrides[require_user] = lambda: {"sub": user_id}
    return app, require_user


def test_admin_create_and_list_codes(api_client):
    app, require_admin_jwt = _admin_override()
    sample = [
        {
            "id": "c1",
            "code": "ABCD-1234",
            "duration_days": 30,
            "max_redemptions": 1,
            "redemption_count": 0,
            "enabled": True,
            "valid_from": None,
            "valid_until": None,
            "note": "test",
            "created_by": "admin-1",
            "created_at": "2026-09-17T00:00:00+00:00",
            "updated_at": "2026-09-17T00:00:00+00:00",
        }
    ]
    try:
        with (
            patch(
                "app.api.admin.vip_activation.activation_admin.create_codes",
                new_callable=AsyncMock,
                return_value=sample,
            ) as create_codes,
            patch(
                "app.api.admin.vip_activation.activation_admin.list_codes",
                new_callable=AsyncMock,
                return_value={"items": sample, "total": 1, "limit": 50, "offset": 0},
            ) as list_codes,
        ):
            create_resp = api_client.post(
                "/admin-api/vip-activation/codes",
                json={"duration_days": 30, "count": 1, "max_redemptions": 1},
            )
            list_resp = api_client.get("/admin-api/vip-activation/codes")

        assert create_resp.status_code == 200
        assert create_resp.json()[0]["code"] == "ABCD-1234"
        create_codes.assert_awaited_once()
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] == 1
        list_codes.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_admin_revoke_redemption(api_client):
    app, require_admin_jwt = _admin_override()
    try:
        with patch(
            "app.api.admin.vip_activation.activation_admin.revoke_redemption",
            new_callable=AsyncMock,
            return_value={"status": "revoked"},
        ) as revoke:
            response = api_client.post(
                "/admin-api/vip-activation/redemptions/r1/revoke",
            )
        assert response.status_code == 200
        revoke.assert_awaited_once_with("r1", admin_user_id="admin-1")
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_user_redeem_code(api_client):
    app, require_user = _user_override()
    payload = {
        "vip": {"is_vip": True, "vip_until": "2026-10-17T00:00:00+00:00"},
        "redemption": {
            "id": "r1",
            "code_id": "c1",
            "duration_days": 30,
            "redeemed_at": "2026-09-17T00:00:00+00:00",
            "effective_start": "2026-09-17T00:00:00+00:00",
            "effective_end": "2026-10-17T00:00:00+00:00",
        },
    }
    try:
        with patch(
            "app.api.public.vip_activation.redeem_code",
            new_callable=AsyncMock,
            return_value=payload,
        ) as redeem:
            response = api_client.post(
                "/me/vip/redeem-code",
                json={"code": "ABCD-1234"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["vip"]["is_vip"] is True
        redeem.assert_awaited_once_with("user-1", "ABCD-1234")
    finally:
        app.dependency_overrides.pop(require_user, None)
