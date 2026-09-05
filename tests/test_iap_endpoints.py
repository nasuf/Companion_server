"""IAP 公开端点：/iap/apple/verify 鉴权与错误码、/notifications 验签与幂等。"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.api.deps import require_redis
from app.services.payments.errors import AppleVerificationError, UnknownProductError


@pytest.fixture(autouse=True)
def _redis_ok():
    """/verify 是写路径带 require_redis（path 依赖，先于 auth/body 校验执行）。
    单测里 redis 健康标志是跨测试的全局状态，全量跑时可能为 False → 503 盖住
    我们要断言的 401/422/200。统一在本模块 override 掉 redis 闸门。
    """
    from app.main import app

    app.dependency_overrides[require_redis] = lambda: None
    yield
    app.dependency_overrides.pop(require_redis, None)


def _granted_result():
    return {
        "status": "granted",
        "kind": "consumable",
        "wallet": {
            "ticket_balance": 90,
            "point_balance": 0,
            "achievement_points_synced": 0,
            "gift_ticket_balance": 0,
        },
        "vip": {
            "is_vip": False,
            "vip_until": None,
            "vip_trial_available": True,
            "gift_ticket_balance": 0,
            "ticket_balance": 90,
            "point_balance": 0,
            "spendable_tickets": 90,
        },
    }


def test_verify_requires_auth(api_client):
    resp = api_client.post("/iap/apple/verify", json={"transaction_id": "t1"})
    assert resp.status_code == 401


def test_verify_ok(api_client, auth_header):
    with patch(
        "app.api.public.iap.grant.verify_and_grant",
        new=AsyncMock(return_value=_granted_result()),
    ):
        resp = api_client.post(
            "/iap/apple/verify",
            json={"transaction_id": "t1"},
            headers=auth_header("u1"),
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "granted"
    assert body["wallet"]["ticket_balance"] == 90


def test_verify_unknown_product_404(api_client, auth_header):
    with patch(
        "app.api.public.iap.grant.verify_and_grant",
        new=AsyncMock(side_effect=UnknownProductError("com.x.nope")),
    ):
        resp = api_client.post(
            "/iap/apple/verify",
            json={"transaction_id": "t1"},
            headers=auth_header("u1"),
        )
    assert resp.status_code == 404


def test_verify_apple_rejected_402(api_client, auth_header):
    with patch(
        "app.api.public.iap.grant.verify_and_grant",
        new=AsyncMock(side_effect=AppleVerificationError("bad")),
    ):
        resp = api_client.post(
            "/iap/apple/verify",
            json={"transaction_id": "t1"},
            headers=auth_header("u1"),
        )
    assert resp.status_code == 402


def test_verify_rejects_unknown_fields(api_client, auth_header):
    # IapVerifyRequest 用 extra="forbid"
    resp = api_client.post(
        "/iap/apple/verify",
        json={"transaction_id": "t1", "surprise": 1},
        headers=auth_header("u1"),
    )
    assert resp.status_code == 422


def test_notifications_missing_payload_400(api_client):
    resp = api_client.post("/iap/apple/notifications", json={})
    assert resp.status_code == 400


def test_notifications_invalid_signature_401(api_client):
    with patch(
        "app.api.public.iap.notifications.apply_notification",
        new=AsyncMock(side_effect=AppleVerificationError("bad sig")),
    ):
        resp = api_client.post(
            "/iap/apple/notifications", json={"signedPayload": "jws"}
        )
    assert resp.status_code == 401


def test_notifications_ok_200(api_client):
    with patch(
        "app.api.public.iap.notifications.apply_notification",
        new=AsyncMock(return_value=None),
    ):
        resp = api_client.post(
            "/iap/apple/notifications", json={"signedPayload": "jws"}
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
