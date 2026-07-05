"""WebSocket authentication tests.

Covers the token extractor/decoder (`authenticate_ws`) and the endpoint's
close-code behavior: conversation_id must not be a capability token — a valid
JWT whose `sub` owns the conversation (or an admin) is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.websockets import WebSocketDisconnect

from app.api.jwt_auth import authenticate_ws
from app.services.auth import create_jwt


def _fake_ws(*, query: dict | None = None, headers: dict | None = None) -> MagicMock:
    ws = MagicMock()
    ws.query_params = query or {}
    ws.headers = headers or {}
    return ws


# ── authenticate_ws unit ────────────────────────────────────────────────

def test_authenticate_ws_reads_query_token():
    token = create_jwt("u1", role="user")
    payload = authenticate_ws(_fake_ws(query={"token": token}))
    assert payload is not None
    assert payload["sub"] == "u1"


def test_authenticate_ws_reads_authorization_header():
    token = create_jwt("u1", role="user")
    payload = authenticate_ws(_fake_ws(headers={"authorization": f"Bearer {token}"}))
    assert payload is not None
    assert payload["sub"] == "u1"


def test_authenticate_ws_query_takes_precedence_over_header():
    good = create_jwt("u1", role="user")
    payload = authenticate_ws(
        _fake_ws(query={"token": good}, headers={"authorization": "Bearer garbage"})
    )
    assert payload is not None and payload["sub"] == "u1"


def test_authenticate_ws_missing_returns_none():
    assert authenticate_ws(_fake_ws()) is None


def test_authenticate_ws_invalid_returns_none():
    assert authenticate_ws(_fake_ws(query={"token": "not-a-jwt"})) is None


# ── endpoint close-code behavior ────────────────────────────────────────

@pytest.fixture
def ws_client(api_client):
    return api_client


def _patch_ws_env(conv):
    """Patch redis health + db lookups used by the WS endpoint pre-accept path."""
    db_mock = MagicMock()
    db_mock.conversation.find_unique = AsyncMock(return_value=conv)
    db_mock.user.find_unique = AsyncMock(return_value=SimpleNamespace(username="x"))
    return (
        patch("app.api.realtime.ws.is_redis_healthy", return_value=True),
        patch("app.api.realtime.ws.db", db_mock),
    )


def _conv(user_id="u1"):
    return SimpleNamespace(
        id="c1", userId=user_id, isDeleted=False, workspaceId="w1",
        agent=SimpleNamespace(id="a1", name="小樱"),
    )


def test_ws_rejects_without_token(ws_client):
    p_redis, p_db = _patch_ws_env(_conv())
    with p_redis, p_db, pytest.raises(WebSocketDisconnect) as exc:
        with ws_client.websocket_connect("/ws/c1"):
            pass
    assert exc.value.code == 4401


def test_ws_rejects_wrong_user(ws_client):
    token = create_jwt("intruder", role="user")
    p_redis, p_db = _patch_ws_env(_conv(user_id="u1"))
    with p_redis, p_db, pytest.raises(WebSocketDisconnect) as exc:
        with ws_client.websocket_connect(f"/ws/c1?token={token}"):
            pass
    assert exc.value.code == 4403


def test_ws_accepts_owner(ws_client):
    token = create_jwt("u1", role="user")
    p_redis, p_db = _patch_ws_env(_conv(user_id="u1"))
    with (
        p_redis, p_db,
        patch("app.api.realtime.ws.send_first_greeting", new_callable=AsyncMock),
        patch("app.api.realtime.ws.manager.connect", new_callable=AsyncMock),
        patch("app.api.realtime.ws.manager.disconnect", new_callable=AsyncMock),
    ):
        with ws_client.websocket_connect(f"/ws/c1?token={token}") as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}


def test_ws_accepts_admin_for_any_conversation(ws_client):
    token = create_jwt("some-admin", role="admin")
    p_redis, p_db = _patch_ws_env(_conv(user_id="u1"))
    with (
        p_redis, p_db,
        patch("app.api.realtime.ws.send_first_greeting", new_callable=AsyncMock),
        patch("app.api.realtime.ws.manager.connect", new_callable=AsyncMock),
        patch("app.api.realtime.ws.manager.disconnect", new_callable=AsyncMock),
    ):
        with ws_client.websocket_connect(f"/ws/c1?token={token}") as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}


def test_ws_auth_disabled_allows_no_token(ws_client):
    """ws_require_auth=False (staged rollout) → 不强制 token."""
    p_redis, p_db = _patch_ws_env(_conv(user_id="u1"))
    from app.config import settings
    with (
        p_redis, p_db,
        patch.object(settings, "ws_require_auth", False),
        patch("app.api.realtime.ws.send_first_greeting", new_callable=AsyncMock),
        patch("app.api.realtime.ws.manager.connect", new_callable=AsyncMock),
        patch("app.api.realtime.ws.manager.disconnect", new_callable=AsyncMock),
    ):
        with ws_client.websocket_connect("/ws/c1") as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}
