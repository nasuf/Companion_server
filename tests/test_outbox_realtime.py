from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.services.runtime import outbox_realtime as api
from app.services.runtime.execution_scope import ExecutionScopeUnavailable


@pytest.fixture
def adapter(monkeypatch):
    scope = object()
    bind = AsyncMock(return_value=scope)
    store = MagicMock()
    store.acknowledge = AsyncMock(return_value=True)
    store.deliver_once = AsyncMock(return_value=False)
    store.has_pending = AsyncMock(return_value=False)
    monkeypatch.setattr(api, "bind_conversation_scope", bind)
    monkeypatch.setattr(api, "SqlOutbox", lambda db: store)
    ws = SimpleNamespace(send_json=AsyncMock(), close=AsyncMock())
    return SimpleNamespace(scope=scope, bind=bind, store=store, ws=ws)


async def test_replay_scopes_to_authenticated_socket(adapter):
    a = adapter
    await api.handle_delivery_frame(
        a.ws,
        "verified-actor",
        "verified-conversation",
        {"type": "delivery_resume", "data": {"actor": "forged"}},
    )
    a.bind.assert_awaited_once_with(
        actor_user_id="verified-actor", conversation_id="verified-conversation"
    )
    assert a.store.deliver_once.call_args.kwargs == {
        "scope": a.scope,
        "reconnect": True,
    }
    sender = a.store.deliver_once.call_args.args[1]
    await sender("verified-conversation", {"type": "reply"})
    a.ws.send_json.assert_awaited_with({"type": "reply"})


async def test_ack_uses_only_event_and_token(adapter):
    a = adapter
    await api.handle_delivery_frame(
        a.ws,
        "actor",
        "conv",
        {
            "type": "delivery_ack",
            "data": {"event_id": "id", "delivery_token": "1", "scope": "forged"},
        },
    )
    a.store.acknowledge.assert_awaited_once_with(a.scope, "id", "1")
    a.store.deliver_once.assert_not_called()


async def test_reset_denies_without_sending_reply(adapter):
    adapter.bind.side_effect = ExecutionScopeUnavailable()
    await api.handle_delivery_frame(
        adapter.ws, "actor", "conv", {"type": "delivery_resume"}
    )
    adapter.ws.close.assert_awaited_once_with(
        code=4403, reason="conversation_access_denied"
    )
    adapter.store.deliver_once.assert_not_called()


async def test_database_failure_does_not_expose_details_or_break_legacy_chat(adapter):
    adapter.store.has_pending.side_effect = RuntimeError(
        "database credential must not leak"
    )
    await api.handle_delivery_frame(
        adapter.ws, "actor", "conv", {"type": "delivery_resume"}
    )
    adapter.ws.send_json.assert_awaited_once_with(
        {"type": "delivery_error", "data": {"code": "storage_unavailable"}}
    )
    adapter.ws.close.assert_not_called()


async def test_bad_ack_is_sanitized(adapter):
    await api.handle_delivery_frame(
        adapter.ws, "actor", "conv", {"type": "delivery_ack", "data": None}
    )
    adapter.ws.send_json.assert_awaited_once_with(
        {"type": "delivery_error", "data": {"code": "invalid_ack"}}
    )
