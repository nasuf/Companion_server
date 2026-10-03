"""Exercise actual HTTP dependencies before chat/proactive side effects."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.public import chat as routes
from app.services.auth import create_jwt


@pytest.fixture
def secured_chat(monkeypatch):
    from app.api import deps, ownership

    agent = SimpleNamespace(id="agent-1", userId="owner", name="Test", status="active")
    conv = SimpleNamespace(id="conv-1", userId="owner", isDeleted=False,
                           agent=agent, workspaceId="workspace-1")
    db = MagicMock()
    db.conversation.find_unique = AsyncMock(return_value=conv)
    db.aiagent.find_unique = AsyncMock(return_value=agent)
    db.message.create = AsyncMock(return_value=SimpleNamespace(id="message-1"))
    monkeypatch.setattr(routes, "db", db)
    monkeypatch.setattr(ownership, "db", db)
    monkeypatch.setattr(deps, "is_redis_healthy", lambda: True)
    business = {}
    for name, result in {
        "get_cached_schedule": [{"activity": "free"}],
        "generate_daily_schedule": [],
        "build_reply_timing_context": {"delay_seconds": 2},
        "plan_user_message_aggregation": SimpleNamespace(
            should_wait=False, metadata={}, final_message="hello",
            final_context={"delay_seconds": 2}, fallback_message="hello",
            fallback_context={"delay_seconds": 7}),
        "enqueue_planned_user_message": True,
        "enqueue_or_append_delayed": None,
        "mark_user_replied_for_conversation": None,
        "resolve_workspace_id": "workspace-1",
        "send_manual_or_triggered_proactive": {"ok": True, "message": "hi"},
        "get_proactive_history": [{"content": "synthetic history"}],
    }.items():
        business[name] = AsyncMock(return_value=result)
        monkeypatch.setattr(routes, name, business[name])
    monkeypatch.setattr(routes, "get_current_status", lambda _: {"status": "idle"})
    # Close rather than schedule a coroutine; no DB/Redis/LLM background work.
    background = MagicMock(side_effect=lambda coroutine: coroutine.close())
    monkeypatch.setattr(routes, "fire_background", background)
    business["fire_background"] = background
    app = FastAPI()
    app.include_router(routes.router)
    return SimpleNamespace(client=TestClient(app), db=db, agent=agent, conv=conv,
                           business=business)


def _request(s, endpoint, headers=None, *, user_id="owner"):
    if endpoint == "chat":
        return s.client.post("/chat/conv-1", json={"message": "hello"}, headers=headers)
    path = "/chat/proactive/agent-1"
    if endpoint == "history":
        return s.client.get(path + "/history", params={"user_id": user_id}, headers=headers)
    return s.client.post(path, params={"user_id": user_id}, headers=headers)


def _no_business(s):
    s.db.message.create.assert_not_called()
    for call in s.business.values():
        call.assert_not_called()


@pytest.mark.parametrize("endpoint", ["chat", "trigger", "history"])
@pytest.mark.parametrize("token_kind", ["missing", "invalid", "expired"])
def test_authentication_precedes_resource_read_and_business(secured_chat, endpoint, token_kind):
    if token_kind == "missing":
        headers = {}
    else:
        token = "invalid-jwt" if token_kind == "invalid" else create_jwt("owner", "user", expiry_hours=-1)
        headers = {"Authorization": f"Bearer {token}"}
    response = _request(secured_chat, endpoint, headers)
    assert response.status_code == 401
    secured_chat.db.conversation.find_unique.assert_not_called()
    secured_chat.db.aiagent.find_unique.assert_not_called()
    _no_business(secured_chat)


@pytest.mark.parametrize("endpoint", ["chat", "trigger", "history"])
def test_other_user_is_rejected_without_side_effects(secured_chat, auth_header, endpoint):
    response = _request(secured_chat, endpoint, auth_header("intruder"))
    assert response.status_code == 403
    _no_business(secured_chat)


@pytest.mark.parametrize("endpoint", ["trigger", "history"])
def test_query_user_cannot_be_spoofed(secured_chat, auth_header, endpoint):
    response = _request(secured_chat, endpoint, auth_header("owner"), user_id="intruder")
    assert response.status_code == 403
    secured_chat.db.aiagent.find_unique.assert_not_called()
    _no_business(secured_chat)


@pytest.mark.parametrize("endpoint", ["trigger", "history"])
def test_foreign_agent_rejected_even_when_user_query_matches_jwt(secured_chat, auth_header, endpoint):
    secured_chat.agent.userId = "other-owner"
    response = _request(secured_chat, endpoint, auth_header("owner"))
    assert response.status_code == 403
    _no_business(secured_chat)


@pytest.mark.parametrize("endpoint", ["trigger", "history"])
def test_admin_must_select_consistent_user_agent_pair(secured_chat, auth_header, endpoint):
    response = _request(secured_chat, endpoint, auth_header("admin", "admin"), user_id="intruder")
    assert response.status_code == 403
    _no_business(secured_chat)


@pytest.mark.parametrize("role", ["user", "admin"])
@pytest.mark.parametrize("case,expected", [
    ("missing", 404), ("deleted", 410), ("provisioning", 503),
    ("no_agent", 404), ("agent_owner_mismatch", 403),
])
def test_chat_resource_guards_precede_side_effects(secured_chat, auth_header, role, case, expected):
    if case == "missing":
        secured_chat.db.conversation.find_unique.return_value = None
    elif case == "deleted":
        secured_chat.conv.isDeleted = True
    elif case == "provisioning":
        secured_chat.agent.status = "provisioning"
    elif case == "no_agent":
        secured_chat.conv.agent = None
    else:
        secured_chat.agent.userId = "other-owner"
    response = _request(secured_chat, "chat", auth_header("admin" if role == "admin" else "owner", role))
    assert response.status_code == expected
    _no_business(secured_chat)


def test_deleted_foreign_conversation_does_not_expose_deleted_state(secured_chat, auth_header):
    secured_chat.conv.isDeleted = True
    assert _request(secured_chat, "chat", auth_header("intruder")).status_code == 403
    _no_business(secured_chat)


@pytest.mark.parametrize("endpoint", ["trigger", "history"])
def test_missing_proactive_agent(secured_chat, auth_header, endpoint):
    secured_chat.db.aiagent.find_unique.return_value = None
    assert _request(secured_chat, endpoint, auth_header("owner")).status_code == 404
    _no_business(secured_chat)


@pytest.mark.parametrize("role", ["user", "admin"])
@pytest.mark.parametrize("route", ["immediate", "aggregated", "aggregation_fallback"])
def test_authorized_chat_preserves_sse_and_queue_payload(secured_chat, auth_header, role, route):
    s = secured_chat
    plan = s.business["plan_user_message_aggregation"].return_value
    plan.should_wait = route != "immediate"
    plan.metadata = {"fragment": True} if plan.should_wait else {}
    s.business["enqueue_planned_user_message"].return_value = route == "aggregated"
    response = _request(s, "chat", auth_header("admin" if role == "admin" else "owner", role))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: done") == 1
    assert response.text.count("event: pending") == 1
    s.db.conversation.find_unique.assert_awaited_once_with(where={"id": "conv-1"}, include={"agent": True})
    s.db.message.create.assert_awaited_once()
    saved = s.db.message.create.await_args.kwargs["data"]
    assert saved["content"] == "hello" and saved["role"] == "user"
    s.business["mark_user_replied_for_conversation"].assert_awaited_once_with("conv-1")
    s.business["fire_background"].assert_called_once()
    if route == "aggregated":
        assert '"status": "aggregating"' in response.text
        s.business["enqueue_or_append_delayed"].assert_not_called()
    else:
        context = plan.fallback_context if route == "aggregation_fallback" else plan.final_context
        delay = context["delay_seconds"]
        s.business["enqueue_or_append_delayed"].assert_awaited_once_with(
            "conv-1", {"conversation_id": "conv-1", "agent_id": "agent-1",
                       "user_id": "owner", "message": "hello", "message_id": "message-1",
                       "reply_context": context}, delay)
        assert '"status": "queued"' in response.text
        assert ("event: delay" in response.text) == (delay > 5)


@pytest.mark.parametrize("role", ["user", "admin"])
def test_authorized_trigger_preserves_contract(secured_chat, auth_header, role):
    s = secured_chat
    response = _request(s, "trigger", auth_header("admin" if role == "admin" else "owner", role))
    assert response.status_code == 200 and response.json() == {"message": "hi"}
    s.business["resolve_workspace_id"].assert_awaited_once_with(user_id="owner", agent_id="agent-1")
    s.business["send_manual_or_triggered_proactive"].assert_awaited_once_with(
        workspace_id="workspace-1", trigger_type="manual_trigger")
    s.db.aiagent.find_unique.assert_awaited_once()


@pytest.mark.parametrize("case,body", [
    ("workspace_missing", {"message": None, "reason": "workspace_not_found"}),
    ("limited", {"message": None, "reason": "no_content_or_limit_reached"}),
])
def test_authorized_trigger_empty_results_unchanged(secured_chat, auth_header, case, body):
    if case == "workspace_missing":
        secured_chat.business["resolve_workspace_id"].return_value = None
    else:
        secured_chat.business["send_manual_or_triggered_proactive"].return_value = {"ok": False}
    response = _request(secured_chat, "trigger", auth_header("owner"))
    assert response.status_code == 200 and response.json() == body
    if case == "workspace_missing":
        secured_chat.business["send_manual_or_triggered_proactive"].assert_not_called()


@pytest.mark.parametrize("role", ["user", "admin"])
@pytest.mark.parametrize("workspace", [None, "workspace-1"])
def test_authorized_history_preserves_limit_and_workspace(secured_chat, auth_header, role, workspace):
    s = secured_chat
    s.business["resolve_workspace_id"].return_value = workspace
    response = s.client.get("/chat/proactive/agent-1/history", params={"user_id": "owner", "limit": 3},
                            headers=auth_header("admin" if role == "admin" else "owner", role))
    assert response.status_code == 200
    assert response.json() == {"history": [{"content": "synthetic history"}]}
    s.business["get_proactive_history"].assert_awaited_once_with("agent-1", "owner", 3, workspace_id=workspace)


@pytest.mark.parametrize("endpoint", ["chat", "trigger"])
def test_redis_unavailable_still_blocks_write_paths(secured_chat, auth_header, monkeypatch, endpoint):
    from app.api import deps
    monkeypatch.setattr(deps, "is_redis_healthy", lambda: False)
    assert _request(secured_chat, endpoint, auth_header("owner")).status_code == 503
    _no_business(secured_chat)


def test_history_remains_readable_when_redis_unavailable(secured_chat, auth_header, monkeypatch):
    from app.api import deps
    monkeypatch.setattr(deps, "is_redis_healthy", lambda: False)
    assert _request(secured_chat, "history", auth_header("owner")).status_code == 200
