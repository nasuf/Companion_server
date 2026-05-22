from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"sub": "admin-1", "role": "admin"}
    return app, require_admin_jwt


def _repair_row(**overrides):
    row = {
        "id": "repair-1",
        "source_type": "bug_report_memory_safety",
        "source_id": "bug-1",
        "status": "open",
        "severity": "high",
        "user_id": "u1",
        "agent_id": "a1",
        "workspace_id": "w1",
        "conversation_id": "c1",
        "message_id": "m1",
        "memory_id": "mem1",
        "memory_source": "user",
        "reason": "记忆错误",
        "suggested_action": "review",
        "evidence": {"k": "v"},
        "resolution_note": None,
        "resolved_by_id": None,
        "resolved_at": None,
        "created_at": datetime(2026, 5, 22, 9, 0, 0),
        "updated_at": datetime(2026, 5, 22, 9, 0, 0),
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_create_memory_repair_item_dedupes_existing_source(monkeypatch):
    from app.services.memory import repair_queue

    fake_db = SimpleNamespace(query_raw=AsyncMock(return_value=[_repair_row()]))
    monkeypatch.setattr(repair_queue, "db", fake_db)

    item = await repair_queue.create_memory_repair_item(
        source_type="bug_report_memory_safety",
        source_id="bug-1",
        severity="high",
        reason="记忆错误",
    )

    assert item["id"] == "repair-1"
    assert item["evidence"] == {"k": "v"}
    assert fake_db.query_raw.await_count == 1


@pytest.mark.asyncio
async def test_create_memory_repair_item_inserts_when_no_existing(monkeypatch):
    from app.services.memory import repair_queue

    fake_db = SimpleNamespace(query_raw=AsyncMock(side_effect=[[], [_repair_row(source_id="bug-2")]]))
    monkeypatch.setattr(repair_queue, "db", fake_db)

    item = await repair_queue.create_memory_repair_item(
        source_type="bug_report_memory_safety",
        source_id="bug-2",
        severity="invalid",
        evidence={"reason": "bad memory"},
    )

    assert item["source_id"] == "bug-2"
    insert_args = fake_db.query_raw.await_args_list[1].args
    assert "INSERT INTO memory_repair_items" in insert_args[0]
    assert insert_args[3] == "bug-2"
    assert insert_args[4] == "medium"


@pytest.mark.asyncio
async def test_update_memory_repair_item_closes_with_resolver(monkeypatch):
    from app.services.memory import repair_queue

    fake_db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[_repair_row(
            status="resolved",
            resolution_note="已修复",
            resolved_by_id="admin-1",
            resolved_at=datetime(2026, 5, 22, 10, 0, 0),
        )]),
    )
    monkeypatch.setattr(repair_queue, "db", fake_db)

    item = await repair_queue.update_memory_repair_item_status(
        "repair-1",
        status="resolved",
        resolution_note="已修复",
        resolved_by_id="admin-1",
    )

    assert item is not None
    assert item["status"] == "resolved"
    assert item["resolved_by_id"] == "admin-1"
    assert item["resolved_at"] == "2026-05-22T10:00:00"


@pytest.mark.asyncio
async def test_retrieval_feedback_unresolved_creates_repair_item(monkeypatch):
    from app.services.memory.interaction import retrieval_feedback

    created = AsyncMock()
    monkeypatch.setattr(retrieval_feedback.memory_repo, "find_unique", AsyncMock(return_value=None))
    monkeypatch.setattr(retrieval_feedback, "best_effort_create_memory_repair_item", created)

    previous = SimpleNamespace(
        id="assistant-1",
        conversationId="conv-1",
        role="assistant",
        content="我记得你喜欢芒果。",
        metadata={
            "memory_retrieval_analysis": {"likely_used_count": 1},
            "memory_retrievals": [{"selected": [{"id": "missing-memory"}]}],
        },
    )

    result = await retrieval_feedback.build_retrieval_feedback_conflict(
        user_message="你记错了，我从来没说过我喜欢芒果。",
        previous_assistant=previous,
        user_id="u1",
        workspace_id="w1",
    )

    assert result is None
    created.assert_awaited_once()
    kwargs = created.await_args.kwargs
    assert kwargs["source_type"] == "retrieval_feedback_unresolved"
    assert kwargs["source_id"] == "assistant-1"
    assert kwargs["conversation_id"] == "conv-1"


@pytest.mark.asyncio
async def test_bug_report_memory_safety_creates_repair_item(monkeypatch):
    from app.api.admin import bug_reports

    msg = SimpleNamespace(
        id="assistant-msg",
        conversationId="conv-1",
        content="我记得你喜欢芒果。",
        metadata={"memory_retrieval_analysis": {"likely_used_count": 1}},
    )
    report = SimpleNamespace(
        id="bug-1",
        errorTypes=["记忆编造"],
        reason="AI 编造了用户喜好",
    )
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(return_value=[{
        "id": "conv-1",
        "user_id": "u1",
        "agent_id": "agent-1",
        "workspace_id": "w1",
    }])
    created = AsyncMock()
    monkeypatch.setattr(bug_reports, "db", fake_db)
    monkeypatch.setattr(bug_reports, "best_effort_create_memory_repair_item", created)

    await bug_reports._maybe_create_memory_repair_from_bug_report(report, msg)

    created.assert_awaited_once()
    kwargs = created.await_args.kwargs
    assert kwargs["source_type"] == "bug_report_memory_safety"
    assert kwargs["source_id"] == "bug-1"
    assert kwargs["severity"] == "high"
    assert kwargs["user_id"] == "u1"


def test_admin_memory_repairs_list_endpoint(api_client):
    app, require_admin_jwt = _admin_override()

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "app.api.admin.memory_repairs.list_memory_repair_items",
                AsyncMock(return_value=[_repair_row()]),
            )
            response = api_client.get("/admin-api/memory-repairs?status=open&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["items"][0]["id"] == "repair-1"
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_admin_memory_repairs_update_endpoint(api_client):
    app, require_admin_jwt = _admin_override()

    try:
        with pytest.MonkeyPatch.context() as mp:
            update = AsyncMock(return_value=_repair_row(
                status="resolved",
                resolution_note="已人工修复",
                resolved_by_id="admin-1",
            ))
            mp.setattr(
                "app.api.admin.memory_repairs.update_memory_repair_item_status",
                update,
            )
            response = api_client.patch(
                "/admin-api/memory-repairs/repair-1",
                json={"status": "resolved", "resolution_note": "已人工修复"},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "resolved"
        update.assert_awaited_once()
        assert update.await_args.kwargs["resolved_by_id"] == "admin-1"
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
