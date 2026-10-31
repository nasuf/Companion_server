"""Actual confirmation -> memory storage -> trigger SQL on owned synthetic data."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from app.services.chat import preflight
from app.services.chat.temporal_context import build_turn_clock
from app.services.reminder import scheduling
from app.services.memory.storage import persistence as store
from tests.test_memory_evidence_postgres import origins

UTC = timezone.utc
BJ = ZoneInfo("Asia/Shanghai")
T = datetime(2026, 10, 31, 15, 59, 50, tzinfo=UTC)


@pytest.mark.parametrize(
    "mode", ["success", "expired", "trigger_failure", "storage_failure"]
)
async def test_source_clock_survives_real_storage_and_failed_creation_never_confirms(
    origins, monkeypatch, mode
):
    db, uid, wid, _, message, *_ = origins
    message = await db.message.update(where={"id": message.id}, data={"createdAt": T})
    conversation = await db.conversation.find_unique(
        where={"id": message.conversationId}
    )
    now = T + timedelta(seconds=30)
    clock = build_turn_clock(
        [{"id": message.id, "role": "user", "createdAt": message.createdAt}],
        exclude_ids={message.id},
        now=now,
    )
    monkeypatch.setattr(preflight, "_now_corrected", lambda: now)
    monkeypatch.setattr(preflight, "clear_pending_deletion", AsyncMock())
    monkeypatch.setattr(
        preflight, "record_confirm_reply", AsyncMock(return_value="记好啦")
    )
    from app.services.workspace import workspaces

    monkeypatch.setattr(
        workspaces,
        "get_active_workspace",
        AsyncMock(return_value=SimpleNamespace(id=wid)),
    )
    monkeypatch.setattr(scheduling, "db", db)
    monkeypatch.setattr(store, "db", db)
    monkeypatch.setattr(
        store, "generate_embedding", AsyncMock(return_value=[0.01] * 1024)
    )
    monkeypatch.setattr(store, "store_embedding", AsyncMock())
    monkeypatch.setattr(
        store,
        "resolve_memory_write",
        AsyncMock(return_value=SimpleNamespace(action="insert_new")),
    )
    monkeypatch.setattr(store, "log_memory_changelog", AsyncMock())
    if mode == "trigger_failure":
        monkeypatch.setattr(
            scheduling, "upsert_reminder_trigger", AsyncMock(return_value=None)
        )
    if mode == "storage_failure":
        monkeypatch.setattr(
            store,
            "store_memory",
            AsyncMock(side_effect=RuntimeError("synthetic storage failure")),
        )
    ctx = preflight.PreflightCtx(
        conversation_id=conversation.id,
        user_id=uid,
        agent_id=conversation.agentId,
        workspace_id=wid,
        message_ids=(message.id,),
        agent=SimpleNamespace(name="Synthetic"),
        tracer=MagicMock(safe_trace_id=None),
        short_circuit_fn=AsyncMock(return_value=[{"event": "reply"}]),
        turn_clock=clock,
        response_diagnostics={},
    )
    before = await db.usermemory.count(where={"userId": uid})
    try:
        text = "1秒" if mode == "expired" else "明天下午3点"
        events = [
            e
            async for e in preflight._handle_pending_set_reminder(
                text, {"summary": "合成提醒时间"}, ctx
            )
        ]
        assert len(events) == 1 and ctx.stopped
        triggers = await db.timetrigger.find_many(where={"userId": uid})
        diag = ctx.response_diagnostics["reminder_confirmation_time"]
        if mode == "success":
            assert len(triggers) == 1
            row = await db.usermemory.find_unique(
                where={"id": triggers[0].actionData["memory_id"]}
            )
            assert row.statementTime == T
            target = datetime(2026, 11, 1, 15, tzinfo=BJ)
            assert row.occurTime == triggers[0].triggerTime == target
            assert diag["creation_status"] == "created"
            preflight.record_confirm_reply.assert_awaited_once()
        else:
            assert not triggers
            preflight.record_confirm_reply.assert_not_awaited()
            assert diag["creation_status"] == (
                "not_attempted" if mode == "expired" else "failed"
            )
            if mode != "trigger_failure":
                assert await db.usermemory.count(where={"userId": uid}) == before
    finally:
        await db.timetrigger.delete_many(where={"userId": uid})
