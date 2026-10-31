"""Direct source clock reaches actual memory and trigger SQL; faults never confirm."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from tests.test_memory_evidence_postgres import origins
from tests.test_reminder_day_clock import _record_ctx
from app.services.chat import intent_handlers as ih
from app.services.time_domain import clock
from app.services.time_domain.context import build_turn_clock
from app.services.reminder import scheduling
from app.services.memory.storage import persistence
from app.services.workspace import workspaces


@pytest.mark.parametrize(
    "mode", ["success", "expired", "trigger_failure", "storage_failure"]
)
async def test_direct_clock_and_trigger_are_consistent_in_sql(
    origins, monkeypatch, mode
):
    db, uid, wid, _, message, *_ = origins
    source = datetime(2026, 10, 31, 15, 59, 50, tzinfo=timezone.utc)
    now = source + timedelta(seconds=30)
    conversation = await db.conversation.find_unique(
        where={"id": message.conversationId}
    )
    message = await db.message.update(
        where={"id": message.id}, data={"createdAt": source}
    )
    ctx = _record_ctx(now)
    ctx.user_id = uid
    ctx.agent_id = conversation.agentId
    ctx.conversation_id = conversation.id
    ctx.workspace_id = wid
    ctx.response_diagnostics = {}
    ctx.turn_clock = build_turn_clock(
        [{"id": message.id, "role": "user", "createdAt": message.createdAt}],
        exclude_ids={message.id},
        now=now,
    )
    monkeypatch.setattr(clock, "_now_corrected", lambda: now)
    monkeypatch.setattr(
        workspaces,
        "get_active_workspace",
        AsyncMock(return_value=SimpleNamespace(id=wid, agentId=conversation.agentId)),
    )
    monkeypatch.setattr(scheduling, "db", db)
    monkeypatch.setattr(persistence, "db", db)
    monkeypatch.setattr(scheduling, "notify_reminder_changed", AsyncMock())
    monkeypatch.setattr(
        persistence, "generate_embedding", AsyncMock(return_value=[0.01] * 1024)
    )
    monkeypatch.setattr(persistence, "store_embedding", AsyncMock())
    monkeypatch.setattr(
        persistence,
        "resolve_memory_write",
        AsyncMock(return_value=SimpleNamespace(action="insert_new")),
    )
    monkeypatch.setattr(persistence, "log_memory_changelog", AsyncMock())
    if mode == "trigger_failure":
        monkeypatch.setattr(
            scheduling, "upsert_reminder_trigger", AsyncMock(return_value=None)
        )
    if mode == "storage_failure":
        monkeypatch.setattr(
            persistence,
            "store_memory",
            AsyncMock(side_effect=RuntimeError("synthetic failure")),
        )
    before = await db.usermemory.count(where={"userId": uid})
    try:
        text = "1秒后提醒我" if mode == "expired" else "明天下午3点提醒我开会"
        status, _ = await ih._direct_create_reminder(user_message=text, ctx=ctx)
        triggers = await db.timetrigger.find_many(where={"userId": uid})
        if mode == "success":
            assert status == "scheduled" and len(triggers) == 1
            row = await db.usermemory.find_unique(
                where={"id": triggers[0].actionData["memory_id"]}
            )
            assert row.statementTime == source
            assert (
                row.occurTime
                == triggers[0].triggerTime
                == datetime(2026, 11, 1, 15, tzinfo=ZoneInfo("Asia/Shanghai"))
            )
        else:
            assert status == "failed" and not triggers
            if mode != "trigger_failure":
                assert await db.usermemory.count(where={"userId": uid}) == before
    finally:
        await db.timetrigger.delete_many(where={"userId": uid})
