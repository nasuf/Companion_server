"""Direct reminder writes use graph source evidence and actual execution time."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.services.chat import intent_handlers as ih
from app.services.time_domain import clock
from app.services.time_domain.context import build_turn_clock
from app.services.reminder import scheduling
from app.services.workspace import workspaces
from tests.test_reminder_day_clock import _record_ctx

SOURCE = datetime(2026, 10, 31, 15, 59, 50, tzinfo=timezone.utc)
NOW = SOURCE + timedelta(seconds=30)
BJ = ZoneInfo("Asia/Shanghai")
pytestmark = pytest.mark.usefixtures("unit_media_scope")


@pytest.fixture
def io(monkeypatch):
    ctx = _record_ctx(NOW)
    ctx.turn_clock = build_turn_clock(
        [{"id": "u", "role": "user", "createdAt": SOURCE}], exclude_ids={"u"}, now=NOW
    )
    ctx.response_diagnostics = {}
    monkeypatch.setattr(clock, "_now_corrected", lambda: NOW)
    monkeypatch.setattr(
        workspaces,
        "get_active_workspace",
        AsyncMock(return_value=SimpleNamespace(id="w", agentId="a")),
    )
    persist = AsyncMock(return_value="memory")
    trigger = AsyncMock(return_value="trigger")
    monkeypatch.setattr(ih, "_persist_one_reminder", persist)
    monkeypatch.setattr(scheduling, "upsert_reminder_trigger", trigger)
    monkeypatch.setattr(scheduling, "notify_reminder_changed", AsyncMock())
    return SimpleNamespace(ctx=ctx, persist=persist, trigger=trigger)


@pytest.mark.parametrize(
    "message,targets,recurrence",
    [
        ("明天下午3点提醒我", [datetime(2026, 11, 1, 15, tzinfo=BJ)], "once"),
        ("15分钟后提醒我", [SOURCE + timedelta(minutes=15)], "once"),
        ("明天8点提醒我吃药", [datetime(2026, 11, 1, 8, tzinfo=BJ)], "once"),
    ],
)
async def test_direct_targets_follow_source_not_processing(
    io, message, targets, recurrence
):
    result = await ih._direct_create_reminder(user_message=message, ctx=io.ctx)
    assert result[0] == "scheduled"
    assert [c.kwargs["trigger_time"] for c in io.trigger.await_args_list] == targets
    assert all(
        c.kwargs["statement_time"] == SOURCE and c.kwargs["recurrence"] == recurrence
        for c in io.persist.await_args_list
    )
    assert (
        io.ctx.response_diagnostics["reminder_direct_time"]["anchor_source"]
        == "persisted_message"
    )


@pytest.mark.parametrize(
    "message,reason", [("15秒后提醒我", "expired_before_processing")]
)
async def test_expired_is_not_shifted(io, message, reason):
    assert (await ih._direct_create_reminder(user_message=message, ctx=io.ctx))[
        0
    ] == "failed"
    io.persist.assert_not_awaited()
    assert io.ctx.response_diagnostics["reminder_direct_time"]["reason"] == reason


async def test_unknown_clock_does_not_fallback_to_receipt(io):
    io.ctx.turn_clock = build_turn_clock([], exclude_ids={"missing"}, now=NOW)
    assert (
        await ih._direct_create_reminder(user_message="15分钟后提醒我", ctx=io.ctx)
    )[0] == "failed"
    io.persist.assert_not_awaited()


async def test_relative_aggregate_remains_ambiguous(io):
    io.ctx.turn_clock = build_turn_clock(
        [
            {"id": str(i), "role": "user", "createdAt": SOURCE + timedelta(seconds=i)}
            for i in range(2)
        ],
        exclude_ids={"0", "1"},
        now=NOW,
    )
    assert (
        await ih._direct_create_reminder(user_message="15分钟后提醒我", ctx=io.ctx)
    )[0] == "failed"
    assert (
        io.ctx.response_diagnostics["reminder_direct_time"]["reason"]
        == "ambiguous_statement_interval"
    )
    io.persist.assert_not_awaited()


async def test_workspace_latency_cannot_make_expired_write(io, monkeypatch):
    monkeypatch.setattr(
        clock, "_now_corrected", iter([NOW, SOURCE + timedelta(minutes=20)]).__next__
    )
    assert (
        await ih._direct_create_reminder(user_message="15分钟后提醒我", ctx=io.ctx)
    )[0] == "failed"
    assert (
        io.ctx.response_diagnostics["reminder_direct_time"]["reason"]
        == "expired_before_write"
    )
    io.persist.assert_not_awaited()


async def test_real_graph_passes_source_clock_to_direct_handler(monkeypatch):
    from tests.graph_harness_support import configure_chat
    from app.services.chat import graph_phases, orchestrator as chat
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    fixture = configure_chat(monkeypatch)
    fixture.db.message.find_many.side_effect = None
    fixture.db.message.find_many.return_value = [
        SimpleNamespace(
            id="u-new",
            role="user",
            content="明天下午3点提醒我",
            createdAt=SOURCE,
            metadata={},
        )
    ]
    monkeypatch.setattr(
        graph_phases,
        "build_turn_clock",
        lambda *a, **kw: build_turn_clock(*a, now=NOW, **kw),
    )
    monkeypatch.setattr(clock, "_now_corrected", lambda: NOW)
    monkeypatch.setattr(
        chat,
        "detect_intent_unified",
        AsyncMock(
            return_value=IntentResult(intent=IntentType.RECORD_REQUEST, confidence=1)
        ),
    )
    monkeypatch.setattr(
        workspaces,
        "get_active_workspace",
        AsyncMock(return_value=SimpleNamespace(id="w-1", agentId="a-1")),
    )
    persist = AsyncMock(return_value="m1")
    trigger = AsyncMock(return_value="t1")
    monkeypatch.setattr(ih, "_persist_one_reminder", persist)
    monkeypatch.setattr(scheduling, "upsert_reminder_trigger", trigger)
    monkeypatch.setattr(scheduling, "notify_reminder_changed", AsyncMock())
    monkeypatch.setattr(ih, "record_confirm_reply", AsyncMock(return_value="已设置"))
    events = [
        e
        async for e in chat.stream_chat_response(
            "c-1",
            "明天下午3点提醒我",
            fixture.agent,
            "u-1",
            user_message_id="u-new",
            reply_context={"received_at": NOW.isoformat()},
        )
    ]
    assert [e["event"] for e in events] == ["reply", "done"]
    assert persist.await_args.kwargs["statement_time"] == SOURCE
    assert trigger.await_args.kwargs["trigger_time"] == datetime(
        2026, 11, 1, 15, tzinfo=BJ
    )
    saved = chat._save_replies.await_args.args[1]
    assert (
        saved[0]["response_diagnostics"]["reminder_direct_time"]["creation_status"]
        == "created"
    )


async def test_daily_future_clock_keeps_recurrence(io, monkeypatch):
    anchor = datetime(2026, 10, 31, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(clock, "_now_corrected", lambda: anchor + timedelta(seconds=30))
    io.ctx.turn_clock = build_turn_clock(
        [{"id": "u", "role": "user", "createdAt": anchor}],
        exclude_ids={"u"},
        now=anchor + timedelta(seconds=30),
    )
    assert (await ih._direct_create_reminder(user_message="每天9点提醒我", ctx=io.ctx))[
        0
    ] == "scheduled"
    assert io.persist.await_args.kwargs["recurrence"] == "daily"
    assert io.trigger.await_args.kwargs["trigger_time"] == datetime(
        2026, 10, 31, 9, tzinfo=BJ
    )


async def test_daily_clock_already_past_schedules_next_occurrence(io):
    assert (
        await ih._direct_create_reminder(user_message="每天8点提醒我", ctx=io.ctx)
    )[0] == "scheduled"
    assert io.trigger.await_args.kwargs["trigger_time"] == datetime(2026, 11, 1, 8, tzinfo=BJ)
    assert io.trigger.await_args.kwargs["recurrence"] == "daily"


@pytest.mark.parametrize("outcome", [None, RuntimeError("synthetic trigger failure")])
async def test_partial_creation_reports_exact_successes_without_inviting_retry(
    io, monkeypatch, outcome
):
    anchor = datetime(2026, 11, 1, 0, tzinfo=BJ)
    monkeypatch.setattr(clock, "_now_corrected", lambda: anchor + timedelta(seconds=30))
    io.ctx.turn_clock = build_turn_clock(
        [{"id": "u", "role": "user", "createdAt": anchor}],
        exclude_ids={"u"},
        now=anchor + timedelta(seconds=30),
    )
    io.trigger.side_effect = ["trigger-1", outcome]

    async def final(reply, **kwargs):
        yield {"event": "reply", "text": reply, "kind": kwargs["kind"]}

    io.ctx.finalize = final
    confirm = AsyncMock()
    monkeypatch.setattr(ih, "record_confirm_reply", confirm)
    handled, gen = await ih.handle_record_request("8点吃药，9点开会提醒我", io.ctx)
    events = [e async for e in gen]
    assert handled and events[0]["kind"] == "record_request_failed"
    # Names both the time that was set and the one to repeat.
    assert "只设好了" in events[0]["text"] and "没设上" in events[0]["text"]
    assert "08:00" in events[0]["text"] and "09:00" in events[0]["text"]
    assert "你再跟我说一遍" not in events[0]["text"]
    confirm.assert_not_awaited()
    diagnostic = io.ctx.response_diagnostics["reminder_direct_time"]
    assert diagnostic["creation_status"] == "partial"
    assert diagnostic["created_count"] == 1 and diagnostic["target_count"] == 2
    assert len(diagnostic["created_targets"]) == 1


async def test_expired_direct_reminder_says_the_time_has_passed(io, monkeypatch):
    async def final(reply, **kwargs):
        yield {"event": "reply", "text": reply, "kind": kwargs["kind"]}

    io.ctx.finalize = final
    monkeypatch.setattr(ih, "record_confirm_reply", AsyncMock())
    handled, gen = await ih.handle_record_request("15秒后提醒我", io.ctx)
    events = [e async for e in gen]
    assert handled and events[0]["kind"] == "record_request_failed"
    assert events[0]["text"] == ih.EXPIRED_REMINDER_REPLY
    io.persist.assert_not_awaited()
