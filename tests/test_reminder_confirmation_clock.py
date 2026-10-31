"""Receipt-anchored reminder confirmation through the real graph and handler."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from app.services.chat import preflight
from app.services.chat.temporal_context import build_turn_clock
from app.services.reminder import scheduling

UTC = timezone.utc
BJ = ZoneInfo("Asia/Shanghai")
RECEIVED = datetime(2026, 10, 31, 15, 59, 50, tzinfo=UTC)
PROCESSING = RECEIVED + timedelta(seconds=30)


def clock_for(*stamps):
    rows = [
        {"id": str(i), "role": "user", "createdAt": stamp}
        for i, stamp in enumerate(stamps)
    ]
    return build_turn_clock(rows, exclude_ids={r["id"] for r in rows}, now=PROCESSING)


@pytest.mark.parametrize(
    "text,target",
    [
        ("明天下午3点", datetime(2026, 11, 1, 15, tzinfo=BJ)),
        ("后天", datetime(2026, 11, 2, 0, tzinfo=BJ)),
        ("下周一上午10点", datetime(2026, 11, 2, 10, tzinfo=BJ)),
        ("15分钟后", RECEIVED + timedelta(minutes=15)),
        ("15分钟", RECEIVED + timedelta(minutes=15)),
        ("2027年1月1日8点", datetime(2027, 1, 1, 8, tzinfo=BJ)),
    ],
)
def test_target_uses_statement_instant_not_processing_clock(text, target):
    result = scheduling.resolve_confirmation_time(
        text, statement_at=RECEIVED, statement_end_at=RECEIVED, now=PROCESSING
    )
    assert result.reason == "resolved" and result.occur_time == target


@pytest.mark.parametrize(
    "text,delay,reason",
    [
        ("15分钟后", 900, "expired_before_processing"),
        ("15分钟后", 901, "expired_before_processing"),
        ("1秒", 30, "expired_before_processing"),
        ("明天", 30, "expired_before_processing"),
        ("昨天", 30, "past_or_unresolved"),
    ],
)
def test_expired_intent_is_never_rebased_to_a_new_future(text, delay, reason):
    result = scheduling.resolve_confirmation_time(
        text,
        statement_at=RECEIVED,
        statement_end_at=RECEIVED,
        now=RECEIVED + timedelta(seconds=delay),
    )
    assert (
        result.has_expression and result.occur_time is None and result.reason == reason
    )


@pytest.mark.parametrize(
    "start,end,reason",
    [
        (None, None, "missing_statement_clock"),
        (RECEIVED, None, "invalid_statement_interval"),
        (RECEIVED, RECEIVED - timedelta(seconds=1), "invalid_statement_interval"),
        (
            PROCESSING + timedelta(seconds=1),
            PROCESSING + timedelta(seconds=1),
            "invalid_statement_interval",
        ),
        (RECEIVED, PROCESSING + timedelta(seconds=1), "invalid_statement_interval"),
        (RECEIVED - timedelta(seconds=2), RECEIVED, "ambiguous_statement_interval"),
    ],
)
def test_unknown_or_ambiguous_offset_is_not_scheduled(start, end, reason):
    result = scheduling.resolve_confirmation_time(
        "15分钟后", statement_at=start, statement_end_at=end, now=PROCESSING
    )
    assert result.occur_time is None and result.reason == reason


@pytest.mark.parametrize("text", ["明天下午3点", "下周一上午10点"])
def test_same_day_aggregate_accepts_only_identical_calendar_target(text):
    result = scheduling.resolve_confirmation_time(
        text,
        statement_at=RECEIVED - timedelta(seconds=2),
        statement_end_at=RECEIVED,
        now=PROCESSING,
    )
    assert result.reason == "resolved"


@pytest.mark.parametrize("anchor", [None, RECEIVED])
def test_unrelated_conversation_is_not_blocked_without_clock(anchor):
    result = scheduling.resolve_confirmation_time(
        "聊点别的吧", statement_at=anchor, statement_end_at=anchor, now=PROCESSING
    )
    assert not result.has_expression and result.reason == "no_expression"


@pytest.fixture
def handler_io(monkeypatch):
    from app.services.workspace import workspaces

    monkeypatch.setattr(preflight, "_now_corrected", lambda: PROCESSING)
    monkeypatch.setattr(
        workspaces,
        "get_active_workspace",
        AsyncMock(return_value=SimpleNamespace(id="w-1")),
    )
    create = AsyncMock(return_value="m-1")
    monkeypatch.setattr(scheduling, "create_user_reminder", create)
    monkeypatch.setattr(preflight, "clear_pending_deletion", AsyncMock())
    monkeypatch.setattr(
        preflight, "record_confirm_reply", AsyncMock(return_value="记好啦")
    )
    ctx = preflight.PreflightCtx(
        conversation_id="c-1",
        user_id="u-1",
        agent_id="a-1",
        agent=SimpleNamespace(name="Synthetic"),
        tracer=MagicMock(safe_trace_id="t"),
        short_circuit_fn=AsyncMock(return_value=[{"event": "reply"}]),
        message_ids=("0",),
    )
    # Dynamic attributes allow the old handler to be replayed as a baseline.
    ctx.turn_clock = clock_for(RECEIVED)
    ctx.response_diagnostics = {}
    return SimpleNamespace(ctx=ctx, create=create)


async def invoke(io, text):
    return [
        e
        async for e in preflight._handle_pending_set_reminder(
            text, {"summary": "合成事项"}, io.ctx
        )
    ]


async def test_handler_persists_source_time_and_formats_from_execution(handler_io):
    await invoke(handler_io, "明天下午3点")
    kw = handler_io.create.await_args.kwargs
    assert kw["occur_time"] == datetime(2026, 11, 1, 15, tzinfo=BJ)
    assert kw["statement_time"] == RECEIVED
    assert (
        preflight.record_confirm_reply.await_args.kwargs["when_text"]
        == "今天下午 15:00"
    )
    diag = handler_io.ctx.response_diagnostics["reminder_confirmation_time"]
    assert (
        diag["creation_status"] == "created"
        and diag["anchor_source"] == "persisted_message"
    )
    assert "合成事项" not in str(diag)


@pytest.mark.parametrize(
    "stamps",
    [
        (None,),
        ("invalid",),
        (PROCESSING + timedelta(seconds=2),),
        (RECEIVED, PROCESSING),
        (RECEIVED - timedelta(seconds=2), RECEIVED),
    ],
)
async def test_handler_unknown_and_aggregated_offset_never_write(handler_io, stamps):
    handler_io.ctx.turn_clock = clock_for(*stamps)
    await invoke(handler_io, "15分钟后")
    handler_io.create.assert_not_awaited()
    assert handler_io.ctx.stopped
    assert (
        handler_io.ctx.response_diagnostics["reminder_confirmation_time"][
            "creation_status"
        ]
        == "not_attempted"
    )


async def test_handler_no_clock_with_message_ids_cannot_use_execution(handler_io):
    handler_io.ctx.turn_clock = None
    await invoke(handler_io, "15分钟后")
    handler_io.create.assert_not_awaited()


async def test_handler_rechecks_expiration_after_workspace_io(handler_io, monkeypatch):
    ticks = iter([PROCESSING, RECEIVED + timedelta(minutes=16)])
    monkeypatch.setattr(preflight, "_now_corrected", lambda: next(ticks))
    await invoke(handler_io, "15分钟后")
    handler_io.create.assert_not_awaited()
    assert (
        handler_io.ctx.response_diagnostics["reminder_confirmation_time"]["reason"]
        == "expired_before_write"
    )


async def test_handler_failed_write_does_not_claim_creation(handler_io):
    handler_io.create.return_value = None
    await invoke(handler_io, "15分钟后")
    preflight.record_confirm_reply.assert_not_awaited()
    assert (
        handler_io.ctx.response_diagnostics["reminder_confirmation_time"][
            "creation_status"
        ]
        == "failed"
    )


async def test_cancel_still_works_without_a_valid_anchor(handler_io):
    handler_io.ctx.turn_clock = clock_for(None)
    await invoke(handler_io, "算了")
    handler_io.create.assert_not_awaited()
    preflight.clear_pending_deletion.assert_awaited_once()
    assert handler_io.ctx.stopped


async def test_real_graph_passes_persisted_clock_and_saves_diagnostics(monkeypatch):
    from tests.graph_harness_support import configure_chat
    from app.services.chat import orchestrator as chat, graph_phases

    io = configure_chat(monkeypatch)
    io.db.message.find_many.side_effect = None
    io.db.message.find_many.return_value = [
        SimpleNamespace(
            id="u-new",
            role="user",
            content="明天下午3点",
            createdAt=RECEIVED,
            metadata={},
        )
    ]
    monkeypatch.setattr(
        graph_phases,
        "build_turn_clock",
        lambda *a, **kw: build_turn_clock(*a, now=PROCESSING, **kw),
    )
    monkeypatch.setattr(
        chat, "resolve_pending_deletion", preflight.resolve_pending_deletion
    )
    monkeypatch.setattr(
        preflight,
        "load_pending_action",
        AsyncMock(return_value={"action": "set_reminder", "summary": "合成事项"}),
    )
    monkeypatch.setattr(preflight, "_now_corrected", lambda: PROCESSING)
    monkeypatch.setattr(preflight, "clear_pending_deletion", AsyncMock())
    monkeypatch.setattr(
        preflight, "record_confirm_reply", AsyncMock(return_value="记好啦")
    )
    from app.services.workspace import workspaces

    monkeypatch.setattr(
        workspaces,
        "get_active_workspace",
        AsyncMock(return_value=SimpleNamespace(id="w-1")),
    )
    create = AsyncMock(return_value="m-1")
    monkeypatch.setattr(scheduling, "create_user_reminder", create)
    events = [
        e
        async for e in chat.stream_chat_response(
            "c-1",
            "明天下午3点",
            io.agent,
            "u-1",
            user_message_id="u-new",
            reply_context={"received_at": PROCESSING.isoformat()},
        )
    ]
    assert [e["event"] for e in events] == ["reply", "done"]
    assert create.await_args.kwargs["statement_time"] == RECEIVED
    assert create.await_args.kwargs["occur_time"] == datetime(
        2026, 11, 1, 15, tzinfo=BJ
    )
    chat._save_replies.assert_awaited_once()
    saved = chat._save_replies.await_args.args[1]
    assert (
        saved[0]["response_diagnostics"]["reminder_confirmation_time"][
            "creation_status"
        ]
        == "created"
    )
    chat.finish_assistant_turn.assert_awaited_once()


@pytest.mark.parametrize("text", ["今年13月", "9999年12月32日 1点"])
def test_invalid_calendar_input_is_bounded_and_never_scheduled(text):
    result = scheduling.resolve_confirmation_time(
        text, statement_at=RECEIVED, statement_end_at=RECEIVED, now=PROCESSING
    )
    assert result.occur_time is None


def test_invalid_execution_clock_cannot_approve_a_write():
    result = scheduling.resolve_confirmation_time(
        "明天", statement_at=RECEIVED, statement_end_at=RECEIVED, now=None
    )
    assert result.reason == "invalid_execution_clock" and result.occur_time is None


async def test_handler_processing_expiry_does_not_create_or_confirm(handler_io):
    await invoke(handler_io, "1秒")
    handler_io.create.assert_not_awaited()
    preflight.record_confirm_reply.assert_not_awaited()
    assert (
        handler_io.ctx.response_diagnostics["reminder_confirmation_time"]["reason"]
        == "expired_before_processing"
    )


async def test_handler_unrelated_message_continues_normal_chat(handler_io):
    await invoke(handler_io, "聊点别的吧")
    assert not handler_io.ctx.stopped
    handler_io.ctx.short_circuit_fn.assert_not_awaited()
    handler_io.create.assert_not_awaited()
    preflight.clear_pending_deletion.assert_awaited_once()
