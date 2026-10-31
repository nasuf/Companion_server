"""Next daily occurrence is chosen at source time, never moved by queue latency."""

from datetime import datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.services.time_domain import reminders, clock
from app.services.chat import intent_handlers as ih, preflight
from app.services.memory.interaction import deletion
from tests.test_direct_reminder_clock import io
from tests.test_reminder_confirmation_clock import handler_io, clock_for, RECEIVED

BJ = ZoneInfo("Asia/Shanghai")


@pytest.mark.parametrize("hour,day", [(7, 31), (8, 1), (9, 1), (23, 1)])
@pytest.mark.parametrize(
    "text", ["每天早上8点提醒我吃药", "每日8点提醒我", "天天上午8点提醒我"]
)
def test_daily_next_occurrence(hour, day, text):
    source = datetime(2026, 10, 31, hour, tzinfo=BJ)
    r = reminders.resolve_reminder_targets(
        text,
        statement_at=source,
        statement_end_at=source,
        now=source,
        recurrence="daily",
    )
    assert r.targets == (datetime(2026, 10 if day == 31 else 11, day, 8, tzinfo=BJ),)


@pytest.mark.parametrize(
    "text,recurrence",
    [
        ("8点提醒我", "once"),
        ("今天8点每天提醒我", "daily"),
        ("昨天8点每天提醒我", "daily"),
        ("2026年10月30日8点每天提醒我", "daily"),
    ],
)
def test_no_roll_for_once_or_explicit_date(text, recurrence):
    source = datetime(2026, 10, 31, 10, tzinfo=BJ)
    r = reminders.resolve_reminder_targets(
        text,
        statement_at=source,
        statement_end_at=source,
        now=source,
        recurrence=recurrence,
    )
    assert not r.targets


def test_daily_expired_in_queue_is_not_shifted_to_tomorrow():
    source = datetime(2026, 10, 31, 7, 59, tzinfo=BJ)
    r = reminders.resolve_reminder_targets(
        "每天8点提醒我",
        statement_at=source,
        statement_end_at=source,
        now=source + timedelta(minutes=2),
        recurrence="daily",
    )
    assert not r.targets and r.reason == "expired_before_processing"


def test_daily_half_hour_and_year_rollover():
    source = datetime(2026, 12, 31, 23, tzinfo=BJ)
    r = reminders.resolve_confirmation_time(
        "早上8点半",
        statement_at=source,
        statement_end_at=source,
        now=source,
        recurrence="daily",
    )
    assert r.occur_time == datetime(2027, 1, 1, 8, 30, tzinfo=BJ)


def test_daily_dst_invalid_next_clock_is_not_guessed(monkeypatch):
    from app.config import settings

    zone = ZoneInfo("America/New_York")
    monkeypatch.setattr(clock, "_TZ", zone)
    monkeypatch.setattr(settings, "schedule_timezone", "America/New_York")
    source = datetime(2026, 3, 7, 12, tzinfo=zone)
    r = reminders.resolve_confirmation_time(
        "每天凌晨2点30分",
        statement_at=source,
        statement_end_at=source,
        now=source,
        recurrence="daily",
    )
    assert r.occur_time is None


async def test_direct_ambiguous_explains_without_writing(io):
    from app.services.time_domain.context import build_turn_clock

    source = io.ctx.turn_clock.statement_at
    io.ctx.turn_clock = build_turn_clock(
        [
            {"id": "0", "role": "user", "createdAt": source},
            {"id": "1", "role": "user", "createdAt": source + timedelta(seconds=1)},
        ],
        exclude_ids={"0", "1"},
        now=source + timedelta(seconds=30),
    )

    async def final(reply, **kw):
        yield {"reply": reply, **kw}

    io.ctx.finalize = final
    handled, gen = await ih.handle_record_request("10分钟后提醒我喝水", io.ctx)
    events = [e async for e in gen]
    assert handled and events[0]["reply"] == ih.AMBIGUOUS_REMINDER_REPLY
    io.persist.assert_not_awaited()


async def test_pending_ambiguous_has_same_explanation(handler_io):
    handler_io.ctx.turn_clock = clock_for(RECEIVED, RECEIVED + timedelta(seconds=1))
    _ = [
        e
        async for e in preflight._handle_pending_set_reminder(
            "10分钟后", {"summary": "喝水"}, handler_io.ctx
        )
    ]
    assert handler_io.ctx.last_short_circuit_reply == ih.AMBIGUOUS_REMINDER_REPLY
    handler_io.create.assert_not_awaited()


@pytest.mark.parametrize(
    "pending,text",
    [
        ({"summary": "每天提醒我吃药"}, "早上8点"),
        ({"summary": "吃药", "recurrence": "daily"}, "早上8点"),
        ({"summary": "吃药"}, "每天早上8点"),
    ],
)
async def test_pending_keeps_daily_and_confirms_it(handler_io, pending, text):
    _ = [
        e
        async for e in preflight._handle_pending_set_reminder(
            text, pending, handler_io.ctx
        )
    ]
    kw = handler_io.create.await_args.kwargs
    assert kw["recurrence"] == "daily"
    assert kw["occur_time"] == datetime(2026, 11, 1, 8, tzinfo=BJ)
    assert preflight.record_confirm_reply.await_args.kwargs["is_recurring"] is True


async def test_pending_explicit_once_overrides_daily(handler_io):
    _ = [
        e
        async for e in preflight._handle_pending_set_reminder(
            "明天下午3点，只提醒一次",
            {"summary": "每天吃药", "recurrence": "daily"},
            handler_io.ctx,
        )
    ]
    assert handler_io.create.await_args.kwargs["recurrence"] == "once"
    assert preflight.record_confirm_reply.await_args.kwargs["is_recurring"] is False


async def test_pending_content_edit_preserves_daily(handler_io, monkeypatch):
    save = AsyncMock()
    monkeypatch.setattr(deletion, "save_pending_action", save)
    _ = [
        e
        async for e in preflight._handle_pending_set_reminder(
            "提醒内容改成吃药",
            {"summary": "每天提醒我喝水", "recurrence": "daily"},
            handler_io.ctx,
        )
    ]
    assert save.await_args.kwargs["recurrence"] == "daily"
    handler_io.create.assert_not_awaited()


@pytest.mark.parametrize("recurrence", ["daily", "once", {}, "invalid"])
async def test_pending_recurrence_roundtrip(monkeypatch, recurrence):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def scope(*args):
        yield

    monkeypatch.setattr(deletion.user_erasure, "conversation_write_scope", scope)
    redis = SimpleNamespace(set=AsyncMock(), get=AsyncMock())
    monkeypatch.setattr(deletion, "get_redis", AsyncMock(return_value=redis))
    await deletion.save_pending_action(
        "c", action="set_reminder", summary="吃药", recurrence=recurrence
    )
    redis.get.return_value = redis.set.await_args.args[1]
    result = await deletion.load_pending_action("c")
    if isinstance(recurrence, str) and recurrence in {"daily", "once"}:
        assert result["recurrence"] == recurrence
    else:
        assert "recurrence" not in result
    redis.get.return_value = json.dumps({"action": "set_reminder", "recurrence": {}})
    assert "recurrence" not in await deletion.load_pending_action("c")


async def test_equal_clock_partial_failure_not_lost(io, monkeypatch):
    target = datetime(2026, 11, 1, 8, tzinfo=BJ)
    monkeypatch.setattr(
        reminders,
        "resolve_reminder_targets",
        lambda *a, **kw: reminders.ReminderTargets(True, (target, target), "resolved"),
    )
    io.trigger.side_effect = ["trigger", None]
    assert (
        await ih._direct_create_reminder(
            user_message="8点吃药，8点喝水提醒我", ctx=io.ctx
        )
    )[0] == "failed"
    d = io.ctx.response_diagnostics["reminder_direct_time"]
    assert d["created_count"] == 1 and d["target_count"] == 2
    assert d["created_targets"] == d["failed_targets"] == [target.isoformat()]
