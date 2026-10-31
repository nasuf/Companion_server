"""Calendar-day truth must not depend on the offset of a server timestamp."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.services.schedule_domain import time_parser as parser
from app.services.temporal_query import resolve_temporal_query

UTC = timezone.utc
SHANGHAI = ZoneInfo("Asia/Shanghai")


# Explicit calendar oracles, kept independent of parser implementation.
@pytest.mark.parametrize(
    "anchor,text,expected",
    [
        ("2026-10-31T00:15:00+08:00", "昨天", "2026-10-30"),
        ("2026-10-31T00:15:00+08:00", "今天", "2026-10-31"),
        ("2026-10-31T00:15:00+08:00", "明天", "2026-11-01"),
        ("2026-11-01T00:15:00+08:00", "上个月", "2026-10-01"),
        ("2026-11-01T00:15:00+08:00", "这个月", "2026-11-01"),
        ("2026-11-01T00:15:00+08:00", "下个月", "2026-12-01"),
        ("2027-01-01T00:15:00+08:00", "去年", "2026-01-01"),
        ("2027-01-01T00:15:00+08:00", "前年", "2025-01-01"),
        ("2027-01-01T00:15:00+08:00", "今年", "2027-01-01"),
        ("2026-11-02T00:15:00+08:00", "上周", "2026-10-26"),
        ("2026-11-02T00:15:00+08:00", "本周一", "2026-11-02"),
        ("2026-11-02T00:15:00+08:00", "下周", "2026-11-09"),
        ("2024-03-01T00:15:00+08:00", "昨天", "2024-02-29"),
        ("2026-10-31T00:15:00+08:00", "3天后", "2026-11-03"),
        ("2026-10-31T00:15:00+08:00", "两周前", "2026-10-17"),
        ("2026-10-31T00:15:00+08:00", "上午", "2026-10-31"),
        ("2026-10-31T00:15:00+08:00", "8点", "2026-10-31"),
        ("2026-10-31T00:15:00+08:00", "2025年2月3日", "2025-02-03"),
    ],
)
@pytest.mark.parametrize("offset_hours", [-7, 0, 8, 14])
def test_calendar_uses_business_day_for_equivalent_instants(
    monkeypatch, anchor, text, expected, offset_hours
):
    monkeypatch.setattr(parser, "_TZ", SHANGHAI)
    local = datetime.fromisoformat(anchor)
    supplied = local.astimezone(timezone(timedelta(hours=offset_hours)))
    (parsed,) = parser.parse_time_expressions(text, now=supplied)
    assert parsed.start.astimezone(SHANGHAI).date().isoformat() == expected
    (canonical,) = parser.parse_time_expressions(text, now=local)
    assert (parsed.start, parsed.end_exclusive, parsed.is_future) == (
        canonical.start,
        canonical.end_exclusive,
        canonical.is_future,
    )


@pytest.mark.parametrize(
    "zone,anchor,day",
    [
        ("America/Los_Angeles", "2026-11-01T02:00:00+00:00", "2026-10-31"),
        ("Pacific/Auckland", "2026-10-31T12:15:00+00:00", "2026-11-01"),
        ("Asia/Kathmandu", "2026-10-31T18:20:00+00:00", "2026-11-01"),
    ],
)
def test_configured_business_zone_is_not_assumed_to_be_china(
    monkeypatch, zone, anchor, day
):
    calendar = ZoneInfo(zone)
    monkeypatch.setattr(parser, "_TZ", calendar)
    (parsed,) = parser.parse_time_expressions(
        "今天", now=datetime.fromisoformat(anchor)
    )
    assert parsed.start.date().isoformat() == day
    assert parsed.start.tzinfo == calendar


@pytest.mark.parametrize("text", ["昨天", "明天", "上个月", "去年", "本周一", "上午"])
def test_recording_and_read_query_share_calendar_boundary(monkeypatch, text):
    monkeypatch.setattr(parser, "_TZ", SHANGHAI)
    anchor = datetime(2027, 1, 3, 16, 15, tzinfo=UTC)
    record = parser.parse_with_statement_time(text, now=anchor)
    query = resolve_temporal_query(
        text,
        anchor=anchor,
        anchor_source="persisted_message",
        timezone_name="Asia/Shanghai",
    )
    assert record.statement_time is anchor
    assert query.status == "resolved"
    (actual,) = record.event_times
    (expected,) = query.parsed_times()
    assert (actual.start, actual.end_exclusive) == (
        expected.start,
        expected.end_exclusive,
    )


@pytest.mark.parametrize(
    "text,seconds", [("15分钟前", -900), ("2小时后", 7200), ("半小时前", -1800)]
)
def test_elapsed_offsets_keep_the_original_instant(text, seconds):
    anchor = datetime(2026, 10, 30, 16, 15, tzinfo=UTC)
    (parsed,) = parser.parse_time_expressions(text, now=anchor)
    assert parsed.start == anchor + timedelta(seconds=seconds)
    assert parsed.is_future == (seconds > 0)


@pytest.mark.parametrize(
    "received_text,message,expected_day",
    [
        ("2026-10-30T16:15:00+00:00", "明天提醒我复习", "2026-11-01"),
        ("2026-10-31T15:59:50+00:00", "两天后提醒我复习", "2026-11-02"),
        ("2026-12-31T16:15:00+00:00", "明天提醒我复习", "2027-01-02"),
    ],
)
async def test_real_reminder_handler_keeps_received_day_after_processing_delay(
    monkeypatch, received_text, message, expected_day
):
    from app.services.chat import intent_handlers as handlers
    from app.services.reminder import scheduling
    from app.services.workspace import workspaces
    from app.services.schedule_domain import time_service

    received = datetime.fromisoformat(received_text)
    monkeypatch.setattr(parser, "_TZ", SHANGHAI)
    monkeypatch.setattr(
        time_service, "_now_corrected", lambda: received + timedelta(seconds=30)
    )
    monkeypatch.setattr(
        workspaces,
        "get_active_workspace",
        AsyncMock(return_value=SimpleNamespace(id="w", agentId="a")),
    )
    persist = AsyncMock(return_value="memory")
    trigger = AsyncMock()
    monkeypatch.setattr(handlers, "_persist_one_reminder", persist)
    monkeypatch.setattr(scheduling, "upsert_reminder_trigger", trigger)
    monkeypatch.setattr(scheduling, "notify_reminder_changed", AsyncMock())
    ctx = SimpleNamespace(
        user_id="u",
        agent_id="a",
        conversation_id="c",
        reply_context={"received_at": received.isoformat()},
        turn_clock=None, response_diagnostics={},
    )
    status, _ = await handlers._direct_create_reminder(user_message=message, ctx=ctx)
    expected = datetime.fromisoformat(expected_day).replace(tzinfo=SHANGHAI)
    assert status == "scheduled"
    persist.assert_awaited_once()
    assert persist.await_args.kwargs["occur_time"] == expected
    assert persist.await_args.kwargs["statement_time"] == received
    trigger.assert_awaited_once()
    assert trigger.await_args.kwargs["trigger_time"] == expected


def test_naive_elapsed_timestamp_retains_legacy_representation():
    anchor = datetime(2026, 10, 31, 0, 15)
    (parsed,) = parser.parse_time_expressions("15分钟前", now=anchor)
    assert parsed.start == anchor - timedelta(minutes=15)
    assert parsed.start.tzinfo is None


@pytest.mark.parametrize(
    "instant,hours",
    [("2026-03-08T05:15:00+00:00", 23), ("2026-11-01T04:15:00+00:00", 25)],
)
def test_local_calendar_day_keeps_dst_boundary_precision(monkeypatch, instant, hours):
    calendar = ZoneInfo("America/New_York")
    monkeypatch.setattr(parser, "_TZ", calendar)
    (parsed,) = parser.parse_time_expressions(
        "今天", now=datetime.fromisoformat(instant)
    )
    assert parsed.start.hour == 0
    assert parsed.end_exclusive.astimezone(calendar).hour == 0
    assert (
        parsed.end_exclusive.astimezone(UTC) - parsed.start.astimezone(UTC)
    ).total_seconds() == hours * 3600
