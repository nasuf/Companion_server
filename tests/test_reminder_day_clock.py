"""A stated day + clock is one reminder at that moment.

The legacy parser returns 明天 and 下午3点 as separate results. Both reminder
entry points used to schedule each future result, so 明天下午3点 fired today
at 15:00 and tomorrow at 00:00, and 今晚8点 read as 08:00 and was refused.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from app.services.reminder.scheduling import precise_reminder_time

pytestmark = pytest.mark.usefixtures("unit_media_scope")

BJ = ZoneInfo("Asia/Shanghai")
AFTERNOON = datetime(2026, 10, 31, 6, 0, tzinfo=timezone.utc)  # BJ 10-31 14:00
PAST_MIDNIGHT = datetime(2026, 10, 31, 17, 30, tzinfo=timezone.utc)  # BJ 11-01 01:30
NEW_YEARS_EVE = datetime(2026, 12, 31, 8, 0, tzinfo=timezone.utc)  # BJ 12-31 16:00


def bj(*args):
    return datetime(*args, tzinfo=BJ)


@pytest.mark.parametrize("now,message,expected", [
    (AFTERNOON, "明天下午3点提醒我开会", bj(2026, 11, 1, 15)),
    (AFTERNOON, "今晚8点提醒我吃药", bj(2026, 10, 31, 20)),
    (AFTERNOON, "明晚九点提醒我", bj(2026, 11, 1, 21)),
    (AFTERNOON, "下周一上午10点提醒我交报告", bj(2026, 11, 2, 10)),
    (AFTERNOON, "晚上八点半提醒我吃药", bj(2026, 10, 31, 20, 30)),
    (AFTERNOON, "提醒我明天下午3点30分开会", bj(2026, 11, 1, 15, 30)),
    (AFTERNOON, "11月5日下午2点提醒我", bj(2026, 11, 5, 14)),
    (AFTERNOON, "1月3日早上8点提醒我", bj(2027, 1, 3, 8)),
    # Past Beijing midnight the anchor day is 11-01, not the UTC date.
    (PAST_MIDNIGHT, "明天下午3点提醒我开会", bj(2026, 11, 2, 15)),
    (PAST_MIDNIGHT, "今晚8点提醒我", bj(2026, 11, 1, 20)),
    (NEW_YEARS_EVE, "明天下午3点提醒我", bj(2027, 1, 1, 15)),
    (NEW_YEARS_EVE, "11月5日下午2点提醒我", bj(2027, 11, 5, 14)),
])
def test_day_and_clock_bind_to_one_future_time(now, message, expected):
    assert precise_reminder_time(message, now=now) == (True, expected)


@pytest.mark.parametrize("now,message", [
    (AFTERNOON, "今天早上8点提醒我"),
    (NEW_YEARS_EVE, "下午三点提醒我"),
])
def test_stated_time_already_passed_is_not_moved(now, message):
    assert precise_reminder_time(message, now=now) == (True, None)


@pytest.mark.parametrize("message", [
    "10分钟后提醒我",       # elapsed offset: legacy parser is exact
    "明天提醒我",           # no clock: day precision is not a trigger time
    "8点吃药，9点开会提醒我",  # several clocks stay several reminders
    "待会提醒我喝水",        # vague: ask back
])
def test_other_phrasings_stay_with_the_legacy_parser(message):
    assert precise_reminder_time(message, now=AFTERNOON) == (False, None)


def _record_ctx(received_at):
    from app.services.chat.intent_handlers import ShortCircuitCtx

    return ShortCircuitCtx(
        conversation_id="c1", agent_id="agent-A", user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context={"received_at": received_at.isoformat()},
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(), pending_sub_fragments={},
        sub_intent_mode=False, reply_index_offset=0, cached_patience=100,
    )


async def _direct(message, received_at, *, failing=()):
    from app.services.chat import intent_handlers as ih

    upserts = []

    async def capture(**kwargs):
        upserts.append(kwargs["trigger_time"])
        return None if kwargs["trigger_time"] in failing else f"t-{len(upserts)}"

    with (
        patch("app.services.workspace.workspaces.get_active_workspace", new_callable=AsyncMock,
              return_value=SimpleNamespace(agentId="agent-A", id="ws-1")),
        patch("app.services.memory.storage.persistence.store_memory",
              new_callable=AsyncMock, return_value="mem-1"),
        patch("app.services.reminder.scheduling.upsert_reminder_trigger", side_effect=capture),
        patch("app.services.reminder.scheduling.notify_reminder_changed", new_callable=AsyncMock),
    ):
        status, _ = await ih._direct_create_reminder(
            user_message=message, ctx=_record_ctx(received_at))
    return status, upserts


@pytest.mark.parametrize("received_at,message,expected", [
    (AFTERNOON, "明天下午3点提醒我开会", [bj(2026, 11, 1, 15)]),
    (AFTERNOON, "今晚8点提醒我吃药", [bj(2026, 10, 31, 20)]),
    (PAST_MIDNIGHT, "明天下午3点提醒我开会", [bj(2026, 11, 2, 15)]),
    (PAST_MIDNIGHT, "8点吃药，9点开会提醒我", [bj(2026, 11, 1, 8), bj(2026, 11, 1, 9)]),
])
async def test_direct_reminder_creates_exactly_the_stated_triggers(received_at, message, expected):
    status, upserts = await _direct(message, received_at)
    assert status == "scheduled" and sorted(upserts) == expected


async def test_direct_reminder_for_a_passed_time_asks_instead_of_scheduling():
    from app.services.memory.interaction import deletion

    with patch.object(deletion, "save_pending_action", new_callable=AsyncMock) as saved:
        status, upserts = await _direct("今天早上8点提醒我", AFTERNOON)
    assert status == "asked" and upserts == [] and saved.await_count == 1


async def _pending(message, now):
    from app.services.chat.preflight import PreflightCtx, resolve_pending_deletion
    from app.services.reminder import scheduling

    ctx = PreflightCtx(
        conversation_id="c1", agent_id="agent-A", user_id="u1", agent=SimpleNamespace(name="A"),
        tracer=MagicMock(safe_trace_id=None, close=MagicMock()),
        short_circuit_fn=AsyncMock(return_value=[{"event": "reply"}]),
    )
    created = []

    async def capture(**kwargs):
        created.append(kwargs["occur_time"])
        return "mem-new"

    pending = {"action": "set_reminder", "summary": "提醒我开会", "candidates": [], "new_time": None}
    with (
        patch("app.services.chat.preflight.load_pending_action", new_callable=AsyncMock,
              return_value=pending),
        patch("app.services.chat.preflight.clear_pending_deletion", new_callable=AsyncMock),
        patch("app.services.chat.preflight._now_corrected", return_value=now),
        patch.object(scheduling, "create_user_reminder", side_effect=capture),
        patch("app.services.workspace.workspaces.get_active_workspace", new_callable=AsyncMock,
              return_value=SimpleNamespace(id="ws-1", agentId="agent-A")),
        patch("app.services.chat.preflight.record_confirm_reply", new_callable=AsyncMock,
              return_value="好"),
    ):
        async for _ in resolve_pending_deletion(message, ctx):
            pass
    return created


@pytest.mark.parametrize("now,message,expected", [
    (AFTERNOON, "明天下午3点吧", bj(2026, 11, 1, 15)),
    (AFTERNOON, "今晚8点", bj(2026, 10, 31, 20)),
    (PAST_MIDNIGHT, "明天下午3点吧", bj(2026, 11, 2, 15)),
])
async def test_pending_reminder_answer_uses_the_stated_day(now, message, expected):
    assert await _pending(message, now) == [expected]


@pytest.mark.parametrize("message,failing", [
    ("明天下午3点提醒我开会", (bj(2026, 11, 1, 15),)),
    # One of two triggers failing must not confirm the other alone.
    ("8点吃药，9点开会提醒我", (bj(2026, 11, 1, 9),)),
])
async def test_direct_reminder_without_a_trigger_is_not_scheduled(message, failing):
    now = PAST_MIDNIGHT if "8点吃药" in message else AFTERNOON
    status, upserts = await _direct(message, now, failing=failing)
    assert status == "failed" and set(failing) <= set(upserts)


async def test_failed_reminder_is_never_confirmed_to_the_user():
    from app.services.chat import intent_handlers as ih

    ctx = _record_ctx(AFTERNOON)
    ctx.finalize = MagicMock(return_value="events")
    with (
        patch.object(ih, "_direct_create_reminder", new_callable=AsyncMock,
                     return_value=("failed", None)),
        patch.object(ih, "record_confirm_reply", new_callable=AsyncMock) as confirm,
    ):
        handled, events = await ih.handle_record_request("明天下午3点提醒我开会", ctx)
    assert handled is True and events == "events"
    confirm.assert_not_awaited()
    reply = ctx.finalize.call_args.args[0]
    assert ctx.finalize.call_args.kwargs["kind"] == "record_request_failed"
    assert "记上" not in reply and "叫你" not in reply and "再跟我说一遍" in reply


@pytest.mark.parametrize("confirm,expected", [
    ("好嘞，明天下午3点叫你", "好嘞，明天下午3点叫你"),
    (None, "好嘞, 明天 15:00叫你, 记好啦~"),  # LLM failure falls back to the template
])
async def test_scheduled_reminder_is_confirmed(confirm, expected):
    from app.services.chat import intent_handlers as ih

    ctx = _record_ctx(AFTERNOON)
    ctx.finalize = MagicMock(return_value="events")
    with (
        patch.object(ih, "_direct_create_reminder", new_callable=AsyncMock,
                     return_value=("scheduled", "明天 15:00")),
        patch.object(ih, "record_confirm_reply", new_callable=AsyncMock, return_value=confirm),
    ):
        handled, events = await ih.handle_record_request("明天下午3点提醒我开会", ctx)
    assert handled is True and events == "events"
    assert ctx.finalize.call_args.args[0] == expected
    assert ctx.finalize.call_args.kwargs["kind"] == "record_request_scheduled"
