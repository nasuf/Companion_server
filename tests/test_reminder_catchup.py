"""TIME10: 重启/部署期间到点的提醒 —— 30 分钟内补发, 更早的标成错过、周期性的续到下一期。"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.proactive import triggers as triggers_mod
from app.services.reminder import mutations
from app.services.time_domain.clock import _TZ

NOW = datetime(2026, 11, 2, 8, 30, tzinfo=_TZ)


def _trig(tid: str, minutes_ago: float, *, recurrence: str = "once", last_fired=None, created_after=False):
    due = NOW - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        id=tid, userId="u1", aiAgentId="a1", actionType="reminder",
        triggerTime=due, lastFired=last_fired, isActive=True,
        createdAt=due + timedelta(minutes=20) if created_after else due - timedelta(days=1),
        actionData={"summary": "吃药", "memory_id": "m1", "recurrence": recurrence},
    )


def _fake_db(window, late, stale):
    db = MagicMock()

    async def find_many(where, take=None, order=None):
        if where.get("lastFired", "absent") is None and "gte" in where["triggerTime"]:
            return list(late)
        if where.get("lastFired", "absent") is None:
            return list(stale)
        return list(window)
    db.timetrigger.find_many = AsyncMock(side_effect=find_many)
    db.timetrigger.update_many = AsyncMock(return_value=1)
    db.aiagent.find_many = AsyncMock(return_value=[])
    return db


@pytest.fixture(autouse=True)
def valid(monkeypatch):
    check = AsyncMock(return_value=True)
    monkeypatch.setattr(mutations, "reminder_still_valid", check)
    return check


@pytest.mark.asyncio
async def test_recently_missed_reminder_is_caught_up():
    late = _trig("late-1", 10)
    db = _fake_db(window=[], late=[late], stale=[])
    executed = AsyncMock()
    with patch.object(triggers_mod, "db", db), \
            patch.object(triggers_mod, "_execute_trigger", executed), \
            patch.object(triggers_mod, "resolve_workspace_id", AsyncMock(return_value="w1")), \
            patch.object(triggers_mod, "datetime", wraps=datetime) as dt:
        dt.now.return_value = NOW
        await triggers_mod.scan_triggers()
    assert [c.args[0].id for c in executed.await_args_list] == ["late-1"]


@pytest.mark.asyncio
async def test_window_and_catchup_do_not_double_execute():
    trig = _trig("t-1", 0.5)
    db = _fake_db(window=[trig], late=[trig], stale=[])
    executed = AsyncMock()
    with patch.object(triggers_mod, "db", db), \
            patch.object(triggers_mod, "_execute_trigger", executed), \
            patch.object(triggers_mod, "resolve_workspace_id", AsyncMock(return_value="w1")), \
            patch.object(triggers_mod, "datetime", wraps=datetime) as dt:
        dt.now.return_value = NOW
        await triggers_mod.scan_triggers()
    assert executed.await_count == 1


@pytest.mark.asyncio
async def test_long_missed_once_reminder_is_marked_not_sent():
    db = _fake_db(window=[], late=[], stale=[_trig("old-1", 120)])
    with patch.object(triggers_mod, "db", db):
        assert await triggers_mod._expire_missed_reminders(NOW) == 1
    call = db.timetrigger.update_many.await_args
    assert call.kwargs["where"] == {"id": "old-1", "isActive": True, "lastFired": None}
    data = call.kwargs["data"]
    assert data["isActive"] is False
    payload = data["actionData"].data if hasattr(data["actionData"], "data") else data["actionData"]
    assert payload["missed_by_seconds"] == 7200 and payload["summary"] == "吃药"


@pytest.mark.asyncio
async def test_long_missed_daily_reminder_renews_to_next_future_occurrence():
    db = _fake_db(window=[], late=[], stale=[_trig("old-d", 60 * 24 * 3 + 60, recurrence="daily")])
    renew = AsyncMock(return_value="new")
    with patch.object(triggers_mod, "db", db), \
            patch("app.services.reminder.scheduling.renew_periodic_trigger", renew):
        await triggers_mod._expire_missed_reminders(NOW)
    nxt = renew.await_args.kwargs["next_trigger_time"]
    assert NOW < nxt <= NOW + timedelta(days=1)
    assert db.timetrigger.update_many.await_count == 1


@pytest.mark.asyncio
async def test_invalid_or_just_created_reminders_are_not_caught_up(valid):
    valid.side_effect = lambda db, t: t.id != "gone"
    db = _fake_db(window=[], late=[_trig("gone", 5), _trig("fresh", 5, recurrence="daily", created_after=True)],
                  stale=[])
    executed = AsyncMock()
    with patch.object(triggers_mod, "db", db), \
            patch.object(triggers_mod, "_execute_trigger", executed), \
            patch.object(triggers_mod, "datetime", wraps=datetime) as dt:
        dt.now.return_value = NOW
        await triggers_mod.scan_triggers()
    executed.assert_not_awaited()


@pytest.mark.asyncio
async def test_expiry_deactivates_before_renewing_and_never_revives_invalid(valid):
    db = _fake_db(window=[], late=[], stale=[_trig("old-d", 120, recurrence="daily")])
    db.timetrigger.update_many = AsyncMock(return_value=0)  # 期间已被处理
    renew = AsyncMock()
    with patch.object(triggers_mod, "db", db), \
            patch("app.services.reminder.scheduling.renew_periodic_trigger", renew):
        await triggers_mod._expire_missed_reminders(NOW)
    renew.assert_not_awaited()
    valid.return_value = False
    db.timetrigger.update_many = AsyncMock(return_value=1)
    with patch.object(triggers_mod, "db", db), \
            patch("app.services.reminder.scheduling.renew_periodic_trigger", renew):
        await triggers_mod._expire_missed_reminders(NOW)
    renew.assert_not_awaited()
    marker = db.timetrigger.update_many.await_args.kwargs["data"]["actionData"]
    payload = marker.data if hasattr(marker, "data") else marker
    assert payload["missed_reason"] == "no_longer_valid"
