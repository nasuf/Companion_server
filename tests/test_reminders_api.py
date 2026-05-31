"""/reminders endpoint 测试 — inspector "提醒" tab 后端契约.

覆盖:
- 401/403 ownership: no token / wrong user
- 200 happy path: 返 items + total + dlq_count
- status filter: active / fired / cancelled / all → DB where 子句正确
- 分页: limit/offset 透传
- DLQ: Redis ZSET 读取 + 按 user 过滤 + 排序倒序
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_auth_header as _hdr


@pytest.fixture
def client(api_client):
    return api_client


def _trigger(
    *, tid="t-1", active=True, last_fired=None,
    summary="喝水", recurrence="once", retry_count=0, habit_weekdays=None,
    completed_at=None, deleted_at=None, completed_dates=None, sent_to_ai=False,
):
    data = {"summary": summary, "memory_id": "mem-1", "recurrence": recurrence}
    if retry_count:
        data["retry_count"] = retry_count
    if habit_weekdays is not None:
        data["habit_weekdays"] = habit_weekdays
    if completed_at is not None:
        data["completed_at"] = completed_at
    if completed_dates is not None:
        data["completed_dates"] = completed_dates
    if deleted_at is not None:
        data["deleted_at"] = deleted_at
    if sent_to_ai:
        data["sent_to_ai"] = True
    return SimpleNamespace(
        id=tid,
        aiAgentId="agent-A",
        userId="u1",
        actionType="reminder",
        actionData=data,
        triggerTime=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
        lastFired=last_fired,
        isActive=active,
        createdAt=datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc),
    )


# ── ownership ─────────────────────────────────────────────────────────


def test_list_no_token_401(client):
    r = client.get("/reminders?user_id=u1")
    assert r.status_code == 401


def test_list_wrong_user_403(client):
    r = client.get("/reminders?user_id=u2", headers=_hdr("u1"))
    assert r.status_code == 403


# ── happy path ─────────────────────────────────────────────────────────


def test_list_happy_returns_items_total_dlq(client):
    triggers = [_trigger(tid="t-1"), _trigger(tid="t-2", active=False, last_fired=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc))]
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=2)
        mock_db.timetrigger.find_many = AsyncMock(return_value=triggers)
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["dlq_count"] == 0
    assert len(body["items"]) == 2
    assert body["items"][0]["id"] == "t-1"
    assert body["items"][0]["status"] == "active"
    assert body["items"][1]["status"] == "fired"


def test_list_dlq_count_from_redis(client):
    fake_redis = MagicMock(zcard=AsyncMock(return_value=7))
    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=0)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[])
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    assert r.status_code == 200
    assert r.json()["dlq_count"] == 7


def test_list_dlq_redis_failure_does_not_500(client):
    """Redis 挂 → dlq_count=0 但端点正常返 (DLQ 是观察性数据, 不能挂主流程)."""
    fake_redis = MagicMock(zcard=AsyncMock(side_effect=ConnectionError("redis down")))
    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=0)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[])
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    assert r.status_code == 200
    assert r.json()["dlq_count"] == 0


# ── status filter → DB where 子句 ─────────────────────────────────────


def _capture_where(client, status_param: str) -> dict:
    captured = {}

    async def _capture_count(where):
        captured["count_where"] = where
        return 0

    async def _capture_find(where, order, take, skip):
        captured["find_where"] = where
        return []

    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))
    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(side_effect=_capture_count)
        mock_db.timetrigger.find_many = AsyncMock(side_effect=_capture_find)
        r = client.get(
            f"/reminders?user_id=u1&status={status_param}",
            headers=_hdr("u1"),
        )
    assert r.status_code == 200
    return captured.get("find_where", {})


def test_status_active_filters_isActive_true(client):
    where = _capture_where(client, "active")
    assert where["isActive"] is True


def test_status_fired_filters_isActive_false_and_lastFired_not_null(client):
    where = _capture_where(client, "fired")
    assert where["isActive"] is False
    assert where["lastFired"] == {"not": None}


def test_status_cancelled_filters_isActive_false_and_lastFired_null(client):
    where = _capture_where(client, "cancelled")
    assert where["isActive"] is False
    assert where["lastFired"] is None


def test_status_all_no_extra_filter(client):
    where = _capture_where(client, "all")
    # 'all' 不该加 isActive/lastFired filter
    assert "isActive" not in where
    assert "lastFired" not in where
    # 但仍按 user_id + actionType=reminder 过滤
    assert where["userId"] == "u1"
    assert where["actionType"] == "reminder"


def test_status_open_keeps_fired_once_until_user_closes(client):
    rows = [
        _trigger(tid="t-active", active=True, recurrence="once"),
        _trigger(
            tid="t-fired-once",
            active=False,
            last_fired=datetime(2026, 5, 3, 10, 5, tzinfo=timezone.utc),
            recurrence="once",
        ),
        _trigger(
            tid="t-completed",
            active=False,
            last_fired=datetime(2026, 5, 3, 10, 6, tzinfo=timezone.utc),
            recurrence="once",
            completed_at="2026-05-03T10:06:00+00:00",
        ),
        _trigger(
            tid="t-deleted",
            active=False,
            recurrence="once",
            deleted_at="2026-05-03T10:06:00+00:00",
        ),
        _trigger(
            tid="t-old-weekly",
            active=False,
            last_fired=datetime(2026, 5, 3, 10, 7, tzinfo=timezone.utc),
            recurrence="weekly",
        ),
    ]
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=len(rows))
        mock_db.timetrigger.find_many = AsyncMock(return_value=rows)
        r = client.get("/reminders?user_id=u1&status=open", headers=_hdr("u1"))

    assert r.status_code == 200
    ids = [item["id"] for item in r.json()["items"]]
    assert ids == ["t-active", "t-fired-once", "t-completed"]


def test_status_open_paginates_after_python_filtering(client):
    rows = [
        _trigger(tid="deleted-newer", deleted_at="2026-05-03T10:00:00+00:00"),
        _trigger(tid="weekly-fired", active=False, last_fired=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc), recurrence="weekly"),
        _trigger(tid="open-1"),
        _trigger(tid="open-2"),
    ]
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(return_value=rows)
        r = client.get(
            "/reminders?user_id=u1&status=open&limit=1&offset=1",
            headers=_hdr("u1"),
        )

    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert [item["id"] for item in body["items"]] == ["open-2"]


# ── pagination ─────────────────────────────────────────────────────────


def test_pagination_limit_offset_passed_through(client):
    captured = {}

    async def _capture_find(where, order, take, skip):
        captured["take"] = take
        captured["skip"] = skip
        return []

    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))
    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=0)
        mock_db.timetrigger.find_many = AsyncMock(side_effect=_capture_find)
        r = client.get(
            "/reminders?user_id=u1&limit=20&offset=40",
            headers=_hdr("u1"),
        )
    assert r.status_code == 200
    assert captured["take"] == 20
    assert captured["skip"] == 40


def test_pagination_limit_capped_at_200(client):
    """Pydantic Query(le=200) 防止前端误传超大 limit 拖崩 DB."""
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))
    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=0)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[])
        r = client.get("/reminders?user_id=u1&limit=10000", headers=_hdr("u1"))
    assert r.status_code == 422


# ── DLQ endpoint ───────────────────────────────────────────────────────


def test_dlq_returns_user_filtered_entries(client):
    """ZSET 跨 user 共享, /reminders/dlq 必须按 user_id 过滤 (用 trigger.userId
    反查 — DLQ 不存 user_id 在 entry 里, 用 trigger_id 关联)."""
    entries = [
        json.dumps({
            "trigger_id": "t-mine", "memory_id": "m1", "summary": "X",
            "recurrence": "once", "error": "boom", "kind": "exhausted",
            "attempt": 3, "failed_at": "2026-05-03T10:00:00",
        }),
        json.dumps({
            "trigger_id": "t-other", "memory_id": "m2", "summary": "Y",
            "recurrence": "once", "error": "boom", "kind": "exhausted",
            "attempt": 3, "failed_at": "2026-05-03T11:00:00",
        }),
    ]
    fake_redis = MagicMock(zrevrange=AsyncMock(return_value=entries))

    # 仅 t-mine 属于 u1, t-other 属于其他 user
    user_triggers = [SimpleNamespace(id="t-mine")]

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_many = AsyncMock(return_value=user_triggers)
        r = client.get("/reminders/dlq?user_id=u1", headers=_hdr("u1"))

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["trigger_id"] == "t-mine"


def test_dlq_redis_failure_returns_empty(client):
    """Redis 挂 → 返空列表, 不 500."""
    fake_redis = MagicMock(zrevrange=AsyncMock(side_effect=ConnectionError("redis down")))
    with patch(
        "app.services.reminder.checkin.get_redis",
        new_callable=AsyncMock, return_value=fake_redis,
    ):
        r = client.get("/reminders/dlq?user_id=u1", headers=_hdr("u1"))
    assert r.status_code == 200
    assert r.json() == []


def test_dlq_no_token_401(client):
    r = client.get("/reminders/dlq?user_id=u1")
    assert r.status_code == 401


# ── ReminderItem shape ─────────────────────────────────────────────────


def test_item_shape_includes_retry_count(client):
    """retry_count 从 actionData JSON 读, 默认 0; 重试中的 trigger 应该露出."""
    trigger = _trigger(tid="t-retry", retry_count=2)
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=1)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[trigger])
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    item = r.json()["items"][0]
    assert item["retry_count"] == 2
    assert item["recurrence"] == "once"
    assert item["summary"] == "喝水"
    assert item["habit_weekdays"] == []
    assert item["completed_dates"] == []
    assert item["sent_to_ai"] is False


def test_item_shape_includes_sent_to_ai(client):
    trigger = _trigger(tid="t-ai", sent_to_ai=True)
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=1)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[trigger])
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    assert r.json()["items"][0]["sent_to_ai"] is True


def test_item_shape_includes_habit_weekdays(client):
    trigger = _trigger(tid="t-habit", recurrence="weekly", habit_weekdays=[5, 1, 1])
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=1)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[trigger])
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    item = r.json()["items"][0]
    assert item["recurrence"] == "weekly"
    assert item["habit_weekdays"] == [1, 5]


def test_item_classify_status_cancelled_when_inactive_no_lastfired(client):
    """isActive=False + lastFired=None → cancelled (用户主动取消, 没真响过)."""
    trigger = _trigger(tid="t-cancel", active=False, last_fired=None)
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=1)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[trigger])
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    assert r.json()["items"][0]["status"] == "cancelled"


# ── check-in write paths ──────────────────────────────────────────────


def test_update_reminder_pinned_round_trips(client):
    trigger = _trigger(tid="t-pin")
    updated = _trigger(tid="t-pin")
    updated.actionData["pinned"] = True

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock) as notify,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock(return_value=updated)
        r = client.patch(
            "/reminders/t-pin",
            json={"pinned": True, "conversation_id": "conv-1"},
            headers=_hdr("u1"),
        )

    assert r.status_code == 200
    assert r.json()["pinned"] is True
    update_data = mock_db.timetrigger.update.await_args.kwargs["data"]
    assert update_data["actionData"].data["pinned"] is True
    notify.assert_awaited_once_with("conv-1", kind="rescheduled", trigger_id="t-pin")


def test_update_reminder_habit_weekdays_round_trips(client):
    trigger = _trigger(tid="t-weekly", recurrence="weekly")
    updated = _trigger(tid="t-weekly", recurrence="weekly", habit_weekdays=[1, 3, 5])

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock(return_value=updated)
        r = client.patch(
            "/reminders/t-weekly",
            json={"habit_weekdays": [5, 1, 3, 3]},
            headers=_hdr("u1"),
        )

    assert r.status_code == 200
    assert r.json()["habit_weekdays"] == [1, 3, 5]
    update_data = mock_db.timetrigger.update.await_args.kwargs["data"]
    assert update_data["actionData"].data["habit_weekdays"] == [1, 3, 5]


def test_update_reminder_sent_to_ai_round_trips(client):
    trigger = _trigger(tid="t-ai")
    updated = _trigger(tid="t-ai", sent_to_ai=True)

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock(return_value=updated)
        r = client.patch(
            "/reminders/t-ai",
            json={"sent_to_ai": True, "conversation_id": "conv-1"},
            headers=_hdr("u1"),
        )

    assert r.status_code == 200
    assert r.json()["sent_to_ai"] is True
    update_data = mock_db.timetrigger.update.await_args.kwargs["data"]
    assert update_data["actionData"].data["sent_to_ai"] is True
    assert update_data["actionData"].data["conversation_id"] == "conv-1"


def test_update_deleted_reminder_rejected(client):
    trigger = _trigger(tid="t-deleted", deleted_at="2026-05-03T10:00:00+00:00")

    with patch("app.services.reminder.checkin.db") as mock_db:
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock()
        r = client.patch(
            "/reminders/t-deleted",
            json={"summary": "新计划"},
            headers=_hdr("u1"),
        )

    assert r.status_code == 409
    mock_db.timetrigger.update.assert_not_awaited()


def test_update_completed_once_allows_pin_but_rejects_content_edit(client):
    trigger = _trigger(
        tid="t-completed-edit",
        active=False,
        last_fired=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
        completed_at="2026-05-03T10:00:00+00:00",
    )
    updated = _trigger(
        tid="t-completed-edit",
        active=False,
        last_fired=datetime(2026, 5, 3, 10, 0, tzinfo=timezone.utc),
        completed_at="2026-05-03T10:00:00+00:00",
    )
    updated.actionData["pinned"] = True

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock(return_value=updated)
        pin = client.patch(
            "/reminders/t-completed-edit",
            json={"pinned": True},
            headers=_hdr("u1"),
        )
        edit = client.patch(
            "/reminders/t-completed-edit",
            json={"summary": "新计划"},
            headers=_hdr("u1"),
        )

    assert pin.status_code == 200
    assert pin.json()["pinned"] is True
    assert edit.status_code == 409
    assert mock_db.timetrigger.update.await_count == 1


def test_update_weekly_reminder_rejects_past_trigger_time(client):
    trigger = _trigger(tid="t-weekly-past", recurrence="weekly")

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock()
        r = client.patch(
            "/reminders/t-weekly-past",
            json={"trigger_time": "2000-01-01T00:00:00+00:00"},
            headers=_hdr("u1"),
        )

    assert r.status_code == 400
    assert r.json()["detail"] == "提醒时间必须在未来"
    mock_db.timetrigger.update.assert_not_awaited()


def test_complete_reminder_archives_memory_and_marks_inactive(client):
    trigger = _trigger(tid="t-done")
    updated = _trigger(
        tid="t-done",
        active=False,
        last_fired=datetime(2026, 5, 3, 10, 5, tzinfo=timezone.utc),
    )

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.archive_reminder_memory", new_callable=AsyncMock) as archive,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock) as notify,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock(return_value=updated)
        r = client.post(
            "/reminders/t-done/complete?conversation_id=conv-1",
            headers=_hdr("u1"),
        )

    assert r.status_code == 200
    assert r.json()["status"] == "fired"
    update_data = mock_db.timetrigger.update.await_args.kwargs["data"]
    assert update_data["isActive"] is False
    archive.assert_awaited_once_with(
        memory_id="mem-1",
        side="user",
        reason="completed_from_checkin",
    )
    notify.assert_awaited_once_with("conv-1", kind="archived", trigger_id="t-done")


def test_complete_once_is_idempotent_when_already_completed(client):
    trigger = _trigger(
        tid="t-already-done",
        active=False,
        last_fired=datetime(2026, 5, 3, 10, 5, tzinfo=timezone.utc),
        completed_at="2026-05-03T10:05:00+00:00",
    )

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.archive_reminder_memory", new_callable=AsyncMock) as archive,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock) as notify,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock()
        r = client.post(
            "/reminders/t-already-done/complete?conversation_id=conv-1",
            headers=_hdr("u1"),
        )

    assert r.status_code == 200
    assert r.json()["completed_at"] == "2026-05-03T10:05:00+00:00"
    mock_db.timetrigger.update.assert_not_awaited()
    archive.assert_not_awaited()
    notify.assert_not_awaited()


def test_complete_weekly_marks_occurrence_without_archiving_habit(client):
    trigger = _trigger(tid="t-habit-done", recurrence="weekly", habit_weekdays=[1, 3])
    updated = _trigger(
        tid="t-habit-done",
        recurrence="weekly",
        habit_weekdays=[1, 3],
        completed_dates=["2026-05-04"],
    )

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.archive_reminder_memory", new_callable=AsyncMock) as archive,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock) as notify,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock(return_value=updated)
        r = client.post(
            "/reminders/t-habit-done/complete?conversation_id=conv-1&occurrence_date=2026-05-04",
            headers=_hdr("u1"),
        )

    assert r.status_code == 200
    assert r.json()["status"] == "active"
    assert r.json()["completed_dates"] == ["2026-05-04"]
    update_data = mock_db.timetrigger.update.await_args.kwargs["data"]
    assert "isActive" not in update_data
    assert update_data["actionData"].data["completed_dates"] == ["2026-05-04"]
    archive.assert_not_awaited()
    notify.assert_awaited_once_with("conv-1", kind="archived", trigger_id="t-habit-done")


def test_delete_reminder_archives_memory_and_returns_204(client):
    trigger = _trigger(tid="t-delete")

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.archive_reminder_memory", new_callable=AsyncMock) as archive,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock) as notify,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock(return_value=trigger)
        r = client.delete(
            "/reminders/t-delete?conversation_id=conv-1",
            headers=_hdr("u1"),
        )

    assert r.status_code == 204
    update_data = mock_db.timetrigger.update.await_args.kwargs["data"]
    assert update_data["isActive"] is False
    assert update_data["actionData"].data["deleted_at"]
    archive.assert_awaited_once_with(
        memory_id="mem-1",
        side="user",
        reason="deleted_from_checkin",
    )
    notify.assert_awaited_once_with("conv-1", kind="cancelled", trigger_id="t-delete")


def test_delete_reminder_is_idempotent_when_already_deleted(client):
    trigger = _trigger(tid="t-deleted", deleted_at="2026-05-03T10:00:00+00:00")

    with (
        patch("app.services.reminder.checkin.db") as mock_db,
        patch("app.services.reminder.checkin.archive_reminder_memory", new_callable=AsyncMock) as archive,
        patch("app.services.reminder.checkin.notify_reminder_changed", new_callable=AsyncMock) as notify,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.find_unique = AsyncMock(return_value=trigger)
        mock_db.timetrigger.update = AsyncMock()
        r = client.delete(
            "/reminders/t-deleted?conversation_id=conv-1",
            headers=_hdr("u1"),
        )

    assert r.status_code == 204
    mock_db.timetrigger.update.assert_not_awaited()
    archive.assert_not_awaited()
    notify.assert_not_awaited()
