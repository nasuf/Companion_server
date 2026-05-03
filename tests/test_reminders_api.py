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
    summary="喝水", recurrence="once", retry_count=0,
):
    data = {"summary": summary, "memory_id": "mem-1", "recurrence": recurrence}
    if retry_count:
        data["retry_count"] = retry_count
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
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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


# ── pagination ─────────────────────────────────────────────────────────


def test_pagination_limit_offset_passed_through(client):
    captured = {}

    async def _capture_find(where, order, take, skip):
        captured["take"] = take
        captured["skip"] = skip
        return []

    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))
    with (
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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
        "app.api.public.reminders.get_redis",
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
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
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


def test_item_classify_status_cancelled_when_inactive_no_lastfired(client):
    """isActive=False + lastFired=None → cancelled (用户主动取消, 没真响过)."""
    trigger = _trigger(tid="t-cancel", active=False, last_fired=None)
    fake_redis = MagicMock(zcard=AsyncMock(return_value=0))

    with (
        patch("app.api.public.reminders.db") as mock_db,
        patch("app.api.public.reminders.get_redis",
              new_callable=AsyncMock, return_value=fake_redis),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.count = AsyncMock(return_value=1)
        mock_db.timetrigger.find_many = AsyncMock(return_value=[trigger])
        r = client.get("/reminders?user_id=u1", headers=_hdr("u1"))

    assert r.json()["items"][0]["status"] == "cancelled"
