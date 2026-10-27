"""Real SQL for each service check and the alert outbox; synthetic rows only.

The checks count the whole database, so assertions use the change caused by
this test's own rows, never an assumption that the shared database is empty.
"""
import asyncio
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from prisma import Json, Prisma

from app.services.ops import service_health as h


@pytest.fixture
async def world(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("isolated PostgreSQL required")
    assert urlsplit(url).hostname in {"127.0.0.1", "localhost"}
    assert urlsplit(url).path == "/companion_proactive_e2e"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    monkeypatch.setattr(h, "db", database)
    monkeypatch.setattr("app.services.notifications.service.db", database, raising=False)
    user = await database.user.create(data={"username": "health-test-" + uuid4().hex})
    admin = await database.user.create(data={"username": "health-admin-" + uuid4().hex, "role": "admin"})
    agent = await database.aiagent.create(data={"userId": user.id, "name": "Synthetic"})
    space = await database.chatworkspace.create(data={"userId": user.id, "agentId": agent.id})
    conv = await database.conversation.create(data={"userId": user.id, "agentId": agent.id, "workspaceId": space.id})
    monkeypatch.setattr(h.settings, "redis_memory_alert_user_ids", admin.id)
    await database.execute_raw("DELETE FROM ops_service_health WHERE id='primary'")
    try:
        yield database, conv.id, space.id, user.id, admin.id
    finally:
        await database.execute_raw("DELETE FROM notification_events WHERE user_id=$1", admin.id)
        await database.execute_raw("DELETE FROM ops_service_health WHERE id='primary'")
        await database.execute_raw("DELETE FROM memory_extraction_batches WHERE conversation_id=$1", conv.id)
        await database.message.delete_many(where={"conversationId": conv.id})
        await database.conversation.delete(where={"id": conv.id})
        await database.chatworkspace.delete(where={"id": space.id})
        await database.aiagent.delete(where={"id": agent.id})
        await database.user.delete_many(where={"id": {"in": [user.id, admin.id]}})
        await database.disconnect()


async def message(database, conv, role, minutes_ago, **metadata):
    at = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).replace(tzinfo=None)
    return await database.message.create(data={"conversationId": conv, "role": role, "content": "synthetic",
                                               "createdAt": at, "metadata": Json(metadata)})


async def test_unanswered_counts_ordinary_messages_and_keeps_alerted_ones_until_answered(world, monkeypatch):
    database, conv, *_ = world
    monkeypatch.setattr(h, "unanswered_after", lambda: h.UNANSWERED_AFTER)
    now = datetime.now(UTC)
    before = (await h._unanswered(now, {}))["count"]
    await message(database, conv, "user", 40)                       # outside the window
    await message(database, conv, "user", 1)                        # still within 3 minutes
    await message(database, conv, "user", 12, human_takeover=True)  # an administrator replies
    first = await message(database, conv, "user", 10)
    second = await message(database, conv, "user", 9)
    result = await h._unanswered(now, {})
    assert result["count"] - before == 2 and result["alerting"]
    carry = result["_carry"]
    assert {first.id, second.id} <= set(carry["unanswered_ids"])
    # Thirty minutes later the window is empty, but nobody answered: still alerting.
    later = now + timedelta(minutes=40)
    still = await h._unanswered(later, carry)
    assert still["carried"] >= 2 and still["alerting"]
    # Below the raising threshold, a remaining alerted message still holds the alert.
    one = await h._unanswered(later, {"unanswered_ids": [first.id]})
    assert one["carried"] == 1 and one["alerting"]
    # An answer is the recovery evidence.
    await message(database, conv, "assistant", 0)
    assert not (await h._unanswered(later, {"unanswered_ids": [first.id, second.id]}))["carried"]


async def test_deleted_conversations_drop_out(world, monkeypatch):
    database, conv, *_ = world
    monkeypatch.setattr(h, "unanswered_after", lambda: h.UNANSWERED_AFTER)
    pending = await message(database, conv, "user", 10)
    await database.conversation.update(where={"id": conv}, data={"isDeleted": True})
    result = await h._unanswered(datetime.now(UTC), {"unanswered_ids": [pending.id]})
    assert result["carried"] == 0 and pending.id not in result["_carry"]["unanswered_ids"]


async def test_reply_failures_need_a_later_success_to_recover(world):
    database, conv, *_ = world
    now = datetime.now(UTC)
    before = await h._reply_failed(now, {})
    await message(database, conv, "assistant", 5, reply_failed=True)
    await message(database, conv, "assistant", 20, reply_failed=True)
    after = await h._reply_failed(now, {})
    assert after["count"] - before["count"] == 1 and after["replies"] - before["replies"] == 1
    # Recovery evidence is global, so place the alerted failure after any row
    # another test may have left behind; then nothing succeeded after it.
    last = await message(database, conv, "assistant", -2, reply_failed=True)
    carry = {"last_failed_at": last.createdAt.replace(tzinfo=None).isoformat()}
    later = now + timedelta(minutes=30)
    assert (await h._reply_failed(later, carry))["awaiting_success"]
    await message(database, conv, "assistant", -3)
    recovered = await h._reply_failed(later, carry)
    assert not recovered["awaiting_success"] and not recovered["alerting"]


async def test_memory_is_blocked_only_while_the_user_keeps_talking(world):
    database, conv, space, user, _ = world
    now = datetime.now(UTC)
    before = (await h._memory_blocked(now, {}))["count"]
    for status, error, hours in (("processing", "extraction_failed", 2), ("processing", None, 2),
                                 ("processing", "extraction_failed", 0), ("completed", "extraction_failed", 2)):
        await database.execute_raw(
            """INSERT INTO memory_extraction_batches(id,user_id,workspace_id,conversation_id,side,scope,sources,
                status,last_error,prepared_at,updated_at) VALUES($1,$2,$3,$4,'user','{}'::jsonb,'[{"ref":"synthetic"}]'::jsonb,$5,$6,
                CURRENT_TIMESTAMP - make_interval(hours => $7::int), CURRENT_TIMESTAMP - make_interval(hours => $7::int))""",
            uuid4().hex, user, space, conv, status, error, hours)
    # The user left: a failed batch waits for the next chat, which is not an outage.
    assert (await h._memory_blocked(now, {}))["count"] - before == 0
    for minutes in (50, 40, 30):
        await message(database, conv, "user", minutes)
    assert (await h._memory_blocked(now, {}))["count"] - before == 1


async def age(database):
    await database.execute_raw("UPDATE ops_service_health SET sampled_at=NOW()-INTERVAL '5 minutes'")


def sample(*alerting, unknown=()):
    checks = {n: {"status": "measured", "count": 0, "threshold": 1, "window_minutes": 15, "after_minutes": 3,
                  "replies": 0, "jobs": [], "alerting": n in alerting} for n in h.CHECKS}
    for n in alerting:
        checks[n].update(count=3, jobs=["daily_schedule"])
    for n in unknown:
        checks[n] = {"status": "unknown", "error_type": "TimeoutError"}
    return {"sampled_at": datetime.now(UTC).isoformat(), "checks": checks, "carry": {}}


async def test_one_push_per_incident_against_the_real_outbox(world, monkeypatch):
    database, *_, admin = world

    async def titles():
        rows = await database.query_raw(
            "SELECT title FROM notification_events WHERE user_id=$1 ORDER BY created_at, id", admin)
        return [r["title"] for r in rows]

    async def step(value):
        await age(database)
        monkeypatch.setattr(h, "collect_metrics", AsyncMock(return_value=value))
        await h.sample_and_notify()

    monkeypatch.setattr(h, "collect_metrics", AsyncMock(return_value=sample("reply_failed")))
    await asyncio.gather(h.sample_and_notify(), h.sample_and_notify())
    assert await titles() == ["服务告警"]
    status = await h.read_status()
    assert status["level"] == "warning" and status["status"] == "active" and "carry" not in status["sample"]
    await step(sample("reply_failed"))                           # same incident: nothing
    await step(sample("reply_failed", "unanswered"))             # a new failing check: one push
    await step(sample("unanswered"))                             # shrinking: nothing
    await step(sample(unknown=tuple(h.CHECKS)))                  # unmeasured: latched, nothing
    assert (await h.read_status())["level"] == "warning"
    await step(sample("unanswered"))                             # back from unknown, known check: nothing
    assert await titles() == ["服务告警", "服务告警"]
    await step(sample())                                         # recovery: once
    await step(sample())
    assert await titles() == ["服务告警", "服务告警", "服务告警恢复"]
    # A recipient change alone pushes nothing.
    other = await database.user.create(data={"username": "health-admin2-" + uuid4().hex, "role": "admin"})
    try:
        monkeypatch.setattr(h.settings, "redis_memory_alert_user_ids", f"{admin},{other.id}")
        await step(sample())
        assert await titles() == ["服务告警", "服务告警", "服务告警恢复"]
    finally:
        await database.execute_raw("DELETE FROM notification_events WHERE user_id=$1", other.id)
        await database.user.delete(where={"id": other.id})


async def test_unknown_pushes_only_after_it_persists(world, monkeypatch):
    database, *_, admin = world
    for expected in (0, 0, 1, 1):
        await age(database)
        monkeypatch.setattr(h, "collect_metrics", AsyncMock(return_value=sample(unknown=("memory_blocked",))))
        await h.sample_and_notify()
        assert len(await database.query_raw("SELECT id FROM notification_events WHERE user_id=$1", admin)) == expected


async def test_a_quiet_first_sample_sends_nothing(world, monkeypatch):
    database, *_, admin = world
    monkeypatch.setattr(h, "collect_metrics", AsyncMock(return_value=sample()))
    await h.sample_and_notify()
    assert await database.query_raw("SELECT id FROM notification_events WHERE user_id=$1", admin) == []
    assert (await h.read_status())["level"] == "normal"
