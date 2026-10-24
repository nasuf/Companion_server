"""Real SQL claim/outbox/rollback; synthetic admins only."""
import asyncio
import os
from urllib.parse import urlsplit
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from prisma import Prisma

from app.services.ops import redis_capacity as c


@pytest.fixture
async def monitor(monkeypatch):
    url = os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("isolated PostgreSQL required")
    assert urlsplit(url).hostname in {"127.0.0.1", "localhost"}
    assert urlsplit(url).path == "/companion_proactive_e2e"
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    admin = await database.user.create(data={"username": "redis-test-" + uuid4().hex, "role": "admin"})
    monkeypatch.setattr(c, "db", database)
    monkeypatch.setattr(c.settings, "redis_memory_alert_user_ids", admin.id)
    await database.execute_raw("DELETE FROM ops_redis_capacity WHERE id='primary'")
    try:
        yield database, admin.id
    finally:
        await database.execute_raw("DELETE FROM notification_events WHERE user_id=$1", admin.id)
        await database.execute_raw("DELETE FROM ops_redis_capacity WHERE id='primary'")
        await database.user.delete(where={"id": admin.id})
        await database.disconnect()


async def age(database):
    await database.execute_raw("UPDATE ops_redis_capacity SET sampled_at=NOW()-INTERVAL '40 seconds'")


def sample(percent):
    return {"available": percent is not None, "capacity_status": "measured" if percent is not None else "unavailable",
            "mem_percent": percent}


async def test_concurrent_workers_one_alert_and_one_recovery(monitor, monkeypatch):
    database, uid = monitor
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value=sample(70)))
    await asyncio.gather(c.sample_and_notify(), c.sample_and_notify())
    rows = await database.query_raw("SELECT * FROM notification_events WHERE user_id=$1", uid)
    assert len(rows) == 1 and rows[0]["status"] == "pending"
    status = await c.read_status()
    assert status["level"] == "warning" and status["status"] == "active"
    assert status["notifications"] == [{"status": "pending", "count": 1}]
    # A failed sample must not clear the alert or enqueue a recovery.
    await age(database)
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value=sample(None)))
    await c.sample_and_notify()
    assert (await c.read_status())["level"] == "warning"
    assert len(await database.query_raw("SELECT id FROM notification_events WHERE user_id=$1", uid)) == 1
    await age(database)
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value=sample(64)))
    await c.sample_and_notify()
    assert (await c.read_status())["level"] == "normal"
    assert len(await database.query_raw("SELECT id FROM notification_events WHERE user_id=$1", uid)) == 2
    await database.execute_raw("UPDATE ops_redis_capacity SET sampled_at=NOW()-INTERVAL '3 minutes'")
    assert (await c.read_status())["status"] == "stale"


async def test_non_admin_no_notification(monitor, monkeypatch):
    database, uid = monitor
    await database.user.update(where={"id": uid}, data={"role": "user"})
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value=sample(90)))
    await c.sample_and_notify()
    assert (await c.read_status())["status"] == "unconfigured"
    assert await database.query_raw("SELECT id FROM notification_events WHERE user_id=$1", uid) == []


async def test_outbox_failure_rolls_back_and_next_tick_retries(monitor, monkeypatch):
    database, uid = monitor
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value=sample(90)))
    enqueue = c.enqueue_notification
    monkeypatch.setattr(c, "enqueue_notification", AsyncMock(side_effect=RuntimeError("synthetic")))
    with pytest.raises(RuntimeError):
        await c.sample_and_notify()
    assert await database.query_raw("SELECT * FROM ops_redis_capacity") == []
    monkeypatch.setattr(c, "enqueue_notification", enqueue)
    await c.sample_and_notify()
    assert len(await database.query_raw("SELECT id FROM notification_events WHERE user_id=$1", uid)) == 1


async def test_delivery_partial_failure_and_expired_records_are_not_success(monitor, monkeypatch):
    database, uid = monitor
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value=sample(80)))
    await c.sample_and_notify()
    await database.execute_raw("UPDATE notification_events SET status='sent', error='synthetic one device failed' WHERE user_id=$1", uid)
    status = await c.read_status()
    assert status["notifications"] == [{"status": "partial", "count": 1}]
    assert status["missing_notification_count"] == 0
    await database.execute_raw("DELETE FROM notification_events WHERE user_id=$1", uid)
    status = await c.read_status()
    assert status["missing_notification_count"] == 1
    assert status["notifications"] == []


async def test_ops_dispatch_does_not_claim_or_reset_unrelated_business_notifications(monitor, monkeypatch):
    from app.services.notifications import dispatcher
    database, uid = monitor
    monkeypatch.setattr(dispatcher, "db", database)
    monkeypatch.setattr(c, "collect_metrics", AsyncMock(return_value=sample(80)))
    await c.sample_and_notify()
    other_id = await c.enqueue_notification(user_id=uid, type="agent_message", title="synthetic",
        body="synthetic", payload={}, dedupe_key="other", dispatch_now=False, database=database)
    await database.execute_raw("UPDATE notification_events SET status='dispatching', updated_at=NOW()-INTERVAL '10 minutes' WHERE user_id=$1", uid)
    delivered = AsyncMock()
    monkeypatch.setattr(dispatcher, "_dispatch_one", delivered)
    assert await dispatcher.dispatch_due_notifications(limit=10, event_type="admin_redis_capacity") == 1
    assert delivered.await_args.args[0]["type"] == "admin_redis_capacity"
    other = await database.query_raw("SELECT status, updated_at < NOW()-INTERVAL '5 minutes' AS unchanged FROM notification_events WHERE id=$1", other_id)
    assert other == [{"status": "dispatching", "unchanged": True}]
