import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from tests.test_service_health_postgres import world, message
from tests.test_runtime_job_queue import redis
from app.services.ops import service_health, task_backlog, reply_slow, worker_health
from app.services.runtime import job_queue


async def test_due_followup_and_terminal_evidence_against_real_sql(world, monkeypatch):
    database, conv, *_ = world
    monkeypatch.setattr(
        task_backlog,
        "read_monitor_jobs",
        AsyncMock(return_value={"items": [], "truncated": False}),
    )
    item = await message(database, conv, "user", 10)
    # Existing synthetic rows from other suites may exceed the bounded page.
    # Place this fixture first without deleting or changing their data.
    await database.execute_raw(
        "UPDATE message_followup_events SET next_dispatch_at='1900-01-01'::timestamptz WHERE message_id=$1",
        item.id,
    )
    result = await task_backlog.measure(datetime.now(UTC), {})
    assert item.id in result["_carry"]["ids"]["message_followup_events"]
    assert result["alerting"]
    # Retry scheduling moves the due date; that is not proof of completion.
    await database.execute_raw(
        "UPDATE message_followup_events SET next_dispatch_at=NOW()+INTERVAL '1 hour' WHERE message_id=$1",
        item.id,
    )
    result = await task_backlog.measure(datetime.now(UTC), result["_carry"])
    assert item.id in result["_carry"]["ids"]["message_followup_events"]
    await database.execute_raw(
        "UPDATE message_followup_events SET status='completed' WHERE message_id=$1",
        item.id,
    )
    result = await task_backlog.measure(datetime.now(UTC), result["_carry"])
    assert item.id not in result["_carry"]["ids"]["message_followup_events"]


async def test_processing_deduplicates_bubbles_excludes_intentional_wait_and_malformed(
    world,
):
    database, conv, *_ = world
    now = datetime(2042, 1, 1, 12, tzinfo=UTC)

    async def add(minutes, diagnostics):
        row = await message(
            database, conv, "assistant", minutes, response_diagnostics=diagnostics
        )
        await database.message.update(
            where={"id": row.id},
            data={"createdAt": (now - timedelta(minutes=minutes)).replace(tzinfo=None)},
        )

    for minutes in (3, 2, 1):
        d = {
            "timing_definition_version": "server-turn-timing-v1",
            "generation_ms": 52000,
            "in_graph_delay_ms": 30000,
            "first_bubble_at": (now - timedelta(minutes=minutes)).isoformat(),
        }
        await add(minutes, d)
        await add(minutes, d)
    result = await reply_slow.measure(now + timedelta(seconds=1), {})
    assert result["count"] == 3 and result["timed_turns"] == 3 and result["alerting"]
    for n in range(3):
        d = {
            "timing_definition_version": "server-turn-timing-v1",
            "generation_ms": 35000,
            "in_graph_delay_ms": 30000,
            "first_bubble_at": (now + timedelta(seconds=n + 2)).isoformat(),
        }
        await add(-(n + 2) / 60, d)
    await add(
        0,
        {
            "timing_definition_version": "server-turn-timing-v1",
            "generation_ms": "bad",
            "in_graph_delay_ms": 0,
            "first_bubble_at": "bad",
        },
    )
    result = await reply_slow.measure(now + timedelta(seconds=6), result["_carry"])
    assert not result["alerting"] and result["recovery_turns"] == 3


async def test_real_redis_oldest_ready_expired_lease_and_readonly_payload_boundary(
    redis, monkeypatch
):
    jid = await job_queue.enqueue_runtime_job("synthetic", {"secret": "must-not-leak"})
    old = time.time() - 600
    await redis.hset(job_queue._job_key(jid), mapping={"not_before": str(old)})
    newer = await job_queue.enqueue_runtime_job("synthetic", {})
    running = await job_queue.enqueue_runtime_job("synthetic", {})
    await redis.lrem(job_queue._READY_KEY, 0, running)
    await redis.hset(
        job_queue._job_key(running),
        mapping={"status": "running", "lease_expires_at": str(old)},
    )
    await redis.zadd(job_queue._RUNNING_KEY, {running: old})
    before = await redis.hgetall(job_queue._job_key(jid))
    read = await job_queue.read_monitor_jobs(limit=1)
    assert read["truncated"] and jid in {r["id"] for r in read["items"]}
    assert "payload" not in str(read) and "must-not-leak" not in str(read)
    monkeypatch.setattr(service_health, "_query", AsyncMock(return_value=[]))
    found = await task_backlog.measure(datetime.now(UTC), {})
    assert set(found["_carry"]["ids"]["redis_runtime"]) == {jid, running}
    assert newer not in found["_carry"]["ids"]["redis_runtime"]
    assert await redis.hgetall(job_queue._job_key(jid)) == before


async def test_real_redis_worker_registration_heartbeat_shutdown_scope_and_bounds(
    redis, monkeypatch
):
    prefix = job_queue._JOB_KEY_PREFIX + "workers"
    monkeypatch.setattr(worker_health, "_PREFIX", prefix)
    monkeypatch.setattr(worker_health, "_REGISTRY", prefix + ":registry")
    monkeypatch.setattr(worker_health, "_START", prefix + ":start")
    monkeypatch.setattr(worker_health, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(worker_health, "_INCARNATION", "first")
    await worker_health.heartbeat()
    assert 0 < await redis.ttl(prefix + ":first") <= 90
    first = json.loads(await redis.hget(prefix + ":registry", "first"))
    await worker_health.heartbeat()
    assert (
        json.loads(await redis.hget(prefix + ":registry", "first"))["started_at"]
        == first["started_at"]
    )
    await worker_health.heartbeat(closed=True)
    assert not await redis.exists(prefix + ":first")
    assert json.loads(await redis.hget(prefix + ":registry", "first"))["closed"]
    monkeypatch.setattr(worker_health, "_INCARNATION", "second")
    await worker_health.heartbeat()
    result = await worker_health.measure(datetime.now(UTC), {})
    assert result["count"] == 1 and result["abnormal_exits"] == 0
    monkeypatch.setattr(worker_health, "_LIMIT", 2)
    monkeypatch.setattr(worker_health, "_INCARNATION", "third")
    with pytest.raises(RuntimeError, match="full"):
        await worker_health.heartbeat()


async def test_observation_only_is_persisted_but_never_notifies(world, monkeypatch):
    database, *_ = world
    monkeypatch.setattr(service_health.provider_budget, "refresh_active", AsyncMock())
    monkeypatch.setattr(service_health.provider_failing, "flush", AsyncMock())
    good = {"status": "measured", "count": 0, "alerting": False}
    snapshot = {
        "sampled_at": datetime.now(UTC).isoformat(),
        "carry": {},
        "checks": {
            "reply_failed": good,
            "task_backlog": {**good, "alerting": True, "observation_only": True},
            "worker_exit": {"status": "unknown", "observation_only": True},
            "reply_slow": good,
        },
    }
    monkeypatch.setattr(
        service_health, "collect_metrics", AsyncMock(return_value=snapshot)
    )
    push = AsyncMock()
    monkeypatch.setattr(service_health, "enqueue_notification", push)
    await service_health.sample_and_notify()
    push.assert_not_awaited()
    records = await database.query_raw(
        "SELECT checks FROM ops_service_observations ORDER BY sampled_at DESC LIMIT 1"
    )
    assert records[0]["checks"]["task_backlog"]["alerting"]
    assert "carry" not in records[0]["checks"]
    view = await service_health.read_status()
    assert view["observation"]["samples"] >= 1
    assert view["observation"]["unknown_samples"] >= 1
    assert view["observation"]["warning_samples"] >= 1


async def test_corrupt_or_expired_redis_task_cannot_prove_recovery(redis, monkeypatch):
    monkeypatch.setattr(service_health, "_query", AsyncMock(return_value=[]))
    jid = await job_queue.enqueue_runtime_job("synthetic", {})
    await redis.hset(job_queue._job_key(jid), mapping={"not_before": "not-a-clock"})
    result = await task_backlog.measure(
        datetime.now(UTC), {"ids": {"redis_runtime": [jid, "expired-owned-task"]}}
    )
    assert (
        result["status"] == "unknown" and result["sources"]["redis_runtime"]["missing"]
    )
    assert jid in result["_carry"]["ids"]["redis_runtime"]
    assert "expired-owned-task" in result["_carry"]["ids"]["redis_runtime"]
    await redis.hset(job_queue._job_key(jid), mapping={"status": "succeeded"})
    next_result = await task_backlog.measure(
        datetime.now(UTC), {"ids": {"redis_runtime": [jid]}}
    )
    assert jid not in next_result["_carry"]["ids"]["redis_runtime"]


async def test_missing_worker_registry_and_start_clock_are_unknown(redis, monkeypatch):
    prefix = job_queue._JOB_KEY_PREFIX + "worker-missing-clock"
    monkeypatch.setattr(worker_health, "_PREFIX", prefix)
    monkeypatch.setattr(worker_health, "_REGISTRY", prefix + ":registry")
    monkeypatch.setattr(worker_health, "_START", prefix + ":start")
    monkeypatch.setattr(worker_health, "get_redis", AsyncMock(return_value=redis))
    with pytest.raises(RuntimeError, match="unavailable"):
        await worker_health.measure(datetime.now(UTC), {})
    await worker_health.heartbeat()
    await redis.delete(prefix + ":start")
    with pytest.raises(RuntimeError, match="unavailable"):
        await worker_health.measure(datetime.now(UTC), {})
