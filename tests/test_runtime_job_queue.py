"""State-transition tests run against disposable Redis, including the real Lua scripts."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis

from app.api.admin.runtime_jobs import router
from app.services.auth import create_jwt
from app.services.runtime import distributed_lock, job_queue as q


@pytest.fixture
async def redis(monkeypatch):
    if os.environ.get("APP_ENV") == "production":
        pytest.fail("Runtime queue tests refuse production mode")
    url = os.environ.get("RUNTIME_JOB_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
    if urlsplit(url).hostname not in {"127.0.0.1", "localhost", "test-redis"}:
        pytest.fail("Runtime queue tests require explicitly isolated local Redis")
    client = Redis.from_url(url, decode_responses=True, max_connections=256)
    await client.ping()
    prefix = f"test:runtime-terminal:{uuid.uuid4().hex}:"
    for field in ("_READY_KEY", "_DELAYED_KEY", "_RUNNING_KEY", "_DLQ_KEY",
                  "_SUCCEEDED_KEY", "_JOB_KEY_PREFIX", "_IDEMP_KEY_PREFIX"):
        monkeypatch.setattr(q, field, prefix + getattr(q, field))
    monkeypatch.setattr(q, "_HANDLERS", {})
    monkeypatch.setattr(q, "_LEGACY_NO_DELAY_HANDLERS", set())
    monkeypatch.setattr(q, "_RECOVERY_DELAYS", {})
    monkeypatch.setattr(q, "_RECONCILE_CURSOR", 0)
    monkeypatch.setattr(q, "_RECONCILE_PENDING", deque())
    monkeypatch.setattr(q, "_RECOVERY_OFFSET", 0)
    monkeypatch.setattr(distributed_lock, "_KEY_PREFIX", prefix + "lock")
    async def get_redis():
        return client
    monkeypatch.setattr(q, "get_redis", get_redis)
    monkeypatch.setattr(distributed_lock, "get_redis", get_redis)
    monkeypatch.setattr(distributed_lock, "is_redis_healthy", lambda: True)
    try:
        yield client
    finally:
        keys = [key async for key in client.scan_iter(match=prefix + "*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()


async def seed(redis, status, *, attempts=1):
    jid = await q.enqueue_runtime_job("test.job", {"x": 1})
    await redis.lrem(q._READY_KEY, 0, jid)
    await redis.hset(q._job_key(jid), mapping={"status": status, "attempts": attempts})
    if status == "dead_letter":
        await redis.lpush(q._DLQ_KEY, jid)
    return jid


async def snapshot(redis, jid):
    return {
        "record": await redis.hgetall(q._job_key(jid)),
        "ready": await redis.lrange(q._READY_KEY, 0, -1),
        "running": await redis.zrange(q._RUNNING_KEY, 0, -1, withscores=True),
        "delayed": await redis.zrange(q._DELAYED_KEY, 0, -1, withscores=True),
        "dead": await redis.lrange(q._DLQ_KEY, 0, -1),
        "succeeded": await redis.lrange(q._SUCCEEDED_KEY, 0, -1),
    }


@pytest.mark.asyncio
async def test_runtime_job_queue_runs_registered_handler(redis):
    calls = []
    async def handler(payload):
        calls.append(payload)
    q.register_job_handler("test.job", handler)
    jid = await q.enqueue_runtime_job("test.job", {"x": 1})
    assert await q.process_runtime_jobs(max_jobs=1) == 1
    assert calls == [{"x": 1}]
    listed = await q.list_runtime_jobs(status="succeeded")
    assert listed["items"][0]["id"] == jid
    assert listed["items"][0]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_runtime_job_queue_retries_then_dead_letters(redis):
    async def handler(payload):
        raise RuntimeError("synthetic failure")
    q.register_job_handler("test.job", handler)
    jid = await q.enqueue_runtime_job("test.job", {"x": 1}, max_attempts=2)
    await q.process_runtime_jobs(max_jobs=1)
    assert (await q.inspect_runtime_job(jid))["status"] == "queued"
    assert await redis.zscore(q._DELAYED_KEY, jid) is not None
    await redis.hset(q._job_key(jid), 'not_before', '1')
    await redis.zadd(q._DELAYED_KEY, {jid: 1})
    await q.process_runtime_jobs(max_jobs=1)
    item = await q.inspect_runtime_job(jid)
    assert item["status"] == "dead_letter"
    assert item["attempts"] == 2
    assert item["last_error"] == "synthetic failure"


@pytest.mark.asyncio
async def test_runtime_job_queue_idempotency_returns_existing_job(redis):
    first = await q.enqueue_runtime_job("test.job", {"x": 1}, idempotency_key="same")
    second = await q.enqueue_runtime_job("test.job", {"x": 2}, idempotency_key="same")
    assert first == second
    assert (await q.inspect_runtime_job(first))["payload"] == {"x": 1}


@pytest.mark.asyncio
async def test_runtime_job_admin_list_retry_and_resolve(redis):
    jid = await seed(redis, "dead_letter")
    assert (await q.list_runtime_jobs(status="dead_letter"))["counts"]["dead_letter"] == 1
    result = await q.retry_runtime_jobs([jid])
    assert result["retried_count"] == 1
    assert result["items"][0]["status"] == "queued"
    assert (await q.resolve_runtime_job(jid))["status"] == "resolved"
    assert await redis.llen(q._READY_KEY) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["succeeded", "resolved", "running", "dead_letter", "unknown"])
async def test_duplicate_ready_entry_never_executes_nonqueued_job(redis, status):
    jid = await seed(redis, status)
    before = await redis.hgetall(q._job_key(jid))
    calls = []
    async def handler(payload):
        calls.append(payload)
    q.register_job_handler("test.job", handler)
    await redis.lpush(q._READY_KEY, jid)
    await q.process_runtime_jobs(max_jobs=1)
    assert calls == []
    assert await redis.hgetall(q._job_key(jid)) == before


@pytest.mark.asyncio
async def test_success_cannot_execute_twice(redis):
    calls = []
    async def handler(payload):
        calls.append(payload)
    q.register_job_handler("test.job", handler)
    jid = await q.enqueue_runtime_job("test.job", {})
    await q._run_job_with_lock(jid)
    before = await snapshot(redis, jid)
    await q._run_job_with_lock(jid)
    assert calls == [{}]
    assert await snapshot(redis, jid) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["succeeded", "resolved", "running", "queued", "unknown"])
async def test_illegal_retry_is_a_conflict_without_mutation(redis, status):
    jid = await seed(redis, status)
    before = await snapshot(redis, jid)
    with pytest.raises(q.RuntimeJobConflict) as caught:
        await q.retry_runtime_job(jid)
    assert caught.value.status == status
    assert await snapshot(redis, jid) == before


@pytest.mark.asyncio
async def test_concurrent_manual_retries_enqueue_only_once(redis):
    jid = await seed(redis, "dead_letter", attempts=3)
    results = await asyncio.gather(*(q.retry_runtime_job(jid) for _ in range(30)),
                                   return_exceptions=True)
    assert sum(isinstance(item, dict) for item in results) == 1
    assert sum(isinstance(item, q.RuntimeJobConflict) for item in results) == 29
    assert await redis.lrange(q._READY_KEY, 0, -1) == [jid]
    assert (await q.inspect_runtime_job(jid))["attempts"] == 3


@pytest.mark.asyncio
async def test_competing_workers_cannot_start_the_same_attempt(redis):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def handler(payload):
        calls.append(payload)
        entered.set()
        await release.wait()
    q.register_job_handler("test.job", handler)
    jid = await q.enqueue_runtime_job("test.job", {})
    first = asyncio.create_task(q._run_job_with_lock(jid))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await q._run_job_with_lock(jid)
        assert calls == [{}]
    finally:
        release.set()
        await first
    assert (await q.inspect_runtime_job(jid))["attempts"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["succeeded", "resolved", "queued", "dead_letter"])
async def test_stale_running_index_cannot_revive_other_states(redis, state):
    jid = await seed(redis, state)
    before = await redis.hgetall(q._job_key(jid))
    await redis.zadd(q._RUNNING_KEY, {jid: 1})
    await q._recover_stale_running_jobs(1)
    assert await redis.hgetall(q._job_key(jid)) == before
    assert await redis.zscore(q._RUNNING_KEY, jid) is None
    assert await redis.llen(q._READY_KEY) == 0


@pytest.mark.asyncio
async def test_live_running_index_is_not_recovered(redis):
    jid = await seed(redis, "running")
    await redis.zadd(q._RUNNING_KEY, {jid: int(time.time())})
    before = await snapshot(redis, jid)
    await q._recover_stale_running_jobs(900)
    assert await snapshot(redis, jid) == before


@pytest.mark.asyncio
async def test_recovered_attempt_rejects_late_old_success_and_failure(redis):
    jid = await seed(redis, "running", attempts=1)
    await redis.zadd(q._RUNNING_KEY, {jid: 1})
    await q._recover_stale_running_jobs(1)
    assert (await q.inspect_runtime_job(jid))["status"] == "queued"
    # Claim a later attempt without executing any business effect.
    claim = await q._claim_job(jid)
    assert claim is not None
    before = await snapshot(redis, jid)
    for state in ["succeeded", "dead_letter", "queued"]:
        assert not await q._finish_job(redis, jid, 1, state, "obsolete", retry_at=1)
        assert await snapshot(redis, jid) == before
    assert await q._finish_job(redis, jid, 2, "succeeded", "", lease_token=claim["lease_token"])
    after = await snapshot(redis, jid)
    assert not await q._finish_job(redis, jid, 2, "dead_letter", "duplicate")
    assert await snapshot(redis, jid) == after


@pytest.mark.asyncio
async def test_expired_or_missing_jobs_are_never_recreated(redis):
    jid = await seed(redis, "running")
    await redis.delete(q._job_key(jid))
    await redis.zadd(q._RUNNING_KEY, {jid: 1})
    await q._recover_stale_running_jobs(1)
    await q._run_job_with_lock(jid)
    assert await q.retry_runtime_job(jid) is None
    assert await q.resolve_runtime_job(jid) is None
    assert not await q._finish_job(redis, jid, 1, "succeeded", "")
    assert await redis.exists(q._job_key(jid)) == 0


@pytest.mark.asyncio
async def test_batch_reports_partial_conflicts_and_missing_without_replaying_success(redis):
    failed = await seed(redis, "dead_letter")
    done = await seed(redis, "succeeded")
    running = await seed(redis, "running")
    result = await q.retry_runtime_jobs([failed, done, running, "missing", failed])
    assert result["retried_count"] == 1
    assert result["missing_ids"] == ["missing"]
    assert result["conflict_count"] == 2
    assert [item["status"] for item in result["conflicts"]] == ["succeeded", "running"]
    assert await redis.lrange(q._READY_KEY, 0, -1) == [failed]


@pytest.mark.asyncio
async def test_resolve_is_idempotent_and_removes_pending_entries(redis):
    jid = await seed(redis, "queued")
    await redis.lpush(q._READY_KEY, jid)
    await redis.zadd(q._DELAYED_KEY, {jid: 1})
    first = await q.resolve_runtime_job(jid)
    before = await snapshot(redis, jid)
    assert await q.resolve_runtime_job(jid) == first
    assert await snapshot(redis, jid) == before
    assert await redis.llen(q._READY_KEY) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["running", "succeeded"])
async def test_resolve_cannot_mask_active_or_successful_work(redis, state):
    jid = await seed(redis, state)
    before = await snapshot(redis, jid)
    with pytest.raises(q.RuntimeJobConflict):
        await q.resolve_runtime_job(jid)
    assert await snapshot(redis, jid) == before


@pytest.mark.asyncio
async def test_unknown_handler_still_dead_letters_normally(redis):
    jid = await q.enqueue_runtime_job("unregistered", {}, max_attempts=1)
    await q.process_runtime_jobs(max_jobs=1)
    assert (await q.inspect_runtime_job(jid))["status"] == "dead_letter"


@pytest.mark.asyncio
async def test_admin_http_state_conflicts_and_normal_failure_retry(redis):
    app = FastAPI()
    app.include_router(router)
    done = await seed(redis, "succeeded")
    failed = await seed(redis, "dead_letter")
    headers = {"Authorization": "Bearer " + create_jwt("admin", "admin")}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/admin-api/runtime-jobs/{done}/retry", headers=headers)
        assert response.status_code == 409
        assert "succeeded" in response.json()["detail"]
        assert (await client.post(f"/admin-api/runtime-jobs/{done}/resolve", headers=headers)).status_code == 409
        assert (await client.post("/admin-api/runtime-jobs/missing/retry", headers=headers)).status_code == 404
        response = await client.post(f"/admin-api/runtime-jobs/{failed}/retry", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        response = await client.post("/admin-api/runtime-jobs/retry", headers=headers,
                                     json={"job_ids": [done, failed, "missing"]})
        assert response.status_code == 200
        assert response.json()["conflict_count"] == 2
        assert response.json()["missing_ids"] == ["missing"]


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [None, "user"])
async def test_admin_http_rejects_unauthorized_actions_before_mutation(redis, role):
    app = FastAPI()
    app.include_router(router)
    jid = await seed(redis, "dead_letter")
    before = await snapshot(redis, jid)
    headers = {} if role is None else {"Authorization": "Bearer " + create_jwt("user", role)}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for path, body in [(f"{jid}/retry", None), (f"{jid}/resolve", None),
                           ("retry", {"job_ids": [jid]})]:
            response = await client.post("/admin-api/runtime-jobs/" + path, headers=headers, json=body)
            assert response.status_code == (401 if role is None else 403)
    assert await snapshot(redis, jid) == before


@pytest.mark.asyncio
async def test_byte_responses_and_existing_expiry_survive_state_transitions(redis, monkeypatch):
    binary = Redis.from_url(os.environ.get("RUNTIME_JOB_TEST_REDIS_URL", "redis://127.0.0.1:6379/15"), decode_responses=False)
    async def get_redis():
        return binary
    monkeypatch.setattr(q, "get_redis", get_redis)
    try:
        jid = await seed(redis, "dead_letter", attempts=3)
        await redis.expire(q._job_key(jid), 120)
        retried = await q.retry_runtime_job(jid)
        assert retried["status"] == "queued" and retried["attempts"] == 3
        assert 0 < await redis.ttl(q._job_key(jid)) <= 120
        async def handler(payload):
            assert payload == {"x": 1}
        q.register_job_handler("test.job", handler)
        await q._run_job_with_lock(jid)
        assert (await q.inspect_runtime_job(jid))["status"] == "succeeded"
        assert 0 < await redis.ttl(q._job_key(jid)) <= 120
    finally:
        await binary.aclose()


@pytest.mark.asyncio
async def test_concurrent_stale_recovery_enqueues_one_retry(redis):
    jid = await seed(redis, "running")
    await redis.zadd(q._RUNNING_KEY, {jid: 1})
    await asyncio.gather(*(q._recover_stale_running_jobs(1) for _ in range(30)))
    assert await redis.lrange(q._READY_KEY, 0, -1) == [jid]
    assert (await q.inspect_runtime_job(jid))["status"] == "queued"


@pytest.mark.asyncio
async def test_invalid_payload_still_dead_letters_without_calling_handler(redis):
    jid = await q.enqueue_runtime_job("test.job", {}, max_attempts=1)
    await redis.hset(q._job_key(jid), "payload", "{invalid")
    calls = []
    async def handler(payload):
        calls.append(payload)
    q.register_job_handler("test.job", handler)
    await q.process_runtime_jobs(max_jobs=1)
    assert calls == []
    assert (await q.inspect_runtime_job(jid))["status"] == "dead_letter"


@pytest.mark.asyncio
@pytest.mark.parametrize("path,body", [
    ("job/retry", None), ("job/resolve", None), ("retry", {"job_ids": ["job"]}),
])
async def test_admin_http_redis_unavailable_is_service_unavailable(redis, monkeypatch, path, body):
    unavailable = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    async def get_redis():
        return unavailable
    monkeypatch.setattr(q, "get_redis", get_redis)
    app = FastAPI()
    app.include_router(router)
    headers = {"Authorization": "Bearer " + create_jwt("admin", "admin")}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/admin-api/runtime-jobs/" + path, headers=headers, json=body)
            assert response.status_code == 503
            assert response.json()["detail"] == "任务队列暂时不可用，请刷新状态后重试。"
    finally:
        await unavailable.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["succeeded", "resolved", "running", "dead_letter", "queued"])
async def test_busy_lock_only_defers_queued_jobs(redis, state):
    jid = await seed(redis, state)
    before = await redis.hgetall(q._job_key(jid))
    async with distributed_lock.distributed_lock(f"runtime_job:{jid}", ttl_s=60):
        await q._run_job(jid)
    assert await redis.hgetall(q._job_key(jid)) == before
    assert (await redis.zscore(q._DELAYED_KEY, jid) is not None) == (state == "queued")


@pytest.mark.asyncio
async def test_success_history_remains_bounded(redis):
    jid = await seed(redis, "running")
    await redis.lpush(q._SUCCEEDED_KEY, *[f"old-{n}" for n in range(501)])
    assert await q._finish_job(redis, jid, 1, "succeeded", "")
    history = await redis.lrange(q._SUCCEEDED_KEY, 0, -1)
    assert len(history) == 500 and history[0] == jid


@pytest.mark.asyncio
async def test_concurrent_retry_and_resolve_cannot_restore_resolved_job(redis):
    jid = await seed(redis, "dead_letter")
    results = await asyncio.gather(
        *(q.retry_runtime_job(jid) if n % 2 == 0 else q.resolve_runtime_job(jid)
          for n in range(30)), return_exceptions=True)
    assert all(isinstance(result, (dict, q.RuntimeJobConflict)) for result in results)
    assert (await q.inspect_runtime_job(jid))["status"] == "resolved"
    assert await redis.llen(q._READY_KEY) == 0
    assert await redis.llen(q._DLQ_KEY) == 0
