"""Real Redis boundaries for enqueue, ownership, recovery and old wire data."""
from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from redis.exceptions import ConnectionError, ResponseError

from app.api.admin.runtime_jobs import router
from app.services.auth import create_jwt
from app.services.runtime import job_queue as q
from app.services import life_story
from tests.test_runtime_job_queue import redis, seed, snapshot  # One shared isolation policy.


async def reconcile_all():
    totals = {"inspected": 0, "repaired": 0, "unverified": 0}
    for _ in range(100):
        result = await q.reconcile_runtime_job_indexes()
        for key, value in result.items():
            totals[key] += value
        if q._RECONCILE_CURSOR == 0 and not q._RECONCILE_PENDING:
            return totals
    pytest.fail("Synthetic Redis scan did not finish within its bounded test window")


async def revoke(redis, claim):
    jid = claim["id"]
    await redis.delete(q._job_lock_key(jid))
    await redis.hset(q._job_key(jid), "lease_expires_at", "0")
    await redis.zadd(q._RUNNING_KEY, {jid: 0})


@pytest.mark.parametrize("delay", [0, 60])
async def test_concurrent_enqueue_has_one_record_binding_and_index(redis, delay):
    ids = await asyncio.gather(*(
        q.enqueue_runtime_job("test.job", {"n": n}, idempotency_key="same", delay_s=delay)
        for n in range(128)
    ))
    assert len(set(ids)) == 1
    jid = ids[0]
    assert await redis.get(q._idempotency_key("same")) == jid
    assert await redis.llen(q._READY_KEY) == (0 if delay else 1)
    assert await redis.zcard(q._DELAYED_KEY) == (1 if delay else 0)
    assert len([k async for k in redis.scan_iter(match=q._JOB_KEY_PREFIX + "*")]) == 1
    assert (await q.inspect_runtime_job(jid))["attempts"] == 0
    assert abs(await redis.pttl(q._job_key(jid)) - await redis.pttl(q._idempotency_key("same"))) < 100


async def test_changed_binding_is_rechecked_without_creating_another_job(redis, monkeypatch):
    jid = await q.enqueue_runtime_job("test.job", {}, idempotency_key="same")
    original = redis.get
    reads = 0
    async def changed(key):
        nonlocal reads
        reads += 1
        return None if reads == 1 else await original(key)
    monkeypatch.setattr(redis, "get", changed)
    assert await q.enqueue_runtime_job("test.job", {"different": True}, idempotency_key="same") == jid
    assert await redis.llen(q._READY_KEY) == 1
    assert (await q.inspect_runtime_job(jid))["payload"] == {}


async def test_orphan_binding_is_explicit_and_never_silently_replayed(redis):
    jid = uuid.uuid4().hex
    key = q._idempotency_key("private-binding-name")
    await redis.set(key, jid, ex=120)
    with pytest.raises(q.RuntimeJobOrphaned) as caught:
        await q.enqueue_runtime_job("test.job", {"private": "data"}, idempotency_key="private-binding-name")
    assert caught.value.job_id == jid
    assert await redis.get(key) == jid
    assert 0 < await redis.ttl(key) <= 120
    assert await redis.exists(q._job_key(jid)) == 0
    assert await redis.llen(q._READY_KEY) == 0


@pytest.mark.parametrize("kind", ["ready", "delayed", "idempotency"])
async def test_wrong_key_types_fail_before_any_enqueue_writes(redis, kind):
    key = {"ready": q._READY_KEY, "delayed": q._DELAYED_KEY,
           "idempotency": q._idempotency_key("same")}[kind]
    if kind == "idempotency":
        await redis.lpush(key, "wrong-type")
    else:
        await redis.set(key, "wrong-type")
    expected = ResponseError if kind == 'idempotency' else q.RuntimeJobEnqueueUncertain
    with pytest.raises(expected) as caught:
        await q.enqueue_runtime_job("test.job", {}, idempotency_key="same")
    if kind != 'idempotency':
        assert isinstance(caught.value.__cause__, ResponseError)
    assert not [k async for k in redis.scan_iter(match=q._JOB_KEY_PREFIX + "*")]
    if kind != "idempotency":
        assert not await redis.exists(q._idempotency_key("same"))


@pytest.mark.parametrize("payload,delay", [([], 0), ({"invalid": object()}, 0), ({"nan": float('nan')}, 0), ({}, 7 * 24 * 3600)])
async def test_invalid_enqueue_inputs_leave_no_task_or_idempotency_key(redis, payload, delay):
    with pytest.raises((ValueError, TypeError)):
        await q.enqueue_runtime_job("test.job", payload, idempotency_key="same", delay_s=delay)
    assert not await redis.exists(q._idempotency_key("same"))
    assert await redis.llen(q._READY_KEY) == 0


async def test_claim_removes_all_duplicates_and_records_one_owner_atomically(redis):
    jid = await q.enqueue_runtime_job("test.job", {})
    await redis.lpush(q._READY_KEY, *([jid] * 20))
    claims = await asyncio.gather(*(q._claim_job(jid, require_ready=True) for _ in range(64)))
    winners = [c for c in claims if c]
    assert len(winners) == 1
    claim = winners[0]
    assert claim["attempts"] == "1"
    assert await redis.llen(q._READY_KEY) == 0
    assert await redis.zscore(q._RUNNING_KEY, jid) == pytest.approx(float(claim["lease_expires_at"]))
    assert await redis.get(q._job_lock_key(jid)) == claim["lease_token"]
    assert await q._finish_job(redis, jid, 1, "succeeded", "", lease_token=claim["lease_token"])
    assert await redis.get(q._job_lock_key(jid)) is None


async def test_read_only_peek_or_missing_ready_entry_cannot_lose_or_claim_a_job(redis):
    jid = await q.enqueue_runtime_job("test.job", {})
    before = await snapshot(redis, jid)
    assert await redis.lindex(q._READY_KEY, -1) == jid
    assert await snapshot(redis, jid) == before
    await redis.lrem(q._READY_KEY, 0, jid)
    assert await q._claim_job(jid, require_ready=True) is None
    assert (await q.inspect_runtime_job(jid))["status"] == "queued"
    assert (await reconcile_all())["repaired"] == 1
    assert await redis.lrange(q._READY_KEY, 0, -1) == [jid]


async def test_due_promotion_concurrency_is_atomic_and_deduplicated(redis):
    jid = await q.enqueue_runtime_job("test.job", {}, delay_s=5)
    await redis.zadd(q._DELAYED_KEY, {jid: 1})
    await redis.lpush(q._READY_KEY, jid)
    await asyncio.gather(*(q._promote_due_jobs() for _ in range(64)))
    assert await redis.lrange(q._READY_KEY, 0, -1) == [jid]
    assert await redis.zscore(q._DELAYED_KEY, jid) is None


async def test_future_delay_survives_lost_index_and_stale_ready_entry(redis):
    calls = []
    async def handler(payload):
        calls.append(payload)
    q.register_job_handler("test.job", handler)
    jid = await q.enqueue_runtime_job("test.job", {}, delay_s=120)
    due = await redis.zscore(q._DELAYED_KEY, jid)
    await redis.zrem(q._DELAYED_KEY, jid)
    assert (await reconcile_all())["repaired"] == 1
    assert await redis.zscore(q._DELAYED_KEY, jid) == due
    await redis.lpush(q._READY_KEY, jid)
    await q.process_runtime_jobs(10)
    assert calls == []
    assert await redis.llen(q._READY_KEY) == 0
    assert await redis.zscore(q._DELAYED_KEY, jid) == due


async def test_missing_delay_index_with_stale_ready_entry_never_executes_early(redis):
    handler = AsyncMock()
    q.register_job_handler('test.job', handler)
    jid = await q.enqueue_runtime_job('test.job', {}, delay_s=120)
    due = await redis.zscore(q._DELAYED_KEY, jid)
    await redis.zrem(q._DELAYED_KEY, jid)
    await redis.lpush(q._READY_KEY, jid)
    await q.process_runtime_jobs(1)
    handler.assert_not_awaited()
    assert await redis.llen(q._READY_KEY) == 0
    assert await redis.zscore(q._DELAYED_KEY, jid) == due
    assert (await q.inspect_runtime_job(jid))['attempts'] == 0


@pytest.mark.parametrize('committed', [False, True])
async def test_enqueue_response_loss_does_not_allow_uncoordinated_local_replay(redis, monkeypatch, committed):
    original = redis.eval
    async def lose_response(*args):
        if committed:
            await original(*args)
        raise ConnectionError('synthetic lost response')
    with monkeypatch.context() as patch:
        patch.setattr(redis, 'eval', lose_response)
        with pytest.raises(q.RuntimeJobEnqueueUncertain) as caught:
            await q.enqueue_runtime_job('test.job', {}, idempotency_key='same')
    jid = caught.value.job_id
    assert await redis.exists(q._job_key(jid)) == int(committed)
    retried = await q.enqueue_runtime_job('test.job', {}, idempotency_key='same')
    if committed:
        assert retried == jid
    assert await redis.lrange(q._READY_KEY, 0, -1) == [retried]


@pytest.mark.parametrize('stage', ['queued', 'initializing', 'llm_generating', 'complete', 'failed', None])
async def test_enqueue_uncertainty_cannot_overwrite_started_or_completed_progress(redis, monkeypatch, stage):
    prefix = q._READY_KEY.removesuffix('runtime:jobs:ready') + 'progress:'
    monkeypatch.setattr(life_story, 'PROGRESS_KEY_PREFIX', prefix)
    monkeypatch.setattr(life_story, 'get_redis', AsyncMock(return_value=redis))
    if stage:
        await life_story.set_progress('agent', stage, message='existing')
    before = await redis.get(prefix + 'agent')
    await life_story.set_progress('agent', 'failed', message='enqueue_uncertain', expected_stage='queued')
    after = await redis.get(prefix + 'agent')
    if stage == 'queued':
        assert json.loads(after)['stage'] == 'failed'
        assert json.loads(after)['message'] == 'enqueue_uncertain'
    else:
        assert after == before


@pytest.mark.parametrize("status", ["succeeded", "resolved", "running", "dead_letter", "unknown"])
async def test_stale_delayed_entry_does_not_revive_nonqueued_state(redis, status):
    jid = await seed(redis, status)
    before = await redis.hgetall(q._job_key(jid))
    await redis.zadd(q._DELAYED_KEY, {jid: 1})
    await q._promote_due_jobs()
    assert await redis.zscore(q._DELAYED_KEY, jid) is None
    assert await redis.llen(q._READY_KEY) == 0
    assert await redis.hgetall(q._job_key(jid)) == before


async def test_legacy_unknown_delay_is_held_but_audited_immediate_handler_can_recover(redis):
    jid = await seed(redis, "queued", attempts=0)
    await redis.hdel(q._job_key(jid), "not_before", "queue_version")
    before = await snapshot(redis, jid)
    assert (await reconcile_all())["unverified"] == 1
    assert await snapshot(redis, jid) == before
    async def handler(payload):
        assert payload == {"x": 1}
    q.register_job_handler("test.job", handler, legacy_no_delay=True)
    assert (await reconcile_all())["repaired"] == 1
    await q.process_runtime_jobs(1)
    assert (await q.inspect_runtime_job(jid))["status"] == "succeeded"


async def test_legacy_running_lock_prevents_overlap_and_missing_index_can_be_restored(redis):
    jid = await seed(redis, "running", attempts=1)
    await redis.hdel(q._job_key(jid), "queue_version", "not_before")
    await redis.hset(q._job_key(jid), "updated_at", "1")
    await redis.set(q._job_lock_key(jid), "legacy-owner", ex=120)
    assert (await reconcile_all())["repaired"] == 1
    before = await snapshot(redis, jid)
    await q._recover_stale_running_jobs(1)
    assert await snapshot(redis, jid) == before
    await redis.delete(q._job_lock_key(jid))
    await q._recover_stale_running_jobs(1)
    assert (await q.inspect_runtime_job(jid))["status"] == "queued"


async def test_lease_expiry_recovery_is_one_generation_and_late_results_cannot_commit(redis):
    jid = await q.enqueue_runtime_job("test.job", {})
    first = await q._claim_job(jid)
    await revoke(redis, first)
    before = await snapshot(redis, jid)
    assert not await q._finish_job(redis, jid, 1, "succeeded", "", lease_token=first["lease_token"])
    assert await snapshot(redis, jid) == before
    await asyncio.gather(*(q._recover_stale_running_jobs(1) for _ in range(32)))
    assert await redis.lrange(q._READY_KEY, 0, -1) == [jid]
    second = await q._claim_job(jid, require_ready=True)
    before = await snapshot(redis, jid)
    for status in ['queued', 'succeeded', 'dead_letter']:
        assert not await q._finish_job(redis, jid, 2, status, "old", lease_token=first["lease_token"])
        assert await snapshot(redis, jid) == before
    assert not await q._renew_job_lease(redis, first)
    assert await q._finish_job(redis, jid, 2, "succeeded", "", lease_token=second["lease_token"])
    item = await q.inspect_runtime_job(jid)
    assert item['recoveries'] == 1 and item['attempts'] == 2
    assert item['lease_expires_at'] is None
    assert 'lease_token' not in item


async def test_repeated_crashes_exhaust_budget_and_manual_retry_still_grants_one_attempt(redis):
    jid = await q.enqueue_runtime_job("test.job", {}, max_attempts=1)
    first = await q._claim_job(jid)
    await revoke(redis, first)
    await q._recover_stale_running_jobs(1)
    assert (await q.inspect_runtime_job(jid))["status"] == 'dead_letter'
    assert await redis.lrange(q._DLQ_KEY, 0, -1) == [jid]
    await q.retry_runtime_job(jid)
    calls = []
    async def handler(payload):
        calls.append(payload)
    q.register_job_handler("test.job", handler)
    await q.process_runtime_jobs(1)
    assert calls == [{}]
    assert (await q.inspect_runtime_job(jid))["attempts"] == 2


async def test_long_handler_is_renewed_without_recovery_or_overlap(redis, monkeypatch):
    monkeypatch.setattr(q, '_DEFAULT_LEASE_S', 1)
    monkeypatch.setattr(q, '_HEARTBEAT_INTERVAL_S', 0.1)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def handler(payload):
        calls.append(payload); entered.set(); await release.wait()
    q.register_job_handler('test.job', handler)
    jid = await q.enqueue_runtime_job('test.job', {})
    running = asyncio.create_task(q.process_runtime_jobs(1))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        initial = float(await redis.hget(q._job_key(jid), 'lease_expires_at'))
        await asyncio.sleep(1.25)
        await q._recover_stale_running_jobs(1)
        assert calls == [{}]
        assert float(await redis.hget(q._job_key(jid), 'lease_expires_at')) > initial
        assert (await q.inspect_runtime_job(jid))['status'] == 'running'
        assert (await q.inspect_runtime_job(jid))['lease_state'] == 'active'
    finally:
        release.set(); await running
    assert (await q.inspect_runtime_job(jid))['status'] == 'succeeded'


@pytest.mark.parametrize('failure', ['ownership', 'connection'])
async def test_lost_lease_or_renewal_connection_cancels_handler_and_preserves_recovery(redis, monkeypatch, failure):
    monkeypatch.setattr(q, '_DEFAULT_LEASE_S', 1)
    monkeypatch.setattr(q, '_HEARTBEAT_INTERVAL_S', 0.1)
    entered = asyncio.Event(); cancelled = asyncio.Event()
    async def handler(payload):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    q.register_job_handler('test.job', handler)
    jid = await q.enqueue_runtime_job('test.job', {})
    running = asyncio.create_task(q.process_runtime_jobs(1))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if failure == 'ownership':
            await redis.set(q._job_lock_key(jid), 'other-owner', ex=1)
        else:
            original = redis.eval
            async def fail(script, *a, **kw):
                if script == q._RENEW_JOB_LUA:
                    raise ConnectionError('synthetic renewal failure')
                return await original(script, *a, **kw)
            monkeypatch.setattr(redis, 'eval', fail)
        await asyncio.wait_for(running, 3)
        assert cancelled.is_set()
        item = await q.inspect_runtime_job(jid)
        assert item['status'] == 'running' and item['attempts'] == 1
        assert await redis.zscore(q._RUNNING_KEY, jid) is not None
        assert await redis.llen(q._SUCCEEDED_KEY) == 0
    finally:
        running.cancel(); await asyncio.gather(running, return_exceptions=True)


async def test_parent_cancellation_preserves_running_lease_for_crash_recovery(redis):
    entered = asyncio.Event(); stopped = asyncio.Event()
    async def handler(payload):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    q.register_job_handler('test.job', handler)
    jid = await q.enqueue_runtime_job('test.job', {})
    task = asyncio.create_task(q.process_runtime_jobs(1))
    await asyncio.wait_for(entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert (await q.inspect_runtime_job(jid))['status'] == 'running'
    assert await redis.zscore(q._RUNNING_KEY, jid) is not None


async def test_server_clock_controls_lease_even_with_bad_worker_wall_clock(redis, monkeypatch):
    monkeypatch.setattr(q.time, 'time', lambda: 0)
    jid = await q.enqueue_runtime_job('test.job', {})
    claim = await q._claim_job(jid)
    seconds, micros = await redis.time()
    assert float(claim['lease_expires_at']) > int(seconds) + int(micros) / 1e6
    await q._recover_stale_running_jobs(1)
    assert (await q.inspect_runtime_job(jid))['status'] == 'running'
    assert await q._finish_job(redis, jid, 1, 'succeeded', '', lease_token=claim['lease_token'])


async def test_expiring_record_is_not_resurrected_by_renewal_or_completion(redis):
    jid = await q.enqueue_runtime_job('test.job', {})
    claim = await q._claim_job(jid)
    await redis.delete(q._job_key(jid))
    assert not await q._renew_job_lease(redis, claim)
    assert not await q._finish_job(redis, jid, 1, 'succeeded', '', lease_token=claim['lease_token'])
    assert await redis.exists(q._job_key(jid)) == 0


async def test_multi_worker_burst_finishes_each_task_once(redis):
    calls = []
    async def handler(payload):
        calls.append(payload['n']); await asyncio.sleep(0)
    q.register_job_handler('test.job', handler)
    ids = await asyncio.gather(*(q.enqueue_runtime_job('test.job', {'n': n}) for n in range(100)))
    for _ in range(3):
        await asyncio.gather(*(q.process_runtime_jobs(25) for _ in range(8)))
    assert sorted(calls) == list(range(100))
    assert await redis.llen(q._READY_KEY) == await redis.zcard(q._RUNNING_KEY) == 0
    for jid in ids:
        assert (await q.inspect_runtime_job(jid))['status'] == 'succeeded'


async def test_oversized_scan_page_is_drained_across_ticks_without_losing_records(redis, monkeypatch):
    ids = await asyncio.gather(*(q.enqueue_runtime_job('test.job', {}) for _ in range(120)))
    await redis.delete(q._READY_KEY)
    scan = AsyncMock(return_value=(0, [q._job_key(jid) for jid in ids]))
    monkeypatch.setattr(redis, 'scan', scan)
    first = await q.reconcile_runtime_job_indexes(limit=80)
    second = await q.reconcile_runtime_job_indexes(limit=80)
    assert first == {'inspected': 80, 'repaired': 80, 'unverified': 0}
    assert second == {'inspected': 40, 'repaired': 40, 'unverified': 0}
    scan.assert_awaited_once()
    assert set(await redis.lrange(q._READY_KEY, 0, -1)) == set(ids)


async def test_locked_legacy_page_cannot_starve_expired_new_lease_recovery(redis):
    ids = await asyncio.gather(*(seed(redis, 'running') for _ in range(110)))
    await redis.zadd(q._RUNNING_KEY, {jid: 1 for jid in ids})
    for jid in ids:
        await redis.set(q._job_lock_key(jid), 'legacy-owner', ex=120)
    expired = await q.enqueue_runtime_job('test.job', {})
    claim = await q._claim_job(expired)
    await revoke(redis, claim)
    await redis.zadd(q._RUNNING_KEY, {expired: 2})
    await q._recover_stale_running_jobs(1)
    assert (await q.inspect_runtime_job(expired))['status'] == 'running'
    await q._recover_stale_running_jobs(1)
    assert (await q.inspect_runtime_job(expired))['status'] == 'queued'
    assert await redis.zcard(q._RUNNING_KEY) == 110


@pytest.mark.parametrize('transition', ['claim', 'renew', 'finish', 'recover', 'promote'])
async def test_transition_type_errors_never_partially_change_job_or_owner(redis, transition):
    jid = await q.enqueue_runtime_job('test.job', {})
    claim = None
    if transition in {'renew', 'finish', 'recover'}:
        claim = await q._claim_job(jid)
    if transition == 'recover':
        await revoke(redis, claim)
    bad_key = q._DELAYED_KEY if transition == 'promote' else q._RUNNING_KEY
    await redis.delete(bad_key)
    await redis.set(bad_key, 'wrong-index-type')
    before = await redis.hgetall(q._job_key(jid))
    owner = await redis.get(q._job_lock_key(jid))
    with pytest.raises(ResponseError):
        if transition == 'claim':
            await q._claim_job(jid)
        elif transition == 'renew':
            await q._renew_job_lease(redis, claim)
        elif transition == 'finish':
            await q._finish_job(redis, jid, 1, 'succeeded', '', lease_token=claim['lease_token'])
        elif transition == 'recover':
            await redis.eval(q._RECOVER_JOB_LUA, 6, q._job_key(jid), q._RUNNING_KEY,
                             q._READY_KEY, q._DELAYED_KEY, q._job_lock_key(jid), q._DLQ_KEY, jid, 1)
        else:
            await redis.eval(q._PROMOTE_JOB_LUA, 3, q._job_key(jid), q._DELAYED_KEY, q._READY_KEY, jid)
    assert await redis.hgetall(q._job_key(jid)) == before
    assert await redis.get(q._job_lock_key(jid)) == owner


async def test_diagnostics_are_read_only_private_and_require_admin(redis):
    jid = await seed(redis, 'queued', attempts=0)
    await redis.hdel(q._job_key(jid), 'not_before', 'queue_version')
    await redis.hset(q._job_key(jid), 'payload', json.dumps({'private': 'never-output'}))
    await redis.set(q._idempotency_key('private-binding-name'), 'missing', ex=60)
    before = await snapshot(redis, jid)
    anomalies = []
    cursor = idem = 0
    for _ in range(100):
        report = await q.diagnose_runtime_job_queue(cursor=cursor, idempotency_cursor=idem)
        anomalies.extend(report['anomalies'])
        cursor = report['next_cursor']; idem = report['next_idempotency_cursor']
        if cursor == idem == 0:
            break
    assert {'id': jid, 'issue': 'legacy_delay_unverified'} in anomalies
    assert {'id': 'missing', 'issue': 'idempotency_record_missing'} in anomalies
    assert 'never-output' not in json.dumps(report) and 'private-binding-name' not in json.dumps(report)
    assert await snapshot(redis, jid) == before
    app = FastAPI(); app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.get('/admin-api/runtime-jobs/diagnostics')).status_code == 401
        assert (await client.get('/admin-api/runtime-jobs/diagnostics', headers={'Authorization': 'Bearer ' + create_jwt('user', 'user')})).status_code == 403
        response = await client.get('/admin-api/runtime-jobs/diagnostics', headers={'Authorization': 'Bearer ' + create_jwt('admin', 'admin')})
        assert response.status_code == 200 and response.json()['read_only'] is True


@pytest.mark.parametrize('path', ['', '/missing', '/diagnostics'])
async def test_admin_read_routes_report_redis_outage_as_503(redis, monkeypatch, path):
    monkeypatch.setattr(q, 'get_redis', AsyncMock(side_effect=ConnectionError('offline')))
    app = FastAPI(); app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.get('/admin-api/runtime-jobs' + path, headers={'Authorization': 'Bearer ' + create_jwt('admin', 'admin')})
        assert response.status_code == 503
