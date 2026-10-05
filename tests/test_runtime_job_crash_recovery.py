"""Actual SIGKILL boundaries against isolated Redis and synthetic effects."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from app.services.runtime import job_queue as q
from tests.test_runtime_job_queue import redis

WORKER = Path(__file__).parent / 'fixtures/runtime_queue_fault_worker.py'


def environment():
    return dict(os.environ, APP_ENV='test', PYTHON_DOTENV_DISABLED='1',
                N05_FAULT_TESTS='synthetic-only',
                RUNTIME_JOB_TEST_REDIS_URL=os.environ.get('RUNTIME_JOB_TEST_REDIS_URL', 'redis://127.0.0.1:6379/15'))


async def barrier(redis, process, prefix):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        value = await redis.get(prefix + 'barrier')
        if value:
            return json.loads(value)
        if process.poll() is not None:
            _, stderr = process.communicate()
            pytest.fail('Synthetic worker exited before its barrier: ' + stderr[-1500:])
        await asyncio.sleep(0.03)
    pytest.fail('Synthetic subprocess did not reach its bounded barrier')


async def kill(process):
    if process.poll() is None:
        process.kill()
    await asyncio.to_thread(process.wait, 5)


async def recover(redis, jid):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        await q._recover_stale_running_jobs(1)
        if (await q.inspect_runtime_job(jid))['status'] != 'running':
            return
        await asyncio.sleep(0.05)
    pytest.fail('Killed worker lease did not become recoverable')


async def handler_for(redis, prefix):
    async def handler(payload):
        await redis.incr(prefix + 'invocations')
        await redis.eval("if redis.call('SET', KEYS[1], '1', 'NX') then return redis.call('INCR', KEYS[2]) end return 0", 2, prefix + 'business-effect-identity', prefix + 'effects')
    q.register_job_handler('test.job', handler)


@pytest.mark.parametrize('mode', ['before_enqueue', 'after_enqueue'])
async def test_sigkill_enqueue_has_either_no_commit_or_one_complete_job(redis, mode):
    prefix = q._READY_KEY.removesuffix('runtime:jobs:ready')
    process = subprocess.Popen([sys.executable, str(WORKER), mode, prefix], env=environment(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        point = await barrier(redis, process, prefix)
        await kill(process)
        if mode == 'before_enqueue':
            assert not await redis.exists(q._job_key(point['job_id']))
            assert not await redis.exists(q._idempotency_key('same'))
        else:
            assert (await q.inspect_runtime_job(point['job_id']))['status'] == 'queued'
            assert await redis.lrange(q._READY_KEY, 0, -1) == [point['job_id']]
        jid = await q.enqueue_runtime_job('test.job', {'synthetic': True}, idempotency_key='same')
        await handler_for(redis, prefix)
        await q.process_runtime_jobs(1)
        assert (await q.inspect_runtime_job(jid))['status'] == 'succeeded'
        assert await redis.get(prefix + 'effects') == '1'
        assert await redis.get(prefix + 'invocations') == '1'
    finally:
        await kill(process)
        if process.stderr:
            process.stderr.close()


@pytest.mark.parametrize('mode', ['after_claim', 'before_finish', 'after_finish', 'heartbeat'])
async def test_sigkill_claim_handler_and_completion_remain_recoverable_without_stale_commit(redis, monkeypatch, mode):
    monkeypatch.setattr(q, '_DEFAULT_LEASE_S', 1)
    monkeypatch.setattr(q, '_HEARTBEAT_INTERVAL_S', 0.1)
    prefix = q._READY_KEY.removesuffix('runtime:jobs:ready')
    jid = await q.enqueue_runtime_job('test.job', {'synthetic': True})
    process = subprocess.Popen([sys.executable, str(WORKER), mode, prefix, '--job-id', jid], env=environment(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        await barrier(redis, process, prefix)
        before = await redis.hgetall(q._job_key(jid))
        if mode == 'heartbeat':
            await asyncio.sleep(1.25)
            await q._recover_stale_running_jobs(1)
            assert (await q.inspect_runtime_job(jid))['status'] == 'running'
            assert float(await redis.hget(q._job_key(jid), 'lease_expires_at')) > float(before['lease_expires_at'])
        await kill(process)
        if mode == 'after_finish':
            assert (await q.inspect_runtime_job(jid))['status'] == 'succeeded'
            # Old stale delivery after committed success must not repeat effects.
            await redis.lpush(q._READY_KEY, jid)
        else:
            assert (await q.inspect_runtime_job(jid))['status'] == 'running'
            assert await redis.zscore(q._RUNNING_KEY, jid) is not None
            await recover(redis, jid)
            assert (await q.inspect_runtime_job(jid))['recoveries'] == 1
        await handler_for(redis, prefix)
        await q.process_runtime_jobs(1)
        after = await q.inspect_runtime_job(jid)
        assert after['status'] == 'succeeded'
        assert await redis.get(prefix + 'effects') == '1'
        expected_invocations = 2 if mode in {'before_finish', 'heartbeat'} else 1
        assert int(await redis.get(prefix + 'invocations')) == expected_invocations
        assert after['attempts'] == (1 if mode == 'after_finish' else 2)
        assert await redis.lrange(q._SUCCEEDED_KEY, 0, -1) == [jid]
        assert await redis.get(q._job_lock_key(jid)) is None
        assert not await q._finish_job(redis, jid, 1, 'dead_letter', 'late', lease_token=before.get('lease_token', ''))
    finally:
        await kill(process)
        if process.stderr:
            process.stderr.close()


async def test_sigkill_after_delay_promotion_preserves_exactly_one_ready_entry(redis):
    prefix = q._READY_KEY.removesuffix('runtime:jobs:ready')
    jid = await q.enqueue_runtime_job('test.job', {})
    await redis.lrem(q._READY_KEY, 0, jid)
    await redis.zadd(q._DELAYED_KEY, {jid: 1})
    process = subprocess.Popen([sys.executable, str(WORKER), 'after_promote', prefix, '--job-id', jid], env=environment(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        await barrier(redis, process, prefix)
        await kill(process)
        await q._promote_due_jobs()
        assert await redis.lrange(q._READY_KEY, 0, -1) == [jid]
        assert await redis.zscore(q._DELAYED_KEY, jid) is None
        await handler_for(redis, prefix)
        await q.process_runtime_jobs(1)
        assert await redis.get(prefix + 'effects') == '1'
    finally:
        await kill(process)
        if process.stderr:
            process.stderr.close()


@pytest.mark.parametrize('override', [{'APP_ENV': 'production'}, {'RUNTIME_JOB_TEST_REDIS_URL': 'redis://203.0.113.1:6379/15'}])
def test_fault_worker_refuses_production_or_remote_redis_before_connection(override):
    env = environment()
    env.update(override)
    result = subprocess.run([sys.executable, str(WORKER), 'after_enqueue', 'test:runtime-terminal:' + 'a' * 32 + ':'], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert 'refuses production or non-isolated execution' in result.stderr
