"""Turn receipt recovery, actual consumer crashes and admin permission boundary."""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.api.admin.runtime_jobs import router
from app.services.auth import create_jwt
from app.services.runtime import turn_followups as t, job_queue as q
from tests.test_turn_followups_postgres import turn, followups, origins, redis, receipt


async def due(db, tid):
    await db.execute_raw("UPDATE turn_followup_events SET next_dispatch_at=NOW() WHERE turn_id=$1", tid)


async def test_turn_redis_loss_and_uncertain_enqueue_reconcile_once(turn, redis, monkeypatch):
    db, user, _, args = turn
    tid = args['turn_id']
    await t.register_completed_turn(**args)
    enqueue = t.enqueue_runtime_job

    async def uncertain(*a, **kw):
        await enqueue(*a, **kw)
        raise TimeoutError('private network data')

    monkeypatch.setattr(t, 'enqueue_runtime_job', uncertain)
    assert await t.dispatch_turn_followups(turn_id=tid) == 0
    assert (await receipt(db, tid))['last_error'] == 'TimeoutError'
    monkeypatch.setattr(t, 'enqueue_runtime_job', enqueue)
    await due(db, tid)
    assert await t.dispatch_turn_followups(turn_id=tid) == 1
    first = await receipt(db, tid)
    await due(db, tid)
    assert await t.dispatch_turn_followups(turn_id=tid) == 0
    await redis.delete(q._job_key(first['job_id']))
    await due(db, tid)
    assert await t.dispatch_turn_followups(turn_id=tid) == 1
    assert (await receipt(db, tid))['generation'] == 1
    await t.run_turn_followup({'turn_id': tid, 'generation': 0})
    assert (await receipt(db, tid))['status'] == 'queued'
    await t.run_turn_followup({'turn_id': tid, 'generation': 1})
    assert (await receipt(db, tid))['status'] == 'completed'
    assert await db.achievementunlock.count(where={'conversationId':user.conversationId,'achievementId':81}) == 1


async def test_turn_api_permissions_filter_retry_and_private_errors(turn, redis, monkeypatch):
    db, _, _, args = turn
    tid = args['turn_id']
    await t.register_completed_turn(**args)
    await t.dispatch_turn_followups(turn_id=tid)
    item = await receipt(db, tid)
    await redis.hset(q._job_key(item['job_id']), 'status', 'dead_letter')
    await due(db, tid)
    await t.dispatch_turn_followups(turn_id=tid)
    assert (await receipt(db, tid))['status'] == 'failed'
    app = FastAPI(); app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        path = '/admin-api/runtime-jobs/turn-followups'
        action = path + '/' + tid + '/retry'
        for method, url in [('get', path), ('post', action)]:
            call = getattr(client, method)
            assert (await call(url)).status_code == 401
            assert (await call(url, headers={'Authorization':'Bearer '+create_jwt('user',role='user')})).status_code == 403
        h = {'Authorization':'Bearer '+create_jwt('admin',role='admin')}
        for query in ['limit=51', 'limit=0', 'status=unknown']:
            assert (await client.get(path+'?'+query,headers=h)).status_code == 422
        result = (await client.get(path+'?status=failed', headers=h)).json()
        assert result['denominator'] == 'recent_sample_only' and result['limit'] == 20
        assert result['items'][0]['turn_id'] == tid
        assert set(result['items'][0]) == {'turn_id','message_id','status','job_id','generation','dispatch_attempts','last_error','outcome','created_at','updated_at'}
        assert (await client.post(path+'/missing/retry',headers=h)).status_code == 404
        response = await client.post(action,headers=h)
        assert response.json() == {'status':'pending_dispatch','turn_id':tid}
        assert (await client.post(action,headers=h)).status_code == 409
        monkeypatch.setattr(t,'inspect_turn_followups',AsyncMock(side_effect=RuntimeError('PRIVATE')))
        response = await client.get(path,headers=h)
        assert response.status_code == 503 and 'PRIVATE' not in response.text
        monkeypatch.setattr(t,'retry_turn_followup',AsyncMock(side_effect=RuntimeError('PRIVATE')))
        response = await client.post(action,headers=h)
        assert response.status_code == 503 and 'PRIVATE' not in response.text
    await t.dispatch_turn_followups(turn_id=tid)
    await t.run_turn_followup({'turn_id':tid,'generation':1})
    assert (await receipt(db,tid))['status'] == 'completed'


@pytest.mark.parametrize('phase', ['after_registration', 'after_effect', 'before_ack'])
async def test_turn_crash_recovers_registration_or_effect_without_duplicate_unlock(turn, redis, phase, tmp_path):
    db, user, _, args = turn
    tid = args['turn_id']
    jid = json.dumps(args)
    if phase != 'after_registration':
        await t.register_completed_turn(**args)
        await t.dispatch_turn_followups(turn_id=tid)
        jid = (await receipt(db,tid))['job_id']
    fields = ('_READY_KEY','_DELAYED_KEY','_RUNNING_KEY','_DLQ_KEY','_SUCCEEDED_KEY','_JOB_KEY_PREFIX','_IDEMP_KEY_PREFIX')
    log_path = tmp_path/'turn-crash.log'
    with log_path.open('wb') as log:
        process = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).parent/'fixtures/message_followup_crash_worker.py'), jid, phase,
            json.dumps({k:getattr(q,k) for k in fields}), 'turn', stdout=log, stderr=log, start_new_session=True)
        try:
            await asyncio.wait_for(process.wait(), timeout=20)
            assert process.returncode == 73, log_path.read_text()[-2000:]
        finally:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            if process.returncode is None: await asyncio.wait_for(process.wait(),timeout=5)
    if phase == 'after_registration':
        assert (await receipt(db,tid))['status'] == 'pending_dispatch'
        await t.dispatch_turn_followups(turn_id=tid)
        jid = (await receipt(db,tid))['job_id']
    else:
        assert (await q.inspect_runtime_job(jid))['status'] == 'running'
        assert await db.achievementunlock.count(where={'conversationId':user.conversationId,'achievementId':81}) == 1
        assert (await receipt(db,tid))['status'] == ('queued' if phase == 'after_effect' else 'completed')
        await redis.delete(q._job_lock_key(jid))
        await redis.hset(q._job_key(jid),'lease_expires_at','0')
        await redis.zadd(q._RUNNING_KEY,{jid:0})
        await q._recover_stale_running_jobs(1)
    await q.process_runtime_jobs(max_jobs=1)
    assert (await receipt(db,tid))['status'] == 'completed'
    assert (await q.inspect_runtime_job(jid))['status'] == 'succeeded'
    assert await db.achievementunlock.count(where={'conversationId':user.conversationId,'achievementId':81}) == 1


async def test_turn_retention_leaves_failed_and_pending_rows_and_original_messages(turn):
    db, _, replies, args = turn
    tid = args['turn_id']
    await t.register_completed_turn(**args)
    await db.execute_raw("UPDATE turn_followup_events SET updated_at=NOW()-INTERVAL '31 days' WHERE turn_id=$1",tid)
    assert await t.prune_turn_followups() == 0
    await db.execute_raw("UPDATE turn_followup_events SET status='failed' WHERE turn_id=$1",tid)
    assert await t.prune_turn_followups() == 0
    await db.execute_raw("UPDATE turn_followup_events SET status='cancelled' WHERE turn_id=$1",tid)
    assert await t.prune_turn_followups() == 1
    assert await db.message.find_unique(where={'id':replies[0].id}) is not None
