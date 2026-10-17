"""Real SQL registration and Redis recovery, using owned synthetic messages only."""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma import Json
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.admin.runtime_jobs import router
from app.services.auth import create_jwt
from app.services.runtime import message_followups as f, job_queue as q
from app.services.achievements import engine, source, repository, mode
from app.services.achievements.rules import user_message_rules, assistant_message_rules
from app.services import interaction_streak
from tests.test_memory_evidence_postgres import origins
from tests.test_runtime_job_queue import redis


@pytest.fixture
async def followups(origins, redis, monkeypatch):
    db, uid, wid, _, message, _, _, users, agents, spaces, convs = origins
    for module in (f, source, repository, interaction_streak, user_message_rules, assistant_message_rules):
        monkeypatch.setattr(module, 'db', db)
    for module in (engine, repository, mode):
        monkeypatch.setattr(module, 'achievement_evaluation_enabled', AsyncMock(return_value=True))
    monkeypatch.setattr(repository, 'achievement_alerts_enabled', AsyncMock(return_value=False))
    monkeypatch.setattr(repository, '_is_unlock_cached', AsyncMock(return_value=False))
    monkeypatch.setattr(repository, '_cache_unlocked_achievements', AsyncMock())
    q.register_job_handler(f.JOB_TYPE, f.run_message_followup)
    # Deterministic business rule with the real guarded event/unlock SQL. Other
    # achievement tests cover the full rules; this suite injects recovery faults.
    async def rule(event):
        identity = source.current_source.get()
        await repository.record_event(user_id=identity.user_id, agent_id=identity.agent_id,
            workspace_id=identity.workspace_id, conversation_id=identity.conversation_id,
            event_type=identity.role+'_message', source_id=identity.message_id, value_int=2)
        await repository.unlock_achievement(user_id=identity.user_id, agent_id=identity.agent_id,
            workspace_id=identity.workspace_id, conversation_id=identity.conversation_id,
            achievement_id=1)
    monkeypatch.setattr(engine, '_evaluate_event', rule)
    return db, message, users, agents, spaces, convs, rule


async def row(db, mid):
    values = await db.query_raw('SELECT * FROM message_followup_events WHERE message_id=$1', mid)
    return values[0] if values else None


async def due(db, mid):
    await db.execute_raw('UPDATE message_followup_events SET next_dispatch_at=NOW() WHERE message_id=$1', mid)


async def process_one(db, mid):
    await due(db, mid)
    await f.dispatch_message_followups(limit=1,message_id=mid)
    await q.process_runtime_jobs(max_jobs=1)
    return await row(db, mid)


async def test_registration_is_atomic_no_hook_and_rollback(followups):
    db, msg, *rest = followups
    assert (await row(db, msg.id))['status'] == 'pending_dispatch'
    mid=uuid4().hex
    with pytest.raises(RuntimeError):
        async with db.tx() as tx:
            await tx.message.create(data={'id':mid,'conversationId':msg.conversationId,'role':'user','content':'rollback'})
            assert await tx.query_raw('SELECT message_id FROM message_followup_events WHERE message_id=$1',mid)
            raise RuntimeError('rollback')
    assert await row(db,mid) is None
    result=await process_one(db,msg.id)
    assert result['status']=='completed' and result['outcome']=='statistics_evaluated'
    assert len(await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id))==1
    assert len(await db.query_raw('SELECT id FROM workspace_interaction_days WHERE workspace_id=$1',rest[2][0]))==1


@pytest.mark.parametrize('role',['user','assistant'])
async def test_partial_failure_then_replay_counts_once(followups,monkeypatch,role):
    db,msg,*_=followups
    await db.message.delete(where={'id':msg.id})
    msg=await db.message.create(data={'conversationId':msg.conversationId,'role':role,'content':'synthetic','metadata':Json({'proactive':True})})
    real=followups[-1]; calls=0
    async def fail_once(event):
        nonlocal calls
        await real(event); calls+=1
        if calls==1:raise RuntimeError('PRIVATE_SQL_AND_TEXT')
    monkeypatch.setattr(engine,'_evaluate_event',fail_once)
    first=await process_one(db,msg.id)
    assert first['status']=='queued' and first['last_error']=='RuntimeError'
    job=await q.inspect_runtime_job(first['job_id'])
    assert job['last_error']=='RuntimeError' and 'PRIVATE' not in str(job)
    await f.run_message_followup({'message_id':msg.id,'generation':0})
    await f.run_message_followup({'message_id':msg.id,'generation':0})
    assert calls==2 and (await row(db,msg.id))['status']=='completed'
    assert len(await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id))==1


async def test_missing_queue_recovers_generation_and_old_delivery_is_noop(followups,redis):
    db,msg,*_=followups
    await f.dispatch_message_followups(limit=1,message_id=msg.id)
    first=await row(db,msg.id)
    await redis.delete(q._job_key(first['job_id']))
    await due(db,msg.id)
    await f.dispatch_message_followups(limit=1,message_id=msg.id)
    second=await row(db,msg.id)
    assert second['generation']==1 and second['job_id']!=first['job_id']
    await f.run_message_followup({'message_id':msg.id,'generation':0})
    assert (await row(db,msg.id))['status']=='queued'
    await f.run_message_followup({'message_id':msg.id,'generation':1})
    assert (await row(db,msg.id))['status']=='completed'


async def test_uncertain_admission_reuses_same_job(followups,monkeypatch):
    db,msg,*_=followups
    real=f.enqueue_runtime_job
    async def lost(*args,**kwargs):
        jid=await real(*args,**kwargs)
        raise q.RuntimeJobEnqueueUncertain(jid)
    monkeypatch.setattr(f,'enqueue_runtime_job',lost)
    assert await f.dispatch_message_followups(limit=1,message_id=msg.id)==0
    assert (await row(db,msg.id))['status']=='pending_dispatch'
    monkeypatch.setattr(f,'enqueue_runtime_job',real)
    await due(db,msg.id);await f.dispatch_message_followups(limit=1,message_id=msg.id)
    assert (await row(db,msg.id))['generation']==0
    await q.process_runtime_jobs(max_jobs=2)
    assert (await row(db,msg.id))['status']=='completed'


@pytest.mark.parametrize('mutation',['content','scope','archive','deleted'])
async def test_changed_source_is_cancelled_before_effects(followups,mutation):
    db,msg,users,agents,spaces,convs,_=followups
    if mutation=='content':await db.message.update(where={'id':msg.id},data={'content':'changed'})
    if mutation=='scope':await db.message.update(where={'id':msg.id},data={'conversationId':convs[2]})
    if mutation=='archive':await db.chatworkspace.update(where={'id':spaces[0]},data={'status':'archived'})
    if mutation=='deleted':await db.conversation.update(where={'id':convs[0]},data={'isDeleted':True})
    result=await process_one(db,msg.id)
    assert result['status']=='cancelled'
    assert not await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id)


async def test_human_and_disabled_modes_are_explicit(followups,monkeypatch):
    db,msg,*_=followups
    await db.message.delete(where={'id':msg.id})
    msg=await db.message.create(data={'conversationId':msg.conversationId,'role':'user','content':'human','metadata':Json({'human_takeover':True})})
    assert (await process_one(db,msg.id))['outcome']=='achievement_source_unavailable'
    assert not await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id)
    monkeypatch.setattr(mode,'achievement_evaluation_enabled',AsyncMock(return_value=False))
    msg=await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'off'})
    assert (await process_one(db,msg.id))['outcome']=='achievement_disabled'


async def test_source_clock_survives_midnight_recovery(followups):
    db,msg,*_=followups
    await db.message.delete(where={'id':msg.id})
    msg=await db.message.create(data={'conversationId':msg.conversationId,'role':'user','content':'clock',
        'createdAt':datetime(2026,10,14,15,59,59,tzinfo=timezone.utc)})
    await process_one(db,msg.id)
    days=await db.query_raw('SELECT local_date::text AS day FROM workspace_interaction_days WHERE workspace_id=$1',followups[4][0])
    assert [x['day'] for x in days]==['2026-10-14']


async def test_audio_and_emotion_enrichment_do_not_invalidate_registered_statistics(followups):
    db,msg,*_=followups
    await db.message.update(where={'id':msg.id},data={'metadata':Json({
        'display_mode':'voice','attachments':[{'id':'synthetic-audio'}],
        'emotion':{'label':'happy'}})})
    assert (await process_one(db,msg.id))['status']=='completed'
    assert len(await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id))==1


async def test_api_auth_bounds_private_projection_and_db_failure(followups,monkeypatch):
    db,msg,*_=followups
    app=FastAPI();app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://test') as client:
        path='/admin-api/runtime-jobs/message-followups'
        assert (await client.get(path)).status_code==401
        assert (await client.get(path,headers={'Authorization':'Bearer '+create_jwt('ordinary',role='user')})).status_code==403
        h={'Authorization':'Bearer '+create_jwt('admin',role='admin')}
        assert (await client.get(path+'?limit=51',headers=h)).status_code==422
        assert (await client.get(path+'?status=invalid',headers=h)).status_code==422
        response=await client.get(path,headers=h);assert response.status_code==200
        result=response.json();assert result['denominator']=='recent_sample_only'
        assert 'content' not in str(result) and msg.content not in str(result)
        monkeypatch.setattr(f,'db',type('Broken',(),{'query_raw':AsyncMock(side_effect=RuntimeError('PRIVATE'))})())
        response=await client.get(path,headers=h)
        assert response.status_code==503 and 'PRIVATE' not in response.text


@pytest.mark.parametrize('status',['dead_letter','failed','resolved','succeeded','unknown'])
async def test_queue_terminal_without_sql_completion_is_never_success(followups,redis,status):
    db,msg,*_=followups
    await f.dispatch_message_followups(limit=1,message_id=msg.id)
    item=await row(db,msg.id)
    await redis.hset(q._job_key(item['job_id']),'status',status)
    await due(db,msg.id);await f.dispatch_message_followups(limit=1,message_id=msg.id)
    result=await row(db,msg.id)
    assert result['status']=='failed' and result['outcome'] is None
    if status!='unknown':
        assert await f.retry_message_followup(msg.id)
        pending=await row(db,msg.id)
        assert pending['generation']==1 and pending['status']=='pending_dispatch'
        with pytest.raises(f.FollowupRetryConflict):await f.retry_message_followup(msg.id)
        await f.dispatch_message_followups(limit=1,message_id=msg.id)
        await f.run_message_followup({'message_id':msg.id,'generation':1})
        assert (await row(db,msg.id))['status']=='completed'


async def test_orphaned_receipt_and_identity_mismatch(followups,redis):
    db,msg,*_=followups
    jid=await q.enqueue_runtime_job(f.JOB_TYPE,{'message_id':msg.id,'generation':0},idempotency_key=f'message-statistics:{msg.id}:0')
    await redis.delete(q._job_key(jid))
    await f.dispatch_message_followups(limit=1,message_id=msg.id)
    item=await row(db,msg.id)
    assert item['generation']==1 and item['last_error']=='queue_record_missing'
    await due(db,msg.id);await f.dispatch_message_followups(limit=1,message_id=msg.id)
    item=await row(db,msg.id)
    await redis.hset(q._job_key(item['job_id']),'type','another_handler')
    await due(db,msg.id);await f.dispatch_message_followups(limit=1,message_id=msg.id)
    assert (await row(db,msg.id))['last_error']=='queue_identity_mismatch'
    with pytest.raises(f.FollowupRetryConflict):await f.retry_message_followup(msg.id)


async def test_pending_queue_never_reenqueues_and_running_retry_conflicts(followups,redis):
    db,msg,*_=followups
    await f.dispatch_message_followups(limit=1,message_id=msg.id)
    first=await row(db,msg.id)
    await due(db,msg.id);await f.dispatch_message_followups(limit=1,message_id=msg.id)
    assert (await row(db,msg.id))['job_id']==first['job_id']
    await db.execute_raw("UPDATE message_followup_events SET status='failed' WHERE message_id=$1",msg.id)
    with pytest.raises(f.FollowupRetryConflict):await f.retry_message_followup(msg.id)
    await redis.delete(q._job_key(first['job_id']))
    assert await f.retry_message_followup(msg.id)
    assert (await row(db,msg.id))['generation']==1


async def test_retention_is_bounded_only_terminal_and_message_delete_cascades(followups):
    db,msg,*_=followups
    await db.execute_raw("UPDATE message_followup_events SET updated_at=NOW()-INTERVAL '31 days' WHERE message_id=$1",msg.id)
    assert await f.prune_message_followups()==0
    await db.execute_raw("UPDATE message_followup_events SET status='completed' WHERE message_id=$1",msg.id)
    assert await f.prune_message_followups()==1
    assert await db.message.find_unique(where={'id':msg.id}) is not None
    await f.run_message_followup({'message_id':msg.id,'generation':0})
    assert not await f.retry_message_followup(msg.id)
    other=await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'delete'})
    await db.message.delete(where={'id':other.id});assert await row(db,other.id) is None


async def test_retention_caps_a_large_terminal_backlog_at_100(followups):
    db,msg,*_=followups
    await db.message.create_many(data=[{'id':uuid4().hex,'conversationId':msg.conversationId,
        'role':'assistant','content':'synthetic retention'} for _ in range(101)])
    await db.execute_raw("""UPDATE message_followup_events SET status='completed',
        updated_at=NOW()-INTERVAL '31 days' WHERE message_id IN (
        SELECT id FROM messages WHERE conversation_id=$1 AND role='assistant')""",msg.conversationId)
    assert await f.prune_message_followups()==100
    assert await f.prune_message_followups()==1
    assert await row(db,msg.id) is not None
    assert await db.message.count(where={'conversationId':msg.conversationId,'role':'assistant'})==101


async def test_future_delivery_does_not_ack_uncommitted_registration(followups):
    db,msg,*_=followups
    with pytest.raises(f.MessageFollowupFailed,match='RuntimeError'):
        await f.run_message_followup({'message_id':msg.id,'generation':1})
    for payload in [{},{'message_id':1,'generation':0},{'message_id':msg.id,'generation':True},
                    {'message_id':msg.id,'generation':-1},{'message_id':'x'*201,'generation':0}]:
        with pytest.raises(ValueError):await f.run_message_followup(payload)


async def test_concurrent_dispatch_and_partial_consumers_are_idempotent(followups):
    db,msg,*_=followups
    assert sum(await asyncio.gather(*(f.dispatch_message_followups(limit=1,message_id=msg.id) for _ in range(3))))==1
    await asyncio.gather(*(f.run_message_followup({'message_id':msg.id,'generation':0}) for _ in range(3)))
    assert (await row(db,msg.id))['status']=='completed'
    assert len(await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id))==1
    assert len(await db.query_raw('SELECT id FROM achievement_unlocks WHERE user_id=$1',followups[2][0]))==1


async def test_retry_endpoint_permissions_conflict_and_error(followups,monkeypatch,redis):
    db,msg,*_=followups;app=FastAPI();app.include_router(router)
    path=f'/admin-api/runtime-jobs/message-followups/{msg.id}/retry'
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://test') as c:
        assert (await c.post(path)).status_code==401
        assert (await c.post(path,headers={'Authorization':'Bearer '+create_jwt('ordinary',role='user')})).status_code==403
        h={'Authorization':'Bearer '+create_jwt('admin',role='admin')}
        assert (await c.post(path,headers=h)).status_code==409
        assert (await c.post('/admin-api/runtime-jobs/message-followups/missing/retry',headers=h)).status_code==404
        await db.execute_raw("UPDATE message_followup_events SET status='failed' WHERE message_id=$1",msg.id)
        assert (await c.post(path,headers=h)).status_code==200
        monkeypatch.setattr(f,'retry_message_followup',AsyncMock(side_effect=RuntimeError('PRIVATE')))
        response=await c.post(path,headers=h)
        assert response.status_code==503 and 'PRIVATE' not in response.text


@pytest.mark.parametrize('metadata',[{'gift':True},{'red_packet':True},{'component_card':{'type':'gift'}}])
async def test_offering_route_keeps_its_own_achievement_contract(followups,metadata):
    db,msg,*_=followups
    await db.message.delete(where={'id':msg.id})
    msg=await db.message.create(data={'conversationId':msg.conversationId,'role':'user','content':'offering','metadata':Json(metadata)})
    assert (await process_one(db,msg.id))['outcome']=='offering_managed_separately'
    assert not await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id)


async def test_source_read_and_failure_receipt_errors_are_redacted(followups,monkeypatch):
    db,msg,*_=followups
    monkeypatch.setattr(f,'db',type('Broken',(),{
        'query_raw':AsyncMock(side_effect=RuntimeError('PRIVATE_SQL')),
        'execute_raw':AsyncMock(side_effect=RuntimeError('PRIVATE_RECEIPT'))})())
    with pytest.raises(f.MessageFollowupFailed,match='^RuntimeError$'):
        await f.run_message_followup({'message_id':msg.id,'generation':0})


@pytest.mark.parametrize('phase',['after_effect','before_ack'])
async def test_consumer_process_crash_recovers_without_repeating_effect(followups,redis,phase,tmp_path):
    import json,os,signal,sys
    from pathlib import Path
    db,msg,*_=followups
    await f.dispatch_message_followups(limit=1,message_id=msg.id)
    registered=await row(db,msg.id);jid=registered['job_id']
    fields=('_READY_KEY','_DELAYED_KEY','_RUNNING_KEY','_DLQ_KEY','_SUCCEEDED_KEY','_JOB_KEY_PREFIX','_IDEMP_KEY_PREFIX')
    # Prisma's child engine may inherit pipes beyond the killed worker's life.
    # Observe the worker exit itself; log files avoid waiting for descendant EOF.
    log_path=tmp_path/'crash-worker.log'
    with log_path.open('wb') as log:
        process=await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).parent/'fixtures/message_followup_crash_worker.py'),jid,phase,
            json.dumps({k:getattr(q,k) for k in fields}),stdout=log,stderr=log,
            start_new_session=True)
        try:
            await asyncio.wait_for(process.wait(),timeout=20)
            assert process.returncode==73,log_path.read_text()[-2000:]
        finally:
            # os._exit intentionally bypasses Prisma.disconnect. Its query engine
            # belongs to this dedicated group and must not leak into later tests.
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.returncode is None:
                await asyncio.wait_for(process.wait(),timeout=5)
    assert (await q.inspect_runtime_job(jid))['status']=='running'
    assert len(await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id))==1
    assert (await row(db,msg.id))['status']==('queued' if phase=='after_effect' else 'completed')
    await redis.delete(q._job_lock_key(jid))
    await redis.hset(q._job_key(jid),'lease_expires_at','0')
    await redis.zadd(q._RUNNING_KEY,{jid:0})
    await q._recover_stale_running_jobs(1)
    await q.process_runtime_jobs(max_jobs=1)
    assert (await row(db,msg.id))['status']=='completed'
    assert (await q.inspect_runtime_job(jid))['status']=='succeeded'
    assert len(await db.query_raw('SELECT id FROM achievement_events WHERE source_id=$1',msg.id))==1


async def test_failed_filter_can_find_old_failure_beyond_recent_sample(followups):
    db,msg,*_=followups
    await db.execute_raw("UPDATE message_followup_events SET status='failed',created_at=NOW()-INTERVAL '1 day' WHERE message_id=$1",msg.id)
    for _ in range(3):await db.message.create(data={'conversationId':msg.conversationId,'role':'assistant','content':'new synthetic'})
    assert msg.id not in [r['message_id'] for r in (await f.inspect_message_followups(limit=1))['items']]
    result=await f.inspect_message_followups(limit=1,status='failed')
    assert result['items'][0]['message_id']==msg.id and result['status_filter']=='failed'
    with pytest.raises(ValueError):await f.inspect_message_followups(status='invalid')
