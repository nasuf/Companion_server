"""Real Redis retention, missing-data semantics and authenticated API projection."""
import asyncio
from datetime import UTC, datetime, timedelta
import json
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.api.admin.runtime_jobs import router
from app.services.auth import create_jwt
from app.services.memory.recording import diagnostics as d
from app.services.memory.recording.outcomes import MemorySideOutcome
from tests.test_runtime_job_queue import redis


@pytest.fixture
async def observations(redis,monkeypatch):
    prefix='test:memory-processing:'+uuid4().hex
    monkeypatch.setattr(d,'_INDEX',prefix+':index');monkeypatch.setattr(d,'_DATA',prefix+':data')
    monkeypatch.setattr(d,'get_redis',AsyncMock(return_value=redis))
    try:yield redis
    finally:await redis.delete(d._INDEX,d._DATA)


def record(i,at=None,status='completed'):
    now=(at or datetime.now(UTC)).isoformat()
    return {'id':str(i),'conversation_id':'synthetic-conversation','started_at':now,'updated_at':now,
            'status':status,'sides':[{'side':'user','status':'completed','reason':'processed','memory_count':0,'progress':'advanced'}]}


async def test_real_retention_is_bounded_and_expires_without_zero_success(observations):
    r=await d.inspect_memory_processing();assert r['collection_status']=='no_retained_records' and not r['records']
    for i in range(225):await d._write(record(i))
    assert await observations.zcard(d._INDEX)==200 and await observations.hlen(d._DATA)==200
    assert 0<await observations.ttl(d._INDEX)<=7*86400 and 0<await observations.ttl(d._DATA)<=7*86400
    r=await d.inspect_memory_processing(limit=50)
    assert len(r['records'])==50 and r['sample_count']==50 and r['replay_supported'] is False
    await observations.delete(d._DATA)
    r=await d.inspect_memory_processing();assert r['collection_status']=='partial' and r['missing_records']==20
    await observations.delete(d._INDEX)
    assert (await d.inspect_memory_processing())['collection_status']=='no_retained_records'


async def test_expired_rows_are_pruned_by_age_and_unconfirmed_not_failure(observations):
    await d._write(record('expired',datetime.now(UTC)-timedelta(days=8)))
    await d._write(record('running',datetime.now(UTC)-timedelta(minutes=16),'running'))
    await d._write(record('live',status='running'))
    result=await d.inspect_memory_processing()
    assert {r['id']:r['status'] for r in result['records']}=={'live':'running','running':'unconfirmed'}
    assert await observations.hget(d._DATA,'expired') is None


async def test_started_then_cancelled_is_visible_without_error_text(observations):
    async with d.observe_memory_run('conv') as sides:
        state=await d.inspect_memory_processing();assert state['records'][0]['status']=='running'
        sides.extend([MemorySideOutcome('user','completed','processed',0,'advanced'),
                      MemorySideOutcome('ai','disabled','prompt_disabled')])
    result=await d.inspect_memory_processing();assert result['records'][0]['status']=='incomplete'
    with pytest.raises(asyncio.CancelledError):
        async with d.observe_memory_run('cancelled'):raise asyncio.CancelledError('PRIVATE_CHAT')
    result=await d.inspect_memory_processing()
    assert result['records'][0]['status']=='cancelled' and 'PRIVATE' not in str(result)


async def test_exception_and_future_fields_cannot_leak_into_admin_projection(observations):
    with pytest.raises(RuntimeError):
        async with d.observe_memory_run('conv'):raise RuntimeError('PRIVATE_SECRET')
    value=record('allowlist');value['secret']='PRIVATE_SECRET';value['sides'][0]['reason']='PRIVATE_SECRET'
    value['sides'][0]['raw_prompt']='PRIVATE_CHAT'
    await d._write(value)
    result=await d.inspect_memory_processing()
    assert 'PRIVATE' not in json.dumps(result)
    assert result['records'][0]['sides'][0]['reason']=='processing_failed'


async def test_collector_failure_does_not_interrupt_processing(monkeypatch):
    monkeypatch.setattr(d,'get_redis',AsyncMock(side_effect=RuntimeError('PRIVATE')))
    completed=[]
    async with d.observe_memory_run('conv') as sides:
        completed.append(True);sides.append(MemorySideOutcome('user','completed','processed',1))
    assert completed==[True]
    with pytest.raises(RuntimeError):await d.inspect_memory_processing()


@pytest.mark.parametrize('limit',[0,51])
async def test_query_limits_fail_before_redis(limit,monkeypatch):
    client=AsyncMock();monkeypatch.setattr(d,'get_redis',client)
    with pytest.raises(ValueError):await d.inspect_memory_processing(limit=limit)
    client.assert_not_awaited()


async def test_api_permissions_limits_and_failure_are_distinct(observations,monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings,'jwt_secret','synthetic-bg03-api-test-secret-long-enough')
    app=FastAPI();app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://test') as client:
        url='/admin-api/runtime-jobs/memory-processing'
        assert (await client.get(url)).status_code==401
        user={'Authorization':'Bearer '+create_jwt('user',role='user')}
        assert (await client.get(url,headers=user)).status_code==403
        admin={'Authorization':'Bearer '+create_jwt('admin',role='admin')}
        assert (await client.get(url+'?limit=51',headers=admin)).status_code==422
        assert (await client.get(url+'?limit=0',headers=admin)).status_code==422
        res=await client.get(url,headers=admin);assert res.status_code==200 and res.json()['collection_status']=='no_retained_records'
        monkeypatch.setattr(d,'get_redis',AsyncMock(side_effect=RuntimeError('PRIVATE_DATABASE_URL')))
        res=await client.get(url,headers=admin);assert res.status_code==503 and 'PRIVATE' not in res.text


@pytest.mark.parametrize('change',[
    {'sides':[{}, {}, {}]}, {'sides':[{'side':'foreign'}]}, {'id':'x'*201},
    {'started_at':'no-time'}, {'updated_at':'2026-10-18'}, {'sides':'invalid'},
])
async def test_corrupt_rows_are_partial_without_hiding_other_records(observations,change):
    await d._write(record('valid'))
    value=record('invalid');value.update(change)
    await observations.hset(d._DATA,'invalid',json.dumps(value))
    await observations.zadd(d._INDEX,{'invalid':datetime.now(UTC).timestamp()})
    result=await d.inspect_memory_processing()
    assert result['collection_status']=='partial' and result['missing_records']==1
    assert [r['id'] for r in result['records']]==['valid']


async def test_unknown_enum_values_and_counts_do_not_escape_projection(observations):
    value=record('future',status='PRIVATE_FUTURE')
    value['sides'][0].update(status='PRIVATE_STATUS',progress='PRIVATE_PROGRESS',memory_count=-1)
    await d._write(value)
    result=await d.inspect_memory_processing()
    assert 'PRIVATE' not in str(result)
    side=result['records'][0]['sides'][0]
    assert side['status']=='unknown' and side['progress']=='unknown' and side['memory_count'] is None
