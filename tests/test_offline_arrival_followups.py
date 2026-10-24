"""Real PostgreSQL/Redis arrival latency, scope fencing and crash recovery."""
import asyncio
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException, FastAPI
from httpx import ASGITransport, AsyncClient

from app.services.offline import arrival_followups as f, repository as repo, activity_service, chat_emit
from app.services.proactive import emit
from app.services.offline import activity_companion
from app.services.runtime import job_queue as q
from app.services.notifications import service as notifications
from app.services.speech_output import policy
from tests.test_memory_evidence_postgres import origins
from tests.test_runtime_job_queue import redis


@pytest.fixture
async def arrival(origins, redis, monkeypatch):
    db, uid, wid, _, _, _, _, users, agents, spaces, convs = origins
    for module in (f, repo, chat_emit, emit):
        monkeypatch.setattr(module, 'db', db)
    monkeypatch.setattr(policy, 'should_generate_voice', AsyncMock(return_value=False))
    monkeypatch.setattr(activity_service, '_guard_event', AsyncMock())
    monkeypatch.setattr(activity_service.shooting_conditions, 'generate_items_for_activity', AsyncMock())
    monkeypatch.setattr(activity_service, 'remember_user_event', lambda **kw: None)
    monkeypatch.setattr(activity_service, '_arrival_guide_text', AsyncMock(return_value='已经到啦，慢慢逛。'))
    monkeypatch.setattr(activity_companion, 'start_opening_segment', AsyncMock())
    monkeypatch.setattr(notifications, 'notify_agent_message_created', AsyncMock())
    monkeypatch.setattr(chat_emit, '_notify_takeover_transcript', AsyncMock())
    monkeypatch.setattr(emit.manager, 'send_to_workspace', AsyncMock())
    q.register_job_handler(f.JOB_TYPE, f.run_arrival_followup)
    aid = uuid4().hex
    await db.execute_raw("""INSERT INTO offline_activity_recommendations
        (id,user_id,agent_id,workspace_id,conversation_id,title,status,updated_at,place_lat,place_lng,discovery_metadata)
        VALUES($1,$2,$3,$4,$5,'合成活动','accepted',NOW(),31.2,121.4,'{"coordinate_system":"wgs84"}')""",
        aid, uid, agents[0], wid, convs[0])
    try:
        yield db, aid, uid, wid, agents, spaces, convs
    finally:
        await db.execute_raw('DELETE FROM proactive_chat_logs WHERE agent_id=ANY($1::text[])', agents)
        await db.execute_raw('DELETE FROM offline_activity_recommendations WHERE id=$1', aid)


async def receipt(db, aid):
    rows = await db.query_raw('SELECT * FROM offline_arrival_followups WHERE activity_id=$1', aid)
    return rows[0] if rows else None


async def arrive(aid, uid):
    return await activity_service.arrive_activity(uid, aid, lat=31.2, lng=121.4, accuracy_m=10,
        observed_at=datetime.now(timezone.utc))


async def process(db, aid):
    await db.execute_raw('UPDATE offline_arrival_followups SET next_dispatch_at=NOW() WHERE activity_id=$1', aid)
    await f.dispatch_arrival_followups(limit=1, activity_id=aid)
    await q.process_runtime_jobs(max_jobs=1)
    return await receipt(db, aid)


async def test_arrival_returns_before_llm_and_registers_once(arrival):
    db, aid, uid, *_ = arrival
    updated = await asyncio.wait_for(arrive(aid, uid), 2)
    assert updated.reached
    activity_service._arrival_guide_text.assert_not_awaited()
    assert (await receipt(db, aid))['status'] == 'pending_dispatch'
    await arrive(aid, uid)
    assert (await receipt(db, aid))['generation'] == 0
    result = await process(db, aid)
    assert result['status'] == 'completed' and result['outcome'] == 'delivered'
    for role in ['user', 'assistant']:
        assert await f.load_delivery(aid, role)
    await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    activity_service._arrival_guide_text.assert_awaited_once()


async def test_receipt_rolls_back_with_arrival(arrival,monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    db, aid, uid, *_ = arrival
    @asynccontextmanager
    async def transaction():
        async with db.tx() as tx:
            async def execute(query,*args):
                result=await tx.execute_raw(query,*args)
                if 'INSERT INTO offline_arrival_followups' in query:
                    assert await tx.query_raw('SELECT 1 FROM offline_arrival_followups WHERE activity_id=$1',aid)
                    raise RuntimeError('lost commit')
                return result
            yield SimpleNamespace(query_raw=tx.query_raw,execute_raw=execute)
    monkeypatch.setattr(repo,'db',SimpleNamespace(tx=transaction))
    with pytest.raises(RuntimeError):
        await repo.mark_arrived(aid,uid,lat=31.2,lng=121.4,verified=True)
    assert await receipt(db, aid) is None
    assert not (await db.query_raw('SELECT reached FROM offline_activity_recommendations WHERE id=$1', aid))[0]['reached']


async def test_old_synchronous_arrival_does_not_register_duplicate_delivery(arrival):
    db,aid,*_=arrival
    await db.execute_raw("UPDATE offline_activity_recommendations SET reached=TRUE,arrival_confirmed_at=NOW() WHERE id=$1",aid)
    assert await receipt(db,aid) is None


@pytest.mark.parametrize('stage', ['card', 'guide', 'notify', 'opening'])
async def test_partial_failure_reuses_persisted_messages(arrival, monkeypatch, stage):
    db, aid, uid, *_ = arrival
    await arrive(aid, uid)
    if stage in {'card', 'guide'}:
        async def send(workspace, event, payload):
            if event == ('message' if stage == 'card' else 'proactive'):
                raise RuntimeError('PRIVATE_LOCATION_AND_SQL')
        monkeypatch.setattr(emit.manager, 'send_to_workspace', AsyncMock(side_effect=send))
    elif stage == 'notify':
        notifications.notify_agent_message_created.side_effect = RuntimeError('PRIVATE_TEXT')
    else:
        activity_companion.start_opening_segment.side_effect = RuntimeError('PRIVATE_TEXT')
    first = await process(db, aid)
    assert first['status'] == 'queued' and first['last_error'] == 'RuntimeError'
    assert 'PRIVATE' not in str(await q.inspect_runtime_job(first['job_id']))
    old_card = await f.load_delivery(aid, 'user')
    old_guide = await f.load_delivery(aid, 'assistant')
    monkeypatch.setattr(emit.manager, 'send_to_workspace', AsyncMock())
    notifications.notify_agent_message_created.side_effect = None
    activity_companion.start_opening_segment.side_effect = None
    await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    assert (await receipt(db, aid))['status'] == 'completed'
    assert (await f.load_delivery(aid, 'user'))['id'] == old_card['id']
    if old_guide:
        assert (await f.load_delivery(aid, 'assistant')) == old_guide
    assert len(await db.query_raw("SELECT id FROM messages WHERE metadata->>'source_id'=$1", aid)) == 2
    activity_service._arrival_guide_text.assert_awaited_once()


async def test_lost_queue_recovers_and_old_generation_cannot_write(arrival, redis):
    db, aid, uid, *_ = arrival
    await arrive(aid, uid)
    await f.dispatch_arrival_followups(limit=1, activity_id=aid)
    first = await receipt(db, aid)
    await redis.delete(q._job_key(first['job_id']))
    await db.execute_raw('UPDATE offline_arrival_followups SET next_dispatch_at=NOW() WHERE activity_id=$1', aid)
    await f.dispatch_arrival_followups(limit=1, activity_id=aid)
    assert (await receipt(db, aid))['generation'] == 1
    await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    assert await f.load_delivery(aid, 'user') is None
    await f.run_arrival_followup({'activity_id': aid, 'generation': 1})
    assert (await receipt(db, aid))['status'] == 'completed'


@pytest.mark.parametrize('mutation', ['archive', 'deleted', 'owner', 'generation', 'expired', 'user_archived'])
async def test_change_during_llm_is_fenced_at_commit(arrival, monkeypatch, mutation):
    db, aid, uid, wid, agents, spaces, convs = arrival
    await arrive(aid, uid)
    async def generate(*args):
        if mutation == 'archive':
            await db.execute_raw("UPDATE offline_activity_recommendations SET status='completed' WHERE id=$1", aid)
        elif mutation == 'deleted':
            await db.conversation.update(where={'id': convs[0]}, data={'isDeleted': True})
        elif mutation == 'owner':
            await db.execute_raw('UPDATE offline_activity_recommendations SET workspace_id=$2,agent_id=$3 WHERE id=$1', aid, spaces[2], agents[2])
        elif mutation == 'expired':
            await db.execute_raw("UPDATE offline_activity_recommendations SET auto_archive_at=NOW()-INTERVAL '1 second' WHERE id=$1", aid)
        elif mutation == 'user_archived':
            await db.user.update(where={'id':uid},data={'status':'archived'})
        else:
            await db.execute_raw('UPDATE offline_arrival_followups SET generation=1 WHERE activity_id=$1', aid)
        return 'must not send'
    monkeypatch.setattr(activity_service, '_arrival_guide_text', generate)
    await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    assert await f.load_delivery(aid, 'assistant') is None
    assert (await receipt(db, aid))['status'] == ('pending_dispatch' if mutation == 'generation' else 'cancelled')


async def test_simultaneous_workers_insert_one_message_per_role(arrival):
    db, aid, uid, *_ = arrival
    await arrive(aid, uid)
    payload = {'activity_id': aid, 'generation': 0}
    await asyncio.gather(f.run_arrival_followup(payload), f.run_arrival_followup(payload))
    assert len(await db.query_raw("SELECT id FROM messages WHERE metadata->>'source_id'=$1", aid)) == 2
    assert (await receipt(db, aid))['status'] == 'completed'


@pytest.mark.parametrize('changes,reason', [({'accuracy_m': 101}, 'low_accuracy'),
    ({'lat': 32.2}, 'too_far'), ({'observed_at': datetime(2020,1,1,tzinfo=timezone.utc)}, 'stale_location')])
async def test_invalid_location_never_registers_arrival(arrival, changes, reason):
    db, aid, uid, *_ = arrival
    values = dict(lat=31.2,lng=121.4,accuracy_m=10,observed_at=datetime.now(timezone.utc))
    values.update(changes)
    with pytest.raises(HTTPException) as exc:
        await activity_service.arrive_activity(uid, aid, **values)
    assert exc.value.detail['reason'] == reason
    assert await receipt(db, aid) is None


def _mentions_fixture_coordinates(value) -> bool:
    """The fixture point is (31.2, 121.4). A substring test over the whole
    payload also hit timestamps such as 10:36:31.231053 (CI 2026-10-24)."""
    if isinstance(value, dict):
        return any(_mentions_fixture_coordinates(k) or _mentions_fixture_coordinates(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_mentions_fixture_coordinates(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value in (31.2, 121.4)
    return isinstance(value, str) and bool(re.search(r"(?<![\d.])(?:31\.2|121\.4)(?!\d)", value))


async def test_admin_diagnostics_are_bounded_redacted_and_recoverable(arrival):
    from app.api.admin.runtime_jobs import router
    from app.services.auth import create_jwt
    db, aid, uid, *_ = arrival
    await arrive(aid, uid)
    await db.execute_raw("UPDATE offline_arrival_followups SET status='failed',last_error='TimeoutError' WHERE activity_id=$1", aid)
    app = FastAPI();app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://test') as client:
        path = '/admin-api/runtime-jobs/arrival-followups'
        assert (await client.get(path)).status_code in {401,403}
        assert (await client.get(path,headers={'Authorization':'Bearer '+create_jwt('user',role='user')})).status_code == 403
        headers={'Authorization':'Bearer '+create_jwt('admin',role='admin')}
        result=await client.get(path,headers=headers)
        assert result.status_code==200
        data=result.json();assert data['denominator']=='recent_sample_only'
        assert 'scope' not in str(data) and '合成' not in str(data)
        assert not _mentions_fixture_coordinates(data)
        assert (await client.get(path+'?limit=51',headers=headers)).status_code==422
        assert (await client.post(path+'/'+aid+'/retry',headers=headers)).status_code==200
        assert (await client.post(path+'/'+aid+'/retry',headers=headers)).status_code==409


async def test_http_arrival_preserves_payload_and_returns_before_delivery(arrival):
    from app.api.public.offline import router, require_user
    db, aid, uid, *_ = arrival
    app=FastAPI();app.include_router(router)
    app.dependency_overrides[require_user]=lambda:{'sub':uid}
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://test') as client:
        response=await asyncio.wait_for(client.post(f'/offline/activities/{aid}/arrive',json={
            'lat':31.2,'lng':121.4,'accuracy_m':10,'observed_at':datetime.now(timezone.utc).isoformat(),
            'manual_confirmation':False}),2)
    assert response.status_code==200 and response.json()['reached']
    assert response.json()['arrival_verified']
    activity_service._arrival_guide_text.assert_not_awaited()
    await process(db,aid)
    assert await f.load_delivery(aid,'assistant')


async def test_manual_confirmation_preserves_unverified_contract(arrival):
    db,aid,uid,*_=arrival
    response=await activity_service.arrive_activity(uid,aid,manual_confirmation=True)
    assert response.reached and not response.arrival_verified
    rows=await db.query_raw('SELECT arrival_lat,arrival_lng FROM offline_activity_recommendations WHERE id=$1',aid)
    assert rows[0]['arrival_lat'] is None and rows[0]['arrival_lng'] is None
    assert (await receipt(db,aid))['status']=='pending_dispatch'


@pytest.mark.parametrize('missing_asset',[False,True])
async def test_voice_is_atomically_bound_replayed_or_falls_back_to_text(arrival,monkeypatch,missing_asset):
    from types import SimpleNamespace
    from app.services.speech_output import delivery
    from app.services.chat_media import repo as media
    from app.services.chat_media.prompt import attachment_to_metadata
    db,aid,uid,wid,agents,spaces,convs=arrival
    monkeypatch.setattr(media,'db',db)
    asset=await media.create_generated_audio_attachment(user_id=uid,conversation_id=convs[0],
        storage_key='synthetic-arrival.wav',url='/synthetic-arrival.wav',mime='audio/wav',size=42,
        duration_seconds=1,transcript='已经到啦',request_id=None)
    metadata={**attachment_to_metadata(asset),'generated_by':'assistant_tts'}
    if missing_asset:metadata['id']='missing-owned-test-asset'
    prepared=SimpleNamespace(speech=SimpleNamespace(request_id=None),attachment=asset,metadata=metadata)
    monkeypatch.setattr(policy,'should_generate_voice',AsyncMock(return_value=True))
    monkeypatch.setattr(delivery,'prepare_voice_output',AsyncMock(return_value=prepared))
    monkeypatch.setattr(delivery,'discard_prepared_voice_output',AsyncMock())
    monkeypatch.setattr(delivery,'bind_prepared_voice_output',AsyncMock(side_effect=AssertionError('must bind in SQL transaction')))
    await arrive(aid,uid)
    activity_companion.start_opening_segment.side_effect=RuntimeError('lost acknowledgement')
    await process(db,aid)
    stored=await f.load_delivery(aid,'assistant')
    bound=await db.query_raw('SELECT message_id FROM chat_message_attachments WHERE id=$1',asset.id)
    if missing_asset:
        assert 'display_mode' not in stored['metadata'] and bound[0]['message_id'] is None
        delivery.discard_prepared_voice_output.assert_awaited_once()
    else:
        assert stored['metadata']['display_mode']=='voice' and bound[0]['message_id']==stored['id']
    activity_companion.start_opening_segment.side_effect=None
    await f.run_arrival_followup({'activity_id':aid,'generation':0})
    assert (await f.load_delivery(aid,'assistant'))==stored
    delivery.prepare_voice_output.assert_awaited_once()
    payloads=[call.args[2] for call in emit.manager.send_to_workspace.await_args_list if call.args[1]=='proactive']
    assert {p['assistant_message_id'] for p in payloads}=={stored['id']}
    if not missing_asset:assert all(p['attachments'][0]['id']==asset.id for p in payloads)


@pytest.mark.parametrize('payload', [
    {}, {'activity_id': '', 'generation': 0},
    {'activity_id': 'x' * 201, 'generation': 0},
    {'activity_id': 'x', 'generation': True},
    {'activity_id': 'x', 'generation': -1},
])
async def test_invalid_dispatch_payload_never_touches_storage(monkeypatch, payload):
    from types import SimpleNamespace
    query = AsyncMock(side_effect=AssertionError('invalid dispatch must not query storage'))
    monkeypatch.setattr(f, 'db', SimpleNamespace(query_raw=query))
    with pytest.raises(ValueError, match='invalid_arrival_followup_payload'):
        await f.run_arrival_followup(payload)
    query.assert_not_awaited()


async def test_uncommitted_dispatch_generation_does_not_deliver(arrival):
    db, aid, uid, *_ = arrival
    await arrive(aid, uid)
    with pytest.raises(f.ArrivalFollowupFailed, match='RuntimeError'):
        await f.run_arrival_followup({'activity_id': aid, 'generation': 1})
    assert (await receipt(db, aid))['generation'] == 0
    assert await f.load_delivery(aid, 'user') is None
    await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    assert (await receipt(db, aid))['status'] == 'completed'


async def test_already_archived_activity_cancels_before_model_or_delivery(arrival):
    db, aid, uid, *_ = arrival
    await arrive(aid, uid)
    await db.execute_raw("UPDATE offline_activity_recommendations SET status='completed' WHERE id=$1", aid)
    await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    assert (await receipt(db, aid))['status'] == 'cancelled'
    activity_service._arrival_guide_text.assert_not_awaited()
    assert await f.load_delivery(aid, 'user') is None


async def test_unavailable_context_records_retryable_redacted_failure(arrival, monkeypatch):
    db, aid, uid, *_ = arrival
    await arrive(aid, uid)
    original = repo.resolve_user_context
    monkeypatch.setattr(repo, 'resolve_user_context', AsyncMock(return_value=None))
    with pytest.raises(f.ArrivalFollowupFailed, match='RuntimeError'):
        await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    assert (await receipt(db, aid))['last_error'] == 'RuntimeError'
    assert await f.load_delivery(aid, 'user') is None
    monkeypatch.setattr(repo, 'resolve_user_context', original)
    await f.run_arrival_followup({'activity_id': aid, 'generation': 0})
    assert (await receipt(db, aid))['status'] == 'completed'


@pytest.mark.parametrize('fault', ['conversation', 'collision', 'empty_voice', 'duplicate_voice'])
async def test_delivery_rejects_mismatched_scope_and_invalid_assets_atomically(arrival, fault):
    db, aid, uid, wid, agents, spaces, convs = arrival
    await arrive(aid, uid)
    conversation_id, metadata = convs[0], {}
    if fault == 'conversation':
        conversation_id = convs[1]
    elif fault == 'collision':
        await db.execute_raw("""INSERT INTO messages(id,conversation_id,role,content)
            VALUES($1,$2,'user','existing row must survive')""", f.delivery_id(aid, 'assistant'), convs[1])
    else:
        metadata = {'display_mode': 'voice', 'attachments': [] if fault == 'empty_voice'
                    else [{'id': 'same'}, {'id': 'same'}]}
    error = f.ArrivalVoiceBindingFailed if fault.endswith('voice') else RuntimeError
    with pytest.raises(error):
        await f.persist_delivery(activity_id=aid, generation=0, role='assistant',
            conversation_id=conversation_id, content='must not commit', metadata=metadata)
    rows = await db.query_raw('SELECT content FROM messages WHERE id=$1', f.delivery_id(aid, 'assistant'))
    assert rows == ([{'content': 'existing row must survive'}] if fault == 'collision' else [])
    assert (await receipt(db, aid))['status'] == 'pending_dispatch'


async def test_admin_arrival_unavailable_is_not_shown_as_empty_or_success(arrival, monkeypatch):
    from app.api.admin.runtime_jobs import router
    from app.services.auth import create_jwt
    app = FastAPI(); app.include_router(router)
    headers = {'Authorization': 'Bearer ' + create_jwt('admin', role='admin')}
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        path = '/admin-api/runtime-jobs/arrival-followups'
        monkeypatch.setattr(f, 'inspect_arrival_followups', AsyncMock(side_effect=RuntimeError('PRIVATE_SQL')))
        unavailable = await client.get(path, headers=headers)
        assert unavailable.status_code == 503 and 'PRIVATE' not in unavailable.text
        retry = AsyncMock(side_effect=RuntimeError('PRIVATE_SQL'))
        monkeypatch.setattr(f, 'retry_arrival_followup', retry)
        unavailable = await client.post(path + '/missing/retry', headers=headers)
        assert unavailable.status_code == 503 and 'PRIVATE' not in unavailable.text
        retry.side_effect = None; retry.return_value = False
        missing = await client.post(path + '/missing/retry', headers=headers)
        assert missing.status_code == 404 and missing.json()['detail'] == 'arrival_followup_not_found'
