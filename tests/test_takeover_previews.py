"""Complete candidates stay private until explicit selection; publishing is scoped and deduplicated."""
import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
import httpx
import pytest
from tests.test_takeover_activity_generation import pipeline, app, SUBJECT, CTX, CARD
from tests.test_proactive_activity_e2e import flow  # noqa: F401
from tests.test_offline_quality_e2e import journey  # noqa: F401
from app.services.admin_takeover import activities
from app.services.offline import activity_generation as gen


class PreviewCache:
    def __init__(self): self.items = {}
    async def get(self, key): return self.items.get(key)
    async def set(self, key, value, *, ex, nx=False):
        if nx and key in self.items: return False
        self.items[key] = value
        return True
    async def delete(self, key): self.items.pop(key, None)
    async def eval(self, script, count, key, token):
        if self.items.get(key) == token:
            self.items.pop(key, None)
            return 1
        return 0


@pytest.fixture
def previews(pipeline, monkeypatch):
    cards = [{**deepcopy(CARD), 'title': f'活动{i}', 'location_name': f'地点{i}'} for i in range(5)]
    generate = AsyncMock(return_value=cards)
    cache = PreviewCache()
    monkeypatch.setattr(activities, 'generate_activity_preview_cards', generate)
    monkeypatch.setattr(activities, 'get_redis', AsyncMock(return_value=cache))
    return pipeline, generate, cache


async def load(client):
    response = await client.post('/admin-api/takeovers/case/activity-previews')
    assert response.status_code == 200, response.text
    return response.json()['items']


async def test_five_complete_candidates_never_publish_until_explicit_choice(previews):
    p, generate, cache = previews
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        items = await load(client)
        assert len(items) == 5 and len({x['preview_id'] for x in items}) == 5
        for item in items:
            assert item['description'] == CARD['description']
            assert item['recommendation_message'] == CARD['discovery_metadata']['recommendation_message']
            assert item['image_urls'] == CARD['image_urls']
            assert 'discovery_metadata' not in item and 'user_relevance' not in item
        assert p['events'] == []
        p['create'].assert_not_awaited()
        chosen = items[2]
        url = f"/admin-api/takeovers/case/activity-previews/{chosen['preview_id']}/publish"
        response = await client.post(url)
        assert response.status_code == 200, response.text
        assert p['events'] == ['invite', 'persist', 'text', 'card', 'schedule']
        assert p['stored']['title'] == chosen['title']
        assert p['stored']['description'] == chosen['description']
        assert p['stored']['discovery_metadata'] == CARD['discovery_metadata']
        assert p['stored']['search_sources'] == CARD['search_sources']
        assert p['emit'].await_args.kwargs['extra_metadata']['human_takeover'] is True
        assert p['emit_card'].await_args.kwargs['extra_metadata']['human_takeover'] is True
        assert (await client.post(url)).json() == response.json()
        assert p['create'].await_count == 1 and generate.await_count == 1
        p['generated'].assert_not_awaited()  # Selection may not silently generate another destination.


async def test_refresh_cancel_and_missing_candidate_have_no_user_side_effects(previews):
    p, generate, _ = previews
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        old = await load(client)
        new = await load(client)
        assert {x['preview_id'] for x in old}.isdisjoint(x['preview_id'] for x in new)
        assert (await client.post(f"/admin-api/takeovers/case/activity-previews/{old[0]['preview_id']}/publish")).status_code == 404
    assert p['events'] == [] and generate.await_count == 2


async def test_ownership_loss_during_preview_does_not_cache_or_send(previews):
    p, _, cache = previews
    p['owned'].side_effect = [SUBJECT, SUBJECT, None]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        assert (await client.post('/admin-api/takeovers/case/activity-previews')).status_code == 403
    assert not cache.items and not p['events']


async def test_duplicate_publish_in_flight_is_rejected_without_second_message(previews, monkeypatch):
    p, _, _ = previews
    entered, gate = asyncio.Event(), asyncio.Event()
    async def invite(**kwargs):
        entered.set()
        await gate.wait()
        return '邀请'
    monkeypatch.setattr(activities.activity_service, 'generate_activity_invite_message', invite)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        items = await load(client)
        url = f"/admin-api/takeovers/case/activity-previews/{items[0]['preview_id']}/publish"
        first = asyncio.create_task(client.post(url))
        await entered.wait()
        assert (await client.post(url)).status_code == 409
        gate.set()
        assert (await first).status_code == 200
    assert p['create'].await_count == 1


async def test_no_publication_result_before_any_write_can_retry(previews, monkeypatch):
    p, _, _ = previews
    publish = AsyncMock(side_effect=[None, {'id': 'recovered', 'title': '已恢复'}])
    monkeypatch.setattr(activities.activity_service, 'create_recommendation_for_user', publish)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        items = await load(client)
        url = f"/admin-api/takeovers/case/activity-previews/{items[0]['preview_id']}/publish"
        assert (await client.post(url)).status_code == 503
        assert (await client.post(url)).status_code == 200
    assert publish.await_count == 2
    p['create'].assert_not_awaited()


async def test_receipt_cache_failure_after_publication_keeps_duplicate_guard(previews, monkeypatch):
    p, _, cache = previews
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        items = await load(client)
        original = cache.set
        async def fail_result(key, value, **kwargs):
            if ':published:' in key:
                raise ConnectionError('receipt unavailable')
            return await original(key, value, **kwargs)
        monkeypatch.setattr(cache, 'set', fail_result)
        url = f"/admin-api/takeovers/case/activity-previews/{items[0]['preview_id']}/publish"
        first = await client.post(url)
        assert first.status_code == 503 and '待确认' in first.json()['detail']
        assert (await client.post(url)).status_code == 409
    assert p['create'].await_count == 1


async def test_batch_uses_the_same_complete_native_pipeline_for_each_distinct_place(monkeypatch):
    from app.config import settings
    from app.services.offline import cleversee_discovery
    monkeypatch.setattr(settings, 'offline_search_provider', 'cleversee')
    monkeypatch.setattr(gen.repo, 'list_user_tags', AsyncMock(return_value=['散步']))
    monkeypatch.setattr(gen.repo, 'list_recent_activity_fingerprints', AsyncMock(return_value=[]))
    monkeypatch.setattr(gen, 'evidence_inputs', AsyncMock(return_value=({'memory': [], 'dialogue': [], 'preference': []}, {})))
    candidates = [{'candidate_id': str(i), 'location_name': f'地点{i}'} for i in range(7)]
    discover = AsyncMock(return_value=candidates + [deepcopy(candidates[0])])
    monkeypatch.setattr(cleversee_discovery, 'discover', discover)
    async def native(items, **kwargs):
        assert len(items) == 1
        return {**deepcopy(CARD), **items[0]}
    complete = AsyncMock(side_effect=native)
    monkeypatch.setattr(gen, '_native_card', complete)
    async def attach(card, *_): return card
    monkeypatch.setattr(gen, 'attach_recommendation_message', AsyncMock(side_effect=attach))
    cards = await gen.generate_activity_preview_cards(user_id='user', workspace_id='workspace', city='镇江', source='manual')
    assert len(cards) == 5 and len({c['location_name'] for c in cards}) == 5
    assert discover.await_count == 1 and complete.await_count == 5
    assert all(c['description'] == CARD['description'] for c in cards)


async def test_selected_preview_is_a_full_user_activity_in_real_postgres(journey, monkeypatch):
    from uuid import uuid4
    import app.db
    from app.api.admin import takeovers
    from app.api.public import offline
    from app.services.auth import create_jwt
    from app.services.offline import module_settings
    j = journey
    admin_id, case = uuid4().hex, uuid4().hex
    await j.db.execute_raw("INSERT INTO users (id,username,role,updated_at) VALUES ($1,$1,'admin',CURRENT_TIMESTAMP)", admin_id)
    await j.db.execute_raw("UPDATE users SET location_city='镇江市' WHERE id=$1", j.user)
    await j.db.execute_raw("INSERT INTO human_takeovers (id,user_id,conversation_id,agent_id,workspace_id,status,owner_admin_id,updated_at) "
        "VALUES ($1,$2,$3,$4,$5,'active',$6,CURRENT_TIMESTAMP)", case,j.user,j.conversation,j.agent,j.workspace,admin_id)
    monkeypatch.setattr(app.db, 'db', j.db)
    monkeypatch.setattr(activities, 'db', j.db)
    monkeypatch.setattr(activities, 'get_redis', AsyncMock(return_value=j.redis))
    monkeypatch.setattr(module_settings, 'is_activity_enabled', AsyncMock(return_value=True))
    monkeypatch.setattr(offline, 'is_activity_enabled', AsyncMock(return_value=True))
    cards = [{**deepcopy(CARD), 'title': f'地点{i}', 'location_name': f'地点{i}'} for i in range(5)]
    monkeypatch.setattr(activities, 'generate_activity_preview_cards', AsyncMock(return_value=cards))
    monkeypatch.setattr(activities.activity_service, 'generate_activity_invite_message', AsyncMock(return_value='完整邀请'))
    j.app.include_router(takeovers.router)
    from app.api.jwt_auth import require_admin_jwt
    j.app.dependency_overrides.pop(require_admin_jwt, None)
    headers = {'Authorization': 'Bearer '+create_jwt(admin_id,'admin')}
    path = f'/admin-api/takeovers/{case}/activity-previews'
    response = await j.client.post(path, headers=headers)
    assert response.status_code == 200, response.text
    assert len(response.json()['items']) == 5
    assert await j.db.query_raw('SELECT id FROM offline_activity_recommendations WHERE user_id=$1', j.user) == []
    selected = response.json()['items'][1]
    response = await j.client.post(path+'/'+selected['preview_id']+'/publish', headers=headers)
    assert response.status_code == 200, response.text
    activity_id = response.json()['activity_id']
    assert len(await j.db.query_raw('SELECT id FROM offline_activity_recommendations WHERE user_id=$1', j.user)) == 1
    response = await j.client.get('/offline/activities/'+activity_id)
    assert response.status_code == 200, response.text
    actual = response.json()
    for key in ('title','description','recommendation_message','image_urls','location_name'):
        assert actual[key] == selected[key]
    accepted = await j.client.post('/offline/activities/'+activity_id+'/accept')
    assert accepted.status_code == 200 and accepted.json()['status'] == 'accepted'
    assert not accepted.json()['easter_egg_task']  # Task generation still belongs to arrival.


async def test_uncertain_publication_does_not_silently_retry_or_duplicate(previews):
    p, _, _ = previews
    p['emit'].side_effect = RuntimeError('synthetic transport failure after persistence')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        items = await load(client)
        url = f"/admin-api/takeovers/case/activity-previews/{items[0]['preview_id']}/publish"
        response = await client.post(url)
        assert response.status_code == 503 and '待确认' in response.text
        assert (await client.post(url)).status_code == 409
        assert p['create'].await_count == 1


async def test_pre_persistence_failure_can_retry_safely(previews, monkeypatch):
    p, _, _ = previews
    monkeypatch.setattr(activities.activity_service, 'generate_activity_invite_message',
        AsyncMock(side_effect=[RuntimeError('synthetic unavailable'), '邀请']))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(), raise_app_exceptions=False), base_url='http://test') as client:
        items = await load(client)
        url = f"/admin-api/takeovers/case/activity-previews/{items[0]['preview_id']}/publish"
        assert (await client.post(url)).status_code == 500
        p['create'].assert_not_awaited()
        assert (await client.post(url)).status_code == 200
        assert p['create'].await_count == 1


async def test_overlapping_preview_requests_do_not_generate_or_overwrite_twice(previews):
    p, generate, _ = previews
    entered, gate = asyncio.Event(), asyncio.Event()
    cards = generate.return_value
    async def waiting(**kwargs):
        entered.set()
        await gate.wait()
        return cards
    generate.side_effect = waiting
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        first = asyncio.create_task(client.post('/admin-api/takeovers/case/activity-previews'))
        await entered.wait()
        assert (await client.post('/admin-api/takeovers/case/activity-previews')).status_code == 409
        gate.set()
        assert (await first).status_code == 200
    assert generate.await_count == 1 and p['events'] == []


@pytest.mark.parametrize('failure,status', [
    ('disabled', 403), ('no_conversation', 422), ('wrong_user', 409),
    ('wrong_agent', 409), ('wrong_workspace', 409), ('no_cards', 503),
    ('timed_out', 503), ('lost_generation_lock', 409),
])
async def test_preview_rejects_stale_or_incomplete_generation_without_sending(previews, failure, status):
    p, generate, cache = previews
    if failure == 'disabled': p['enabled'].return_value = False
    if failure == 'no_conversation': p['context'].return_value = {**CTX, 'conversation_id': None}
    if failure == 'wrong_user': p['context'].return_value = {**CTX, 'user_id': 'another-user'}
    if failure == 'wrong_agent': p['context'].return_value = {**CTX, 'agent_id': 'another-agent'}
    if failure == 'wrong_workspace': p['context'].return_value = {**CTX, 'workspace_id': 'another-workspace'}
    if failure == 'no_cards': generate.return_value = []
    if failure == 'timed_out': generate.side_effect = TimeoutError('generation budget')
    if failure == 'lost_generation_lock':
        cards = generate.return_value
        async def steal_lock(**kwargs):
            cache.items[activities._preview_key('case', 'admin') + ':generating'] = 'another-request'
            return cards
        generate.side_effect = steal_lock
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        response = await client.post('/admin-api/takeovers/case/activity-previews')
    assert response.status_code == status, response.text
    assert p['events'] == [] and p['create'].await_count == 0
    assert activities._preview_key('case', 'admin') not in cache.items


async def test_preview_release_failure_does_not_discard_completed_candidates(previews, monkeypatch):
    p, _, cache = previews
    monkeypatch.setattr(cache, 'eval', AsyncMock(side_effect=ConnectionError('redis lock release')))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        assert len(await load(client)) == 5
    assert p['create'].await_count == 0


@pytest.mark.parametrize('failure,status', [
    ('disabled', 403), ('missing_batch', 404), ('changed_subject', 404),
    ('expired_aware', 409), ('expired_naive', 409),
    ('no_conversation', 422), ('wrong_user', 409), ('wrong_agent', 409),
])
async def test_publish_rechecks_candidate_and_recipient_before_persistence(previews, failure, status):
    p, _, cache = previews
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        items = await load(client)
        key = activities._preview_key('case', 'admin')
        if failure == 'disabled': p['enabled'].return_value = False
        if failure == 'missing_batch': cache.items.pop(key)
        if failure == 'changed_subject':
            batch = json.loads(cache.items[key]); batch['subject']['agentId'] = 'another-agent'
            cache.items[key] = json.dumps(batch)
        if failure.startswith('expired_'):
            batch = json.loads(cache.items[key])
            expired = datetime.now(UTC) - timedelta(minutes=1)
            if failure == 'expired_naive': expired = expired.replace(tzinfo=None)
            batch['items'][0]['card']['expires_at'] = expired.isoformat()
            cache.items[key] = json.dumps(batch)
        if failure == 'no_conversation': p['context'].return_value = {**CTX, 'conversation_id': None}
        if failure == 'wrong_user': p['context'].return_value = {**CTX, 'user_id': 'another-user'}
        if failure == 'wrong_agent': p['context'].return_value = {**CTX, 'agent_id': 'another-agent'}
        response = await client.post(f"/admin-api/takeovers/case/activity-previews/{items[0]['preview_id']}/publish")
    assert response.status_code == status, response.text
    assert p['create'].await_count == 0 and p['events'] == []


async def test_future_naive_expiry_can_publish_and_failed_attempt_can_retry(previews):
    p, _, cache = previews
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app()), base_url='http://test') as client:
        items = await load(client)
        key = activities._preview_key('case', 'admin')
        batch = json.loads(cache.items[key])
        batch['items'][0]['card']['ends_at'] = (datetime.now(UTC) + timedelta(days=1)).replace(tzinfo=None).isoformat()
        cache.items[key] = json.dumps(batch)
        url = f"/admin-api/takeovers/case/activity-previews/{items[0]['preview_id']}/publish"
        p['context'].return_value = {**CTX, 'agent_id': 'another-agent'}
        assert (await client.post(url)).status_code == 409
        p['context'].return_value = CTX
        assert (await client.post(url)).status_code == 200
    assert p['create'].await_count == 1


async def test_parallel_completion_keeps_other_valid_candidates(monkeypatch):
    from app.config import settings
    from app.services.offline import cleversee_discovery
    monkeypatch.setattr(settings, 'offline_search_provider', 'cleversee')
    monkeypatch.setattr(gen.repo, 'list_user_tags', AsyncMock(return_value=[]))
    monkeypatch.setattr(gen.repo, 'list_recent_activity_fingerprints', AsyncMock(return_value=[]))
    monkeypatch.setattr(gen, 'evidence_inputs', AsyncMock(return_value=({'memory': [], 'dialogue': []}, {})))
    candidates = [{'location_name': '地点一'}, {'location_name': '地点一'},
                  {'location_name': '地点二'}, {'location_name': '地点三'}]
    monkeypatch.setattr(cleversee_discovery, 'discover', AsyncMock(return_value=candidates))
    async def native(items, **kwargs):
        if items[0]['location_name'] == '地点二': raise RuntimeError('one provider completion failed')
        return {**deepcopy(CARD), **items[0]}
    monkeypatch.setattr(gen, '_native_card', AsyncMock(side_effect=native))
    monkeypatch.setattr(gen, 'attach_recommendation_message', AsyncMock(side_effect=lambda card, *_: card))
    cards = await gen.generate_activity_preview_cards(user_id='user', workspace_id=None, city='镇江市', source='manual')
    assert {card['location_name'] for card in cards} == {'地点一', '地点三'}
