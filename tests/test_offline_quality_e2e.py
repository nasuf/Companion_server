"""Authenticated HTTP → actual Postgres/Redis → lifecycle/evidence/gallery.

External search/model/notification transports use deterministic fixtures. No
production users or messages are used. See offline-quality-verification.md.
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.api.public import offline
from app.services.auth import create_jwt
from app.services.offline import activity_service as service, repository as repo
from app.services.offline import place_catalog, memory_note, recognition
from app.services.offline.chat_emit import build_activity_component_card
from app.services.offline.geocode import wgs84_to_gcj02
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from tests.test_proactive_activity_e2e import flow  # noqa: F401

RELEASE = json.loads((Path(__file__).parents[1]/'scripts/prompt_releases/20261005_offline_quality.json').read_text())


@pytest.fixture
async def journey(flow, monkeypatch):
    flow.app.include_router(offline.router)
    flow.client.headers['Authorization'] = 'Bearer '+create_jwt(flow.user, role="user")
    for module in (place_catalog,):
        monkeypatch.setattr(module, 'db', flow.db)
    for name in ('emit_assistant', 'emit_activity_card', 'insert_user_activity_card'):
        monkeypatch.setattr(service, name, AsyncMock(return_value='sent'))
    monkeypatch.setattr(service, 'fire_background', lambda c: c.close())
    monkeypatch.setattr(service, 'remember_user_event', lambda **k: None)
    monkeypatch.setattr(service, '_arrival_guide_text', AsyncMock(return_value='到了呀，慢慢逛'))
    monkeypatch.setattr(service, 'settings', type('Settings', (), {
        'offline_arrival_require_geocode': False, 'offline_arrival_radius_m': 200,
        'offline_activity_companion_enabled': False})())
    for entry in RELEASE['prompts']:
        d= PROMPT_DEFINITION_MAP[entry['key']]
        await flow.db.prompttemplate.upsert(where={'key':d.key},data={
            'create':dict(key=d.key,stage=d.stage,category=d.category,title=d.title,description=d.description,content=d.default_text,defaultContent=d.default_text),
            'update':dict(content=d.default_text,defaultContent=d.default_text,isEnabled=True)})
        row=await flow.db.prompttemplate.find_unique(where={'key':d.key})
        result=await flow.client.put('/admin-api/prompts/'+d.key,json={'content':entry['content'],'expected_updated_at':row.updatedAt.isoformat()})
        assert result.status_code==200, result.text
    async def create(**over):
        lat,lng=wgs84_to_gcj02(23.01,113.75)
        return await repo.create_activity(dict(user_id=flow.user,agent_id=flow.agent,
            workspace_id=flow.workspace,conversation_id=flow.conversation,status='accepted',
            title='莲湖公园走走',location_name='莲湖公园',city='东莞',address='桥头镇莲湖路',
            summary='慢慢走走',description='推荐介绍：建议看看湖面',place_lat=lat,place_lng=lng,
            **over))
    flow.create=create
    return flow


async def post(j, a, action, body=None):
    return await j.client.post('/offline/activities/'+a['id']+'/'+action,json=body or {})


async def test_gps_validation_arrival_only_no_invented_memory(journey, monkeypatch):
    j=journey; a=await j.create()
    for payload, reason in [({},'location_required'),({'lat':23.01,'lng':113.75},'low_accuracy'),
            ({'lat':24,'lng':113.75,'accuracy_m':10},'too_far'),
            ({'lat':23.01,'lng':113.75,'accuracy_m':500},'low_accuracy'),
            ({'manual_confirmation':True},'location_required')]:
        r=await post(j,a,'arrive',payload)
        assert r.status_code==422 and r.json()['detail']['reason']==reason
        assert not (await repo.get_activity(a['id'],j.user))['reached']
    r=await post(j,a,'arrive',{'lat':23.01,'lng':113.75,'accuracy_m':10})
    assert r.status_code==200 and r.json()['arrival_verified']
    assert (await post(j,a,'arrive')).status_code==200  # idempotent
    llm=AsyncMock(side_effect=AssertionError('arrival alone must never ask model to invent a story'))
    monkeypatch.setattr(memory_note,'invoke_text',llm)
    assert (await post(j,a,'archive')).status_code==200
    review=await j.client.get('/offline/activities/'+a['id']+'/review')
    assert review.status_code==200,review.text
    data=review.json()
    assert '还没有留下' in data['story'] and '推荐介绍' not in data['story']
    assert not data['can_generate_memory_note'] and not data['has_memory_note']
    assert (await post(j,a,'memory-note')).status_code==409
    llm.assert_not_called()


@pytest.mark.parametrize('image_count', [0, 1, 3])
@pytest.mark.parametrize('has_material', [False, True])
async def test_archived_review_preserves_place_album_separate_from_materials(
    journey, monkeypatch, image_count, has_material
):
    j = journey
    images = [f'/offline/media/place_fixture_{i}.jpg' for i in range(image_count)]
    a = await j.create(image_urls=images)
    path = '/offline/activities/' + a['id']
    assert (await j.client.get(path)).json()['image_urls'] == images
    assert (await post(j, a, 'arrive', {'lat': 23.01, 'lng': 113.75, 'accuracy_m': 10})).status_code == 200
    storage_key = f'user_journey_{a["id"]}.jpg'
    material = '/offline/media/' + storage_key
    if has_material:
        await repo.create_captured_media(
            recommendation_id=a['id'], user_id=j.user, storage_key=storage_key,
            url=material, mime='image/jpeg', size=123, width=10, height=10,
            source_message_id=None,
        )
    monkeypatch.setattr(memory_note, 'invoke_text', AsyncMock(return_value=json.dumps({
        'body': '你留下了一张现场照片。'
    }, ensure_ascii=False)))
    archived = await post(j, a, 'archive')
    assert archived.status_code == 200, archived.text
    assert archived.json()['image_urls'] == images
    for _ in range(2):  # Reopening an already archived trip retains the same album.
        response = await j.client.get(path + '/review')
        assert response.status_code == 200, response.text
        review = response.json()
        assert review['image_urls'] == images
        assert review['gallery'] == ([material] if has_material else [])
        assert review['cover_url'] == (images[0] if images else material if has_material else None)
        if not has_material:
            assert not review['can_generate_memory_note']  # Place photos aren't trip evidence.
    assert (await j.client.get(path)).json()['image_urls'] == images


async def test_actual_journey_snapshot_excludes_before_after_and_agent_claims(journey, monkeypatch):
    j=journey;a=await j.create()
    async def message(text, offset, role='user', metadata=None):
        await j.db.execute_raw('''INSERT INTO messages(id,conversation_id,role,content,metadata,created_at)
            VALUES ($1,$2,$3,$4,$5::jsonb,CURRENT_TIMESTAMP+($6||' seconds')::interval)''',
            uuid4().hex,j.conversation,role,text,json.dumps(metadata or {}),str(offset))
    await message('出发前的旧话',-100)
    await post(j,a,'arrive',{'lat':23.01,'lng':113.75,'accuracy_m':5})
    await message('我在湖边坐了一会儿，风吹着很舒服',0)
    await message('你已经参观了整座博物馆',0,'assistant')
    await message('用户确认到达',0,metadata={'component_card':{'type':'offline_activity'}})
    await message('明天无关的事',100)
    async def generate(model,prompt):
        assert '湖边坐了一会儿' in prompt
        for excluded in ('出发前的旧话','你已经参观了整座博物馆','用户确认到达','明天无关的事','推荐介绍：'):
            assert excluded not in prompt
        return json.dumps({'body':'你说在湖边坐了一会儿，风吹着很舒服。这句感想也一起收好了。'},ensure_ascii=False)
    llm=AsyncMock(side_effect=generate);monkeypatch.setattr(memory_note,'invoke_text',llm)
    r=await post(j,a,'archive');assert r.status_code==200,r.text
    review=(await j.client.get('/offline/activities/'+a['id']+'/review')).json()
    assert review['can_generate_memory_note'] and review['has_memory_note']
    note=(await post(j,a,'memory-note')).json()
    assert note['travel_note']==review['story']
    llm.assert_awaited_once()


async def test_delete_owned_unstarted_only_and_no_resurrection(journey):
    j=journey;a=await j.create()
    path='/offline/activities/'+a['id']
    denied=await j.client.delete(path,headers={'Authorization':'Bearer '+create_jwt('other-user', role='user')})
    assert denied.status_code==404
    r=await j.client.delete(path);assert r.status_code==200,r.text
    assert (await j.client.delete(path)).status_code==200
    assert (await post(j,a,'accept')).status_code==409
    assert (await post(j,a,'arrive')).status_code==409
    listing=(await j.client.get('/offline/activities')).json()
    assert a['id'] not in {x['id'] for x in listing['pending']}
    b=await j.create();await post(j,b,'arrive',{'lat':23.01,'lng':113.75,'accuracy_m':10})
    assert (await j.client.delete('/offline/activities/'+b['id'])).status_code==409


async def test_arrive_delete_race_has_one_valid_terminal_state(journey):
    j=journey;a=await j.create()
    results=await asyncio.gather(post(j,a,'arrive',{'lat':23.01,'lng':113.75,'accuracy_m':10}),j.client.delete('/offline/activities/'+a['id']))
    row=await repo.get_activity(a['id'],j.user)
    assert (row['status']=='cancelled' and not row['reached']) or (row['status']=='accepted' and row['reached'])
    assert sorted(r.status_code for r in results)==[200,409]


async def test_concurrent_fragments_cannot_repeat_or_exceed_cap(journey):
    j=journey;a=await j.create();await post(j,a,'arrive',{'lat':23.01,'lng':113.75,'accuracy_m':10})
    async def fragment(text):
        return await repo.create_fragment(recommendation_id=a['id'],tier='rare',text=text)
    results=await asyncio.gather(*(fragment('这张照片的光让我想起小时候放学的路') for _ in range(4)))
    assert sum(bool(r) for r in results)==1
    assert not await fragment('这张照片的光，让我想起小时候放学的路。')
    await asyncio.gather(*(fragment(text) for text in ['我忽然想给家里那盆花浇点水','小店的字写得歪歪扭扭的很可爱','听你说起这个我也馋那碗热面了']))
    assert await repo.count_fragments(a['id'])==3
    await post(j,a,'archive')
    assert not await fragment('结束之后不能再产生新的感想')


async def test_explicit_manual_arrival_is_not_gps_verified(journey):
    j=journey;a=await j.create()
    await j.db.execute_raw('UPDATE offline_activity_recommendations SET place_lat=NULL,place_lng=NULL WHERE id=$1',a['id'])
    rejected = await post(j,a,'arrive')
    assert rejected.status_code == 422
    assert rejected.json()['detail'] == {
        'reason': 'no_geocode', 'message': '暂时没能确认到达，请稍后再试'
    }
    r=await post(j,a,'arrive',{'manual_confirmation':True})
    assert r.status_code==200 and r.json()['reached'] and not r.json()['arrival_verified']
    assert service.insert_user_activity_card.await_args.kwargs['status_label'] == '我到了'
    assert service.emit_assistant.await_args.kwargs['message'] == '到了呀，慢慢逛'


async def test_public_place_cache_contains_no_personal_copy(journey):
    card={'city':'东莞','location_name':'莲湖公园','address':'桥头镇莲湖路','official_url':'https://example.com/place','summary':'个人偏好和秘密'}
    await place_catalog.save_place(card,[{'url':'https://example.com/a.jpg','source_url':card['official_url']}])
    cached=await place_catalog.load_place(card)
    assert cached and 'summary' not in cached
    assert await place_catalog.load_place({**card,'city':'西安'}) is None


async def test_recommendation_http_card_detail_share_same_record(journey, monkeypatch, tmp_path):
    from app.services.offline import activity_generation as gen, activity_images as media
    from app.services.offline.providers.search import SearchResult
    j=journey
    ctx=dict(user_id=j.user,agent_id=j.agent,workspace_id=j.workspace,conversation_id=j.conversation,
             user_location_city='东莞',user_location_region='广东',has_location=True,agent_name='小伴')
    monkeypatch.setattr(repo,'resolve_user_context',AsyncMock(return_value=ctx))
    monkeypatch.setattr(repo,'list_user_tags',AsyncMock(return_value=[]))
    monkeypatch.setattr(repo,'memory_brief',AsyncMock(return_value=''))
    monkeypatch.setattr(offline,'is_activity_enabled',AsyncMock(return_value=True))
    monkeypatch.setattr(service,'geocode_address',AsyncMock(return_value=wgs84_to_gcj02(23.01,113.75)))
    url='https://example.com/park-'+uuid4().hex
    result=SearchResult(title='东莞莲湖公园',url=url,content='东莞桥头镇莲湖路的莲湖公园，可沿湖散步',
                        images=[{'url':'https://image.example/park.jpg'}])
    monkeypatch.setattr(gen,'tavily_search',AsyncMock(return_value=[result]))
    monkeypatch.setattr(media,'tavily_place_images',AsyncMock(return_value=[]))
    monkeypatch.setattr(media.storage,'_MEDIA_DIR',tmp_path)
    monkeypatch.setattr(media,'page_image_evidence',AsyncMock(return_value={'https://image.example/park.jpg':'place_page_cover'}))
    monkeypatch.setattr(media,'_download_image',AsyncMock(return_value=(b'fixture','b'*64,123)))
    card=dict(title='莲湖公园走走',location_name='莲湖公园',address='桥头镇莲湖路',summary='沿湖慢慢走走',
              description='沿湖散步 https://example.com/unusable',category='公园',official_url=url,image_urls=['https://untrusted.example/invented.jpg'])
    async def model(_,prompt):
        return json.dumps(card,ensure_ascii=False) if 'image_urls' in prompt else json.dumps({'text':'沿湖慢慢走走'},ensure_ascii=False)
    monkeypatch.setattr(gen,'invoke_text',model)
    r=await j.client.post('/offline/activities/recommend')
    assert r.status_code==200 and r.json(),r.text
    activity=r.json()
    detail=(await j.client.get('/offline/activities/'+activity['id'])).json()
    chat=build_activity_component_card(activity,status_label='待确定')
    assert chat['title']==detail['title']==activity['title']
    assert chat['body']==detail['summary']==activity['summary']
    assert chat['payload']['image_url']==detail['image_urls'][0]
    assert chat['payload']['activity_id']==detail['id']
    assert 'http' not in detail['description'] and detail['official_url']==url
    assert detail['image_urls'][0].startswith('/offline/media/place_')


async def test_flutter_real_http_client(journey, monkeypatch):
    import os, shutil, socket
    import uvicorn
    if not os.getenv('OFFLINE_FLUTTER_E2E'):
        pytest.skip('Set OFFLINE_FLUTTER_E2E=1 to run the sibling Flutter checkout')
    assert shutil.which('flutter'), 'Flutter SDK required for cross-client E2E'
    j=journey
    monkeypatch.setattr(offline, 'is_activity_enabled', AsyncMock(return_value=True))
    monkeypatch.setattr(repo, 'resolve_user_context', AsyncMock(return_value={
        'conversation_id':j.conversation, 'workspace_id':j.workspace, 'user_location_city':'镇江市',
        'agent_id':j.agent, 'user_id':j.user,
    }))
    monkeypatch.setattr(service, 'generate_activity_card', AsyncMock(return_value=None))
    a=await j.create(image_urls=[f'/offline/media/place_fixture_{i}.jpg' for i in range(3)])
    b=await j.create()
    await j.db.execute_raw('UPDATE offline_activity_recommendations SET place_lat=NULL,place_lng=NULL WHERE id=$1',a['id'])
    sock=socket.socket();sock.bind(('127.0.0.1',0))
    port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(j.app,log_level='error',lifespan='off'))
    running=asyncio.create_task(server.serve(sockets=[sock]))
    process = None
    try:
        for _ in range(100):
            if server.started:break
            await asyncio.sleep(.02)
        assert server.started
        env={**os.environ,'OFFLINE_E2E_API':f'http://127.0.0.1:{port}',
             'OFFLINE_E2E_TOKEN':create_jwt(j.user,role='user'),
             'OFFLINE_E2E_ACTIVITY':a['id'],'OFFLINE_E2E_DELETE':b['id']}
        process=await asyncio.create_subprocess_exec('flutter','test','--no-pub','test/offline_api_e2e_test.dart','-r','expanded',
            cwd=Path(__file__).parents[2]/'Companion_flutter',env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        output,_=await asyncio.wait_for(process.communicate(),timeout=120)
        assert process.returncode==0,output.decode()
        assert 'All tests passed' in output.decode()
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        server.should_exit=True
        await running


@pytest.mark.parametrize('model_result', ['unavailable', '你拍下了沿路的照片，玩得很开心'])
async def test_note_failure_or_invented_photo_uses_actual_user_quote(journey, monkeypatch, model_result):
    j=journey; a=await j.create()
    await post(j,a,'arrive',{'lat':23.01,'lng':113.75,'accuracy_m':10})
    await j.db.execute_raw("INSERT INTO messages(id,conversation_id,role,content) VALUES ($1,$2,'user',$3)",uuid4().hex,j.conversation,'湖边的风有点凉')
    result=AsyncMock(side_effect=RuntimeError('unavailable')) if model_result=='unavailable' else AsyncMock(return_value=json.dumps({'body':model_result},ensure_ascii=False))
    monkeypatch.setattr(memory_note,'invoke_text',result)
    assert (await post(j,a,'archive')).status_code==200
    data=(await j.client.get('/offline/activities/'+a['id']+'/review')).json()
    assert '湖边的风有点凉' in data['story']
    assert '照片' not in data['story'] and '开心' not in data['story']


async def test_explicit_completion_feedback_after_archive_is_included(journey, monkeypatch):
    j=journey; a=await j.create()
    await post(j,a,'arrive',{'lat':23.01,'lng':113.75,'accuracy_m':10})
    await post(j,a,'archive')
    await repo.create_activity_feedback(recommendation_id=a['id'],user_id=j.user,kind='completion',text='临走前喝了一杯热茶')
    async def generate(model,prompt):
        assert '临走前喝了一杯热茶' in prompt
        return json.dumps({'body':'你说临走前喝了一杯热茶，这句话也收进了这次的记录。'},ensure_ascii=False)
    monkeypatch.setattr(memory_note,'invoke_text',generate)
    data=(await j.client.get('/offline/activities/'+a['id']+'/review')).json()
    assert data['can_generate_memory_note'] and '热茶' in data['story']


async def test_legacy_gallery_repair_updates_chat_cover_with_backup(journey, monkeypatch, tmp_path):
    from scripts import repair_offline_galleries as repair
    from types import SimpleNamespace
    j=journey; a=await j.create()
    await j.db.execute_raw('UPDATE offline_activity_recommendations SET image_urls=$1::jsonb WHERE id=$2',json.dumps(['https://unverified.example/wrong.jpg']),a['id'])
    card=build_activity_component_card({**a,'image_urls':['https://unverified.example/wrong.jpg']},status_label='待出行')
    mid=uuid4().hex
    await j.db.execute_raw("INSERT INTO messages(id,conversation_id,role,content,metadata) VALUES ($1,$2,'assistant','',$3::jsonb)",mid,j.conversation,json.dumps({'component_card':card}))
    # Reuse fixture connection; the production script owns its own lifecycle.
    class Database:
        async def connect(self): pass
        async def disconnect(self): pass
        def __getattr__(self,name): return getattr(j.db,name)
    monkeypatch.setattr(repair,'db',Database())
    monkeypatch.setattr(repair,'persist_activity_images',AsyncMock(return_value=[]))
    backup=tmp_path/'before.json'
    await repair.run(SimpleNamespace(apply=True,backup=str(backup)))
    assert any(r['id']==a['id'] for r in json.loads(backup.read_text()))
    assert (await repo.get_activity(a['id'],j.user))['image_urls']==[]
    row=(await j.db.query_raw('SELECT metadata FROM messages WHERE id=$1',mid))[0]
    assert row['metadata']['component_card']['payload']['image_url'] is None


def test_release_matches_web_save_normalization():
    from scripts.publish_offline_quality_prompts import validate_entry
    for entry in RELEASE['prompts']:
        assert entry['content'] == entry['content'].strip()
        validate_entry(entry)


async def test_cancellation_revokes_plan_context_without_erasing_real_memories(journey, monkeypatch):
    from app.services.proactive import context as proactive
    j=journey; a=await j.create()
    memory_id=uuid4().hex
    await j.db.execute_raw("""INSERT INTO memories_user (id,user_id,workspace_id,content,importance,level,main_category,sub_category,updated_at)
        VALUES ($1,$2,$3,'我喜欢莲湖公园的荷花',0.7,2,'偏好边界','喜好',CURRENT_TIMESTAMP)""", memory_id,j.user,j.workspace)
    await j.db.execute_raw("UPDATE offline_activity_recommendations SET next_companion_at=CURRENT_TIMESTAMP, companion_claim_token='inflight', companion_claimed_at=CURRENT_TIMESTAMP WHERE id=$1",a['id'])
    assert (await j.client.delete('/offline/activities/'+a['id'])).status_code == 200
    row=await repo.get_activity(a['id'], j.user)
    assert row['status']=='cancelled' and row.get('next_companion_at') is None and row.get('companion_claim_token') is None
    brief=await repo.get_active_activity_brief(j.user, j.workspace)
    assert not brief.get('title')
    facts=brief['cancelled_plans']
    assert len(facts)==1 and facts[0]['activity_id']==a['id'] and not facts[0]['arrival_record']
    assert facts[0]['preference_change'] is None and facts[0]['cancellation_reason'] is None
    assert await repo.get_cancelled_activity_plans('other-user', j.workspace)==[]
    assert await repo.get_cancelled_activity_plans(j.user, 'other-workspace')==[]
    assert (await j.db.query_raw('SELECT content FROM memories_user WHERE id=$1',memory_id))[0]['content']=='我喜欢莲湖公园的荷花'
    # The same live state is visible to everyday proactive replies, without an
    # asynchronous LLM extraction or deleting unrelated memories.
    monkeypatch.setattr(proactive,'db',j.db)
    for name,value in [('get_cached_schedule',None),('load_core_memory_strings',[]),
        ('_load_proactive_memories',([],[])),('get_topic_intimacy',50),
        ('get_latest_portrait',''),('_load_recent_context','以前说过想去莲湖公园'),('load_ai_mood',None)]:
        monkeypatch.setattr(proactive,name,AsyncMock(return_value=value))
    ctx=await proactive.build_proactive_context(workspace_id=j.workspace,user_id=j.user,agent_id=j.agent,trigger_type='silence_wakeup',stage='warming')
    assert 'cancelled' in ctx['recent_context'] and a['id'] in ctx['recent_context']
    # Lightweight memory tiers also bypass the main chat prompt; cancellation
    # must reach their existing context field rather than disappear on this path.
    from types import SimpleNamespace
    from app.services.chat.reply_generate import generate_reply
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    monkeypatch.setattr('app.services.chat.prompt_builder._build_personality_section',AsyncMock(return_value=None))
    async def tier(**params):
        assert 'cancelled' in params['context'] and a['id'] in params['context']
        return '好呀，按你的安排来'
    result=await generate_reply(user_id=j.user,workspace_id=j.workspace,contradiction_inquiry=None,
        detected_intent=IntentResult(intent=IntentType.NONE,confidence=1), memory_relevance='weak',
        relational_context=None,schedule_context=None,delay_context=None,l3_memories=[],classified_memories=[],
        messages_dicts=[],portrait=None,prompt_user_emotion=None,user_message='今天先不去了',
        agent=SimpleNamespace(name='小伴'),reply_count=1,max_reply_count=4,max_total=80,
        tier_fns={'weak':tier},truncate_fn=lambda text,_:text,pipe_fallback_fn=lambda text,*_: [text])
    assert result[1]=='好呀，按你的安排来'
    # A new plan at the same place remains active; cancellation is per activity.
    new=await j.create()
    brief=await repo.get_active_activity_brief(j.user, j.workspace)
    assert brief['title']==new['title'] and len(brief['cancelled_plans'])==1


async def test_fill_only_gallery_repair_preserves_existing_photos(journey, monkeypatch, tmp_path):
    from scripts import repair_offline_galleries as repair
    from types import SimpleNamespace
    j=journey; a=await j.create(); empty=await j.create()
    await j.db.execute_raw('UPDATE offline_activity_recommendations SET image_urls=$1::jsonb WHERE id=$2',json.dumps(['/offline/media/place_existing.jpg']),a['id'])
    class Database:
        async def connect(self): pass
        async def disconnect(self): pass
        def __getattr__(self,name): return getattr(j.db,name)
    monkeypatch.setattr(repair,'db',Database())
    async def images(**kwargs):
        return ['/offline/media/place_recovered.jpg'] if kwargs['card']['id']==empty['id'] else []
    monkeypatch.setattr(repair,'persist_activity_images',images)
    await repair.run(SimpleNamespace(apply=True,backup=str(tmp_path/'before.json'),fill_only=True))
    assert (await repo.get_activity(a['id'],j.user))['image_urls']==['/offline/media/place_existing.jpg']
    assert (await repo.get_activity(empty['id'],j.user))['image_urls']==['/offline/media/place_recovered.jpg']


@pytest.mark.parametrize('have_images', [False, True])
async def test_recommendation_rejects_collections_but_prefers_photos_without_requiring_them(journey, monkeypatch, tmp_path, have_images):
    from app.services.offline import activity_generation as gen, activity_images as media
    from app.services.offline.providers.search import SearchResult
    j = journey
    ctx = dict(user_id=j.user, agent_id=j.agent, workspace_id=j.workspace, conversation_id=j.conversation,
               user_location_city='镇江市', has_location=True, agent_name='小伴')
    monkeypatch.setattr(repo, 'resolve_user_context', AsyncMock(return_value=ctx))
    monkeypatch.setattr(repo, 'list_user_tags', AsyncMock(return_value=[]))
    monkeypatch.setattr(repo, 'memory_brief', AsyncMock(return_value=''))
    monkeypatch.setattr(offline, 'is_activity_enabled', AsyncMock(return_value=True))
    monkeypatch.setattr(service, 'geocode_address', AsyncMock(return_value=None))
    article = SearchResult(title='镇江这些咖啡店，藏着整个春天！', url='https://example.com/collection', content='镇江市咖啡店合集')
    cafe = SearchResult(title='库迪咖啡(临湖苑店) - 镇江市', url='https://example.com/cafe', content='镇江市句容市临湖苑商业B2幢110号')
    museum = SearchResult(title='镇江博物馆', url='https://example.com/museum', content='镇江市润州区伯先路85号', raw_content='''# 镇江博物馆

## 展览陈列

[![Image 1](https://photo.example/one.jpg)](https://example.com/album "青铜器展")青铜器展
[![Image 2](https://photo.example/two.jpg)](https://example.com/album "陶瓷器精品展")陶瓷器精品展
[![Image 3](https://photo.example/three.jpg)](https://example.com/album "金银器精品展")金银器精品展
''')
    sources = [article, cafe, museum] if have_images else [article, cafe]
    monkeypatch.setattr(service, '_location_for_activity', lambda ctx: ('镇江市', '镇江', ['镇江']))
    monkeypatch.setattr(gen, '_search_activity_candidates', AsyncMock(return_value=(sources, sources, '镇江')))
    monkeypatch.setattr(gen.repo, 'list_recent_activity_fingerprints', AsyncMock(return_value=[]))
    # Even an invalid model result must pass the same validation as the fallback.
    monkeypatch.setattr(gen, 'invoke_text', AsyncMock(return_value=json.dumps({
        'title': article.title, 'location_name': '镇江这些咖啡店',
        'address': '镇江这些咖啡店', 'official_url': article.url,
    }, ensure_ascii=False)))
    monkeypatch.setattr(gen, '_recommendation_copy', AsyncMock(return_value=''))
    monkeypatch.setattr(media.place_catalog, 'load_place', AsyncMock(return_value=None))
    monkeypatch.setattr(media.place_catalog, 'save_place', AsyncMock())
    monkeypatch.setattr(media.storage, '_MEDIA_DIR', tmp_path)
    monkeypatch.setattr(media, 'page_image_evidence', AsyncMock(return_value={}))
    monkeypatch.setattr(media, 'tavily_place_images', AsyncMock(return_value=[]))
    async def photo(client, url):
        index = ['one', 'two', 'three'].index(url.rsplit('/', 1)[-1].split('.')[0])
        return b'image', str(index) * 64, [0, 65535, 281474976645120][index]
    monkeypatch.setattr(media, '_download_image', photo)
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == 200, response.text
    if not have_images:
        card = response.json()
        assert card['location_name'] == '库迪咖啡(临湖苑店)'
        assert card['image_urls'] == []
        detail = (await j.client.get('/offline/activities/' + card['id'])).json()
        assert detail['image_urls'] == []
        return
    card = response.json()
    assert card['location_name'] == '镇江博物馆'
    assert len(card['image_urls']) == 3
    detail = (await j.client.get('/offline/activities/' + card['id'])).json()
    assert detail['image_urls'] == card['image_urls']
    assert detail['title'] == card['title'] == '镇江博物馆'


@pytest.mark.parametrize('case,status,reason', [
    ('disabled', 403, 'activity_disabled'),
    ('conversation', 422, 'conversation_required'),
    ('location', 422, 'location_required'),
    ('candidates', 503, 'no_suitable_activity'),
])
async def test_recommendation_failure_explains_actual_reason_without_side_effects(journey, monkeypatch, case, status, reason):
    j = journey
    ctx = dict(conversation_id=j.conversation, workspace_id=j.workspace, user_location_city='镇江市')
    if case == 'conversation':
        ctx = None
    elif case == 'location':
        ctx.pop('user_location_city')
    monkeypatch.setattr(offline, 'is_activity_enabled', AsyncMock(return_value=case != 'disabled'))
    monkeypatch.setattr(repo, 'resolve_user_context', AsyncMock(return_value=ctx))
    generate = AsyncMock(return_value=None)
    monkeypatch.setattr(service, 'generate_activity_card', generate)
    response = await j.client.post('/offline/activities/recommend')
    assert response.status_code == status
    assert response.json()['detail']['reason'] == reason
    if case == 'candidates':
        assert '定位' not in response.json()['detail']['message']
        generate.assert_awaited_once()
    else:
        generate.assert_not_awaited()
    service.emit_assistant.assert_not_awaited()
    assert await j.db.query_raw('SELECT id FROM offline_activity_recommendations WHERE user_id=$1', j.user) == []


async def test_scheduled_recommendation_still_skips_without_conversation(monkeypatch):
    monkeypatch.setattr(repo, 'resolve_user_context', AsyncMock(return_value=None))
    assert await service.create_recommendation_for_user(user_id='none', source='scheduled') is None


async def test_scoped_gallery_refill_preserves_existing_photos_and_skips_failed_discovery(journey, monkeypatch, tmp_path):
    from scripts import repair_offline_galleries as repair
    from types import SimpleNamespace
    j = journey
    old = '/offline/media/place_original.jpg'
    a = await j.create(image_urls=[old], search_sources=[{'kind': 'image', 'local_url': old}])
    untouched = await j.create(image_urls=[old])
    class Database:
        async def connect(self): pass
        async def disconnect(self): pass
        def __getattr__(self, name): return getattr(j.db, name)
    monkeypatch.setattr(repair, 'db', Database())
    args = SimpleNamespace(apply=True, backup=str(tmp_path / 'before.json'), refill_incomplete=True, activity_id=[a['id']])
    monkeypatch.setattr(repair, 'persist_activity_images', AsyncMock(return_value=[]))
    await repair.run(args)
    assert (await repo.get_activity(a['id'], j.user))['image_urls'] == [old]
    monkeypatch.setattr(repair, 'persist_activity_images', AsyncMock(return_value=['/offline/media/place_second.jpg', old, '/offline/media/place_third.jpg']))
    args.backup = str(tmp_path / 'before-refill.json')
    await repair.run(args)
    assert (await repo.get_activity(a['id'], j.user))['image_urls'] == [old, '/offline/media/place_second.jpg', '/offline/media/place_third.jpg']
    assert (await repo.get_activity(untouched['id'], j.user))['image_urls'] == [old]
