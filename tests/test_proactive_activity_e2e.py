"""Real PostgreSQL + Redis + admin HTTP save → render → generate → persist → WS.

Only model output and external notification/voice transport are stubbed. Requires
an isolated, migrated database named companion_proactive_e2e on localhost.
"""
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from prisma import Prisma
from redis.asyncio import Redis

from app.api.admin import prompts as admin
from app.services.offline import activity_companion as companion
from app.services.offline import activity_generation as generation
from app.services.offline import activity_service as service
from app.services.offline import activity_message_context as context
from app.services.offline import repository as repo
from app.services.proactive import emit
from app.services.prompting import store
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.trace_components import start_prompt_render_trace, reset_prompt_render_trace

RELEASE = json.loads((Path(__file__).resolve().parents[1] / 'scripts/prompt_releases/20261005_proactive_common.json').read_text())


@pytest.fixture
async def flow(monkeypatch):
    url, redis_url = os.getenv('PROACTIVE_E2E_DATABASE_URL'), os.getenv('PROACTIVE_E2E_REDIS_URL')
    if not url or not redis_url:
        pytest.skip('isolated PostgreSQL/Redis URLs required')
    assert urlsplit(url).hostname in {'localhost', '127.0.0.1'}
    assert urlsplit(url).path == '/companion_proactive_e2e'
    assert urlsplit(redis_url).hostname in {'localhost', '127.0.0.1'}
    database = Prisma(datasource={'url': url}, http={'trust_env': False})
    redis = Redis.from_url(redis_url, decode_responses=True)
    await database.connect()
    tag = uuid4().hex
    monkeypatch.setattr(store, '_redis_key', lambda key: f'e2e:{tag}:text:{key}')
    monkeypatch.setattr(store, '_enabled_redis_key', lambda key: f'e2e:{tag}:enabled:{key}')
    monkeypatch.setattr(store, 'get_redis', AsyncMock(return_value=redis))
    for module in (store, repo, context, companion, emit):
        monkeypatch.setattr(module, 'db', database)
    monkeypatch.setattr('app.services.runtime.tasks.fire_background', lambda coro: coro.close())
    monkeypatch.setattr('app.services.speech_output.policy.should_generate_voice', AsyncMock(return_value=False))
    ws = AsyncMock()
    monkeypatch.setattr(emit, 'manager', SimpleNamespace(send_to_workspace=ws))
    ids = [uuid4().hex for _ in range(4)]
    user, agent, workspace, conversation = ids
    now = datetime.now(UTC).replace(tzinfo=None)
    await database.execute_raw('INSERT INTO users (id, username, updated_at) VALUES ($1,$2,$3::timestamp)', user, 'e2e-'+tag, now)
    await database.execute_raw("INSERT INTO ai_agents (id,name,user_id,updated_at,mbti) VALUES ($1,'小伴',$2,$3::timestamp,'{\"EI\":80,\"NS\":60,\"TF\":40,\"JP\":30}'::jsonb)", agent,user,now)
    await database.execute_raw("INSERT INTO chat_workspaces (id,user_id,agent_id,status,updated_at) VALUES ($1,$2,$3,'active',$4::timestamp)",workspace,user,agent,now)
    await database.execute_raw('INSERT INTO conversations (id,user_id,agent_id,workspace_id,updated_at) VALUES ($1,$2,$3,$4,$5::timestamp)',conversation,user,agent,workspace,now)
    for entry in RELEASE['prompts']:
        d = PROMPT_DEFINITION_MAP[entry['key']]
        await database.prompttemplate.upsert(where={'key':d.key}, data={
            'create': dict(key=d.key, stage=d.stage,category=d.category,title=d.title,description=d.description,content=d.default_text,defaultContent=d.default_text),
            'update': dict(content=d.default_text,defaultContent=d.default_text,isEnabled=True),
        })
    app = FastAPI()
    app.include_router(admin.router)
    app.dependency_overrides[admin.require_admin_jwt] = lambda: {'sub':'e2e-admin'}
    token = start_prompt_render_trace()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            yield SimpleNamespace(app=app,db=database,redis=redis,client=client,ws=ws,user=user,agent=agent,workspace=workspace,conversation=conversation)
    finally:
        reset_prompt_render_trace(token)
        keys = [key async for key in redis.scan_iter(match=f'e2e:{tag}:*')]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        await database.disconnect()


async def publish(flow):
    for entry in RELEASE['prompts']:
        key = entry['key']
        before = await flow.db.prompttemplate.find_unique(where={'key':key})
        response = await flow.client.put('/admin-api/prompts/'+key, json={
            'content':entry['content'],'expected_updated_at':before.updatedAt.isoformat(),
        })
        assert response.status_code == 200, response.text
        row = await flow.db.prompttemplate.find_unique(where={'key':key})
        versions = await flow.db.prompttemplateversion.find_many(where={'promptKey':key},order={'createdAt':'desc'})
        assert row.isEnabled == before.isEnabled
        assert row.defaultContent == before.defaultContent
        assert versions[0].changeType == 'manual_save'
        assert row.content == await flow.redis.get(store._redis_key(key)) == entry['content']
        # Stale Web editor cannot overwrite the newly published version.
        response = await flow.client.put('/admin-api/prompts/'+key, json={
            'content':'stale overwrite','expected_updated_at':before.updatedAt.isoformat(),
        })
        assert response.status_code == 409


@pytest.mark.asyncio
async def test_web_publication_activity_generation_and_guarded_delivery(flow, monkeypatch):
    await publish(flow)
    await flow.db.message.create(data={'conversationId':flow.conversation,'role':'user','content':'我到了莲湖公园，想随便逛逛'})
    # Deliberately seed different owners to detect accidental AI→user attribution.
    for table, text in [('memories_user','用户喜欢拍照'),('memories_ai','AI喜欢摸鱼')]:
        await flow.db.execute_raw(f"INSERT INTO {table} (id,user_id,workspace_id,content,importance,level,main_category,sub_category,updated_at) VALUES ($1,$2,$3,$4,0.8,2,'偏好边界','喜好',CURRENT_TIMESTAMP)",uuid4().hex,flow.user,flow.workspace,text)
    ctx = await repo.resolve_user_context(flow.user,flow.workspace)
    activity = dict(id=uuid4().hex,user_id=flow.user,agent_id=flow.agent,workspace_id=flow.workspace,conversation_id=flow.conversation,title='莲湖公园散步',location_name='莲湖公园',city='东莞',category='公园',summary='公园散步',arrival_confirmed_at=datetime.now(UTC))
    captured = []
    async def model(_model,prompt):
        captured.append(prompt)
        assert '【主动交流通用前提】' in prompt
        assert '我到了莲湖公园' in prompt and '小伴' in prompt
        assert '{dialogue_context}' not in prompt and '{personality_brief}' not in prompt
        assert 'AI喜欢摸鱼' not in prompt
        return '{"text":"莲湖慢慢逛就好||不用赶时间"}' if '只输出合法JSON' in prompt else '莲湖公园可以看看||随便走走'
    for module in (generation,service,companion):
        monkeypatch.setattr(module,'invoke_text',model)
        # The model transport is stubbed; constructing a paid provider is also
        # outside this deterministic DB/Redis/HTTP/WS integration test.
        monkeypatch.setattr(module,'get_chat_model',lambda: None)
    invite = await generation.generate_activity_invite_message(activity=activity,user_id=flow.user,workspace_id=flow.workspace)
    arrival = await service._arrival_guide_text(activity,ctx)
    messages = await companion._recent_messages(flow.conversation)
    memory, preference = await companion._user_memory_and_preference(activity,ctx)
    assert '用户喜欢拍照' in memory and 'AI喜欢摸鱼' not in memory
    outputs = [invite,arrival]
    for topic in ('观察','感受','杂谈'):
        outputs.append(await companion._generate_companion_message(topic=topic,activity=activity,ctx=ctx,messages=messages,user_memory=memory,user_preference=preference))
    assert len(captured) == 5
    assert all('||' in output for output in outputs)  # preserved until single-bubble delivery
    for output in outputs:
        mid = await emit.emit_proactive_message(conversation_id=flow.conversation,user_id=flow.user,agent_id=flow.agent,workspace_id=flow.workspace,message=output,trigger_type='offline_activity_e2e',voice_eligible=False)
        row = await flow.db.message.find_unique(where={'id':mid})
        assert row.content == output.replace('||',' ')
        assert flow.ws.call_args.args[2]['text'] == row.content
        assert 'proactive.common_rules' in str(row.metadata)
    assert flow.ws.await_count == 5
    assert await flow.db.proactivechatlog.count(where={'conversationId':flow.conversation}) == 5
    # A reply arriving during generation prevents DB insertion, log and WS.
    cutoff = datetime.now(UTC)-timedelta(seconds=1)
    await flow.db.message.create(data={'conversationId':flow.conversation,'role':'user','content':'我先走了'})
    mid = await emit.emit_proactive_message(conversation_id=flow.conversation,user_id=flow.user,agent_id=flow.agent,workspace_id=flow.workspace,message='应取消',trigger_type='offline_activity_e2e',voice_eligible=False,abort_if_user_replied_since=cutoff)
    assert not mid and flow.ws.await_count == 5
    # A companion delivery key is consumed only once; closed activity never delivers.
    reserved = datetime.now(UTC)
    key = uuid4().hex
    await flow.db.execute_raw("INSERT INTO offline_activity_recommendations (id,user_id,agent_id,workspace_id,conversation_id,title,status,reached,companion_state,updated_at) VALUES ($1,$2,$3,$4,$5,'莲湖公园','accepted',TRUE,$6::jsonb,CURRENT_TIMESTAMP)",activity['id'],flow.user,flow.agent,flow.workspace,flow.conversation,json.dumps({'delivery_key':key,'delivery_reserved_at':reserved.isoformat()}))
    kwargs=dict(conversation_id=flow.conversation,user_id=flow.user,agent_id=flow.agent,workspace_id=flow.workspace,message=outputs[-1],trigger_type='offline_activity_e2e',voice_eligible=False,guard_activity_id=activity['id'],guard_delivery_key=key,extra_metadata={'offline_companion_delivery_key':key})
    assert await emit.emit_proactive_message(**kwargs)
    assert not await emit.emit_proactive_message(**kwargs)
    next_key = uuid4().hex
    await flow.db.execute_raw("UPDATE offline_activity_recommendations SET companion_state=$2::jsonb WHERE id=$1",activity['id'],json.dumps({'delivery_key':next_key,'delivery_reserved_at':datetime.now(UTC).isoformat()}))
    kwargs.update(guard_delivery_key=next_key,extra_metadata={'offline_companion_delivery_key':next_key})
    await flow.db.message.create(data={'conversationId':flow.conversation,'role':'user','content':'不用陪我了'})
    assert not await emit.emit_proactive_message(**kwargs)
    await flow.db.execute_raw("UPDATE offline_activity_recommendations SET status='completed',companion_state=$2::jsonb WHERE id=$1",activity['id'],json.dumps({'delivery_key':next_key,'delivery_reserved_at':datetime.now(UTC).isoformat()}))
    assert not await emit.emit_proactive_message(**kwargs)
    assert flow.ws.await_count == 6
    assert await flow.db.proactivechatlog.count(where={'conversationId':flow.conversation}) == 6
    # Same common rules are hot-read by daily proactive messages; admin disable is respected.
    daily = await store.get_prompt_text('proactive.silence_plain')
    assert str(daily).startswith(RELEASE['prompts'][0]['content'])
    response = await flow.client.put('/admin-api/prompts/proactive.common_rules/enabled',json={'is_enabled':False})
    assert response.status_code == 200
    assert '【主动交流通用前提】' not in str(await store.get_prompt_text('offline.arrival_guide'))
    assert '【主动交流通用前提】' not in str(await store.get_prompt_text('proactive.silence_plain'))
