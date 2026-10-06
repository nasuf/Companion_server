"""Isolated PostgreSQL/Redis: atomic audit, cache failure, stale saves and worker snapshots."""
import asyncio
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from prisma import Prisma
from redis.asyncio import Redis
from app.services.prompting import store
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services import runtime_config as runtime
from app.api.admin import prompts as admin

@pytest.fixture
async def flow(monkeypatch):
    url, redis_url = os.getenv('PROACTIVE_E2E_DATABASE_URL'), os.getenv('PROACTIVE_E2E_REDIS_URL')
    if not url or not redis_url: pytest.skip('isolated PG/Redis required')
    assert urlsplit(url).hostname in {'localhost','127.0.0.1'}
    assert urlsplit(url).path=='/companion_proactive_e2e'
    assert urlsplit(redis_url).hostname in {'localhost','127.0.0.1'}
    database=Prisma(datasource={'url':url},http={'trust_env':False})
    redis=Redis.from_url(redis_url,decode_responses=True)
    await database.connect()
    d=replace(PROMPT_DEFINITION_MAP['memory.relevance'],key='test.c01.'+uuid4().hex,default_text='initial {message}')
    monkeypatch.setattr(store,'PROMPT_DEFINITION_MAP',{d.key:d})
    monkeypatch.setattr(store,'PROMPT_DEFINITIONS',[d])
    monkeypatch.setattr(store,'db',database)
    monkeypatch.setattr(runtime,'db',database)
    monkeypatch.setattr(store,'get_redis',AsyncMock(return_value=redis))
    monkeypatch.setattr(store,'_schedule_eval',lambda _: None)
    token=store._prompt_snapshot.set(None)
    config_token=runtime._current_snapshot.set(None)
    await store.ensure_prompt_templates()
    app=FastAPI(); app.include_router(admin.router)
    app.dependency_overrides[admin.require_admin_jwt]=lambda: {'role':'admin'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        try: yield SimpleNamespace(app=app,db=database,redis=redis,key=d.key,d=d,client=client,url=url)
        finally:
            store._prompt_snapshot.reset(token); runtime._current_snapshot.reset(config_token)
            await database.prompttemplateversion.delete_many(where={'promptKey':d.key})
            await database.prompttemplate.delete_many(where={'key':d.key})
            await database.promptpublicationcounter.delete_many(where={'promptKey':d.key})
            await redis.delete('prompt_snapshot:'+d.key,store._redis_key(d.key),store._enabled_redis_key(d.key))
            await redis.aclose(); await database.disconnect()

async def test_concurrent_guarded_saves_have_one_winner_and_one_audit(flow):
    row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    results=await asyncio.gather(*(store.update_prompt_text(flow.key,c,expected_revision=row.revision)
                                  for c in ['winner A','winner B']),return_exceptions=True)
    assert sum(isinstance(x,store.PromptUpdateConflictError) for x in results)==1
    saved=next(x for x in results if isinstance(x,dict))
    current=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    history=await store.list_prompt_versions(flow.key)
    assert current.content==saved['content']==history[0]['content']
    assert saved['revision']==current.revision==history[0]['revision']
    assert len(history)==2
    assert await flow.redis.get(store._redis_key(flow.key))==saved['content']

async def test_unconditional_old_clients_still_serialize_versions(flow):
    results=await asyncio.gather(*(store.update_prompt_text(flow.key,c) for c in ['a','b','c']))
    assert sorted(x['revision'] for x in results)==[2,3,4]
    row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    assert row.revision==4
    assert await flow.redis.get(store._redis_key(flow.key))==row.content
    assert len(await store.list_prompt_versions(flow.key))==4

async def test_audit_failure_rolls_back_content_revision_and_cache(flow):
    # A genuine database failure, not a mocked transaction.
    await flow.db.execute_raw("CREATE FUNCTION c01_fail_audit() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.prompt_key LIKE 'test.c01.%' AND NEW.change_type='manual_save' THEN RAISE EXCEPTION 'synthetic audit failure'; END IF; RETURN NEW; END $$")
    await flow.db.execute_raw('CREATE TRIGGER c01_fail_audit BEFORE INSERT ON prompt_template_versions FOR EACH ROW EXECUTE FUNCTION c01_fail_audit()')
    try:
        with pytest.raises(Exception): await store.update_prompt_text(flow.key,'must roll back')
        row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
        assert row.revision==1 and row.content==flow.d.default_text
        assert await flow.redis.get(store._redis_key(flow.key))==row.content
        assert len(await store.list_prompt_versions(flow.key))==1
    finally:
        await flow.db.execute_raw('DROP TRIGGER c01_fail_audit ON prompt_template_versions')
        await flow.db.execute_raw('DROP FUNCTION c01_fail_audit()')

async def test_cache_outage_after_commit_does_not_undo_save(flow,monkeypatch):
    original=store.get_redis
    monkeypatch.setattr(store,'get_redis',AsyncMock(side_effect=ConnectionError('synthetic outage')))
    saved=await store.update_prompt_text(flow.key,'committed despite cache failure')
    assert saved['cache_synced'] is False
    assert (await flow.db.prompttemplate.find_unique(where={'key':flow.key})).content==saved['content']
    # Hot path reads the database, even while Redis is unavailable.
    assert await store.get_prompt_text(flow.key)==saved['content']
    assert len(await store.list_prompt_versions(flow.key))==2
    monkeypatch.setattr(store,'get_redis',original)
    await store.ensure_prompt_templates()
    assert await flow.redis.get(store._redis_key(flow.key))==saved['content']

async def test_delayed_old_cache_write_cannot_overwrite_newer_revision(flow):
    old=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    saved=await store.update_prompt_text(flow.key,'latest')
    assert await store._sync_cache(old) is False
    assert await flow.redis.get(store._redis_key(flow.key))==saved['content']

async def test_old_worker_raw_cache_is_not_authoritative(flow):
    saved=await store.update_prompt_text(flow.key,'latest database')
    await flow.redis.set(store._redis_key(flow.key),'stale old-worker write')
    assert await store.get_prompt_text(flow.key)==saved['content']
    assert (await store.list_prompts())[0]['content']==saved['content']
    assert await flow.redis.get(store._redis_key(flow.key))==saved['content']

@pytest.mark.parametrize('operation',['enabled','reset','restore-version'])
async def test_every_admin_mutation_has_409_guard(flow,operation):
    history=await store.list_prompt_versions(flow.key)
    await store.update_prompt_text(flow.key,'changed by another administrator')
    data={'expected_revision':1,'expected_updated_at':None}
    if operation=='enabled': data['is_enabled']=False
    if operation=='restore-version': data['version_id']=history[0]['id']
    method='PUT' if operation=='enabled' else 'POST'
    response=await flow.client.request(method,f'/admin-api/prompts/{flow.key}/{operation}',json=data)
    assert response.status_code==409
    assert (await flow.db.prompttemplate.find_unique(where={'key':flow.key})).isEnabled is True

async def test_enable_reset_restore_preserve_enable_and_audit(flow):
    first=(await store.list_prompt_versions(flow.key))[0]
    await store.set_prompt_enabled(flow.key,False)
    changed=await store.update_prompt_text(flow.key,'custom')
    reset=await store.reset_prompt_text(flow.key,expected_revision=changed['revision'])
    restored=await store.restore_prompt_version(flow.key,first['id'],expected_revision=reset['revision'])
    assert reset['is_enabled'] is restored['is_enabled'] is False
    assert restored['web_managed'] is True
    with pytest.raises(store.PromptDisabledError): await store.get_prompt_text(flow.key)
    assert len(await store.list_prompt_versions(flow.key))==5

@pytest.mark.parametrize('kind',['manual_save','reset_default','restore:historical','unknown'])
async def test_startup_never_overwrites_historical_web_updates(flow,monkeypatch,kind):
    row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    await flow.db.prompttemplateversion.create(data={'promptId':row.id,'promptKey':row.key,'content':row.content,'source':'db','changeType':kind})
    monkeypatch.setattr(store,'PROMPT_DEFINITIONS',[replace(flow.d,default_text='new code default')])
    await store.ensure_prompt_templates()
    current=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    assert current.content==flow.d.default_text
    assert current.defaultContent=='new code default' and current.webManaged
    assert len(await store.list_prompt_versions(flow.key))==2

async def test_code_only_startup_sync_is_atomic_and_enabled_preserved(flow,monkeypatch):
    await store.set_prompt_enabled(flow.key,False)
    monkeypatch.setattr(store,'PROMPT_DEFINITIONS',[replace(flow.d,default_text='new code default')])
    await store.ensure_prompt_templates()
    row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    history=await store.list_prompt_versions(flow.key)
    assert row.content=='new code default' and row.isEnabled is False and row.webManaged is False
    assert history[0]['change_type']=='code_sync' and history[0]['revision']==row.revision

async def test_turn_snapshots_and_live_disable(flow):
    binding=await runtime.bind_agent_context(None)
    try:
        assert await store.get_prompt_text(flow.key)==flow.d.default_text
        await store.update_prompt_text(flow.key,'next turn')
        assert await store.get_prompt_text(flow.key)==flow.d.default_text
        await store.set_prompt_enabled(flow.key,False)
        with pytest.raises(store.PromptDisabledError): await store.get_prompt_text(flow.key)
    finally: runtime.reset_current_agent(binding)
    await store.set_prompt_enabled(flow.key,True)
    binding=await runtime.bind_agent_context(None)
    try: assert await store.get_prompt_text(flow.key)=='next turn'
    finally: runtime.reset_current_agent(binding)

async def test_worker_restart_and_missed_broadcast_read_models_prices_overrides(flow):
    original=await flow.db.systemconfig.find_unique(where={'id':1})
    await runtime.load_caches()
    old=runtime.resolve_config_sync().local_chat_model
    tag='c01-'+uuid4().hex
    model=await flow.db.modelregistry.create(data={'identifier':tag,'provider':'ollama','displayName':tag,'inputCostPerMillion':11.0,'outputCostPerMillion':22.0})
    binding=await runtime.bind_agent_context(None)
    try:
        await flow.db.systemconfig.update(where={'id':1},data={'localChatModel':tag})
        await flow.db.modelregistry.update(where={'id':model.id},data={'inputCostPerMillion':33.0})
        await runtime.load_caches()
        assert runtime.resolve_config_sync().local_chat_model==old
        assert runtime.get_pricing(tag)['input']==11.0
    finally: runtime.reset_current_agent(binding)
    try:
        # Separate worker process starts with empty caches; no Pub/Sub is sent.
        code="""import asyncio,json
from app.db import db
from app.services import runtime_config as r
async def main():
 await db.connect()
 await r.ensure_loaded()
 print(json.dumps({'model':r.resolve_config_sync().local_chat_model,'price':r.get_pricing(%r)}))
 await db.disconnect()
asyncio.run(main())
""" % tag
        env={**os.environ,'DATABASE_URL':flow.url,'DIRECT_DATABASE_URL':flow.url,'PYTHON_DOTENV_DISABLED':'1'}
        process=await asyncio.create_subprocess_exec(sys.executable,'-c',code,env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        stdout,stderr=await process.communicate()
        assert process.returncode==0,stderr.decode()[-500:]
        actual=json.loads(stdout.decode().splitlines()[-1]); assert actual['model']==tag and actual['price']['input']==33.0
        await runtime.ensure_loaded()
        assert runtime.resolve_config_sync().local_chat_model==tag
        assert runtime.get_pricing(tag)['input']==33.0
    finally:
        await flow.db.systemconfig.update(where={'id':1},data={'localChatModel':original.localChatModel})
        await flow.db.modelregistry.delete(where={'id':model.id})
        await runtime.load_caches()

async def test_refresh_failure_keeps_last_complete_configuration(flow,monkeypatch):
    await runtime.load_caches(); before=runtime.resolve_config_sync()
    monkeypatch.setattr(runtime,'db',SimpleNamespace(systemconfig=SimpleNamespace(find_unique=AsyncMock(side_effect=ConnectionError('synthetic DB failure')))))
    await runtime.load_caches()
    assert runtime.resolve_config_sync()==before

async def test_evicted_redis_fence_cannot_resurrect_old_content(flow):
    old=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    saved=await store.update_prompt_text(flow.key,'new after eviction')
    await flow.redis.delete('prompt_snapshot:'+flow.key)
    assert await store._sync_cache(old) is False
    assert await flow.redis.get(store._redis_key(flow.key))==saved['content']

async def test_startup_racing_web_save_preserves_the_committed_customization(flow,monkeypatch):
    monkeypatch.setattr(store,'PROMPT_DEFINITIONS',[replace(flow.d,default_text='new code default')])
    await asyncio.gather(store.ensure_prompt_templates(),store.update_prompt_text(flow.key,'concurrent Web save'))
    row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    assert row.content=='concurrent Web save' and row.webManaged
    assert await flow.redis.get(store._redis_key(flow.key))==row.content

async def test_running_worker_refreshes_agent_override_without_any_broadcast(flow):
    from datetime import UTC,datetime
    tag=uuid4().hex; user='c01-user-'+tag; agent='c01-agent-'+tag
    now=datetime.now(UTC).replace(tzinfo=None)
    await flow.db.execute_raw('INSERT INTO users(id,username,updated_at) VALUES($1,$2,$3::timestamp)',user,user,now)
    await flow.db.execute_raw("INSERT INTO ai_agents(id,name,user_id,updated_at) VALUES($1,'synthetic',$2,$3::timestamp)",agent,user,now)
    await flow.db.agentconfigoverride.create(data={'agentId':agent,'localChatModel':'before-'+tag})
    code="""import asyncio,json,sys
from app.db import db
from app.services import runtime_config as r
async def main():
 await db.connect(); await r.ensure_loaded()
 print(json.dumps({'model':r.resolve_config_sync(%r).local_chat_model}),flush=True)
 await asyncio.to_thread(sys.stdin.readline)
 await r.ensure_loaded()
 print(json.dumps({'model':r.resolve_config_sync(%r).local_chat_model}),flush=True)
 await db.disconnect()
asyncio.run(main())
""" % (agent,agent)
    process=None
    try:
        env={**os.environ,'DATABASE_URL':flow.url,'DIRECT_DATABASE_URL':flow.url,'PYTHON_DOTENV_DISABLED':'1'}
        process=await asyncio.create_subprocess_exec(sys.executable,'-c',code,env=env,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        first=json.loads(await asyncio.wait_for(process.stdout.readline(),15)); assert first['model']=='before-'+tag
        await flow.db.agentconfigoverride.update(where={'agentId':agent},data={'localChatModel':'after-'+tag})
        process.stdin.write(b'next\n'); await process.stdin.drain()
        second=json.loads(await asyncio.wait_for(process.stdout.readline(),15)); assert second['model']=='after-'+tag
        await asyncio.wait_for(process.wait(),15); assert process.returncode==0
    finally:
        if process and process.returncode is None:
            process.kill(); await process.wait()
        await flow.db.agentconfigoverride.delete_many(where={'agentId':agent})
        await flow.db.aiagent.delete_many(where={'id':agent})
        await flow.db.user.delete_many(where={'id':user})

async def test_old_turn_rebuilding_model_cache_cannot_poison_new_turn(flow,monkeypatch):
    from app.services.llm import models
    from app.config import settings
    original=await flow.db.systemconfig.find_unique(where={'id':1})
    monkeypatch.setattr(settings,'chat_model','')
    monkeypatch.setattr(settings,'chat_provider','')
    monkeypatch.setattr(settings,'llm_provider','')
    built=[]
    def factory(provider,identifier):
        result=SimpleNamespace(provider=provider,identifier=identifier)
        built.append(result); return result
    monkeypatch.setattr(models,'build_chat_model',factory)
    models.get_chat_model.cache_clear()
    await flow.db.systemconfig.update(where={'id':1},data={'onlineModel':False,'localChatModel':'old-c01'})
    binding=await runtime.bind_agent_context(None)
    try:
        old=models.get_chat_model()
        await flow.db.systemconfig.update(where={'id':1},data={'localChatModel':'new-c01'})
        await runtime.load_caches(); runtime.invalidate_caches()
        # An in-flight turn rebuilding after API invalidation must keep its old model.
        assert models.get_chat_model().identifier=='old-c01'
    finally: runtime.reset_current_agent(binding)
    try:
        binding=await runtime.bind_agent_context(None)
        try: assert models.get_chat_model().identifier=='new-c01'
        finally: runtime.reset_current_agent(binding)
    finally:
        await flow.db.systemconfig.update(where={'id':1},data={'onlineModel':original.onlineModel,'localChatModel':original.localChatModel})
        models.get_chat_model.cache_clear()
        await runtime.load_caches()

async def test_recreated_row_has_a_new_cache_epoch(flow):
    old=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    await store.update_prompt_text(flow.key,'old row revision 2')
    await flow.db.prompttemplate.delete(where={'key':flow.key})
    await store.ensure_prompt_templates()
    current=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    assert current.id!=old.id and current.revision==1
    assert await flow.redis.get(store._redis_key(flow.key))==current.content
    assert await store._sync_cache(old) is False

async def test_empty_restore_id_has_no_write(flow):
    response=await flow.client.post('/admin-api/prompts/'+flow.key+'/restore-version',json={'version_id':''})
    assert response.status_code==404
    assert len(await store.list_prompt_versions(flow.key))==1

async def test_recreated_key_retains_web_ownership_from_old_row_history(flow,monkeypatch):
    await store.update_prompt_text(flow.key,'old Web customization')
    await flow.db.prompttemplate.delete(where={'key':flow.key})
    await store.ensure_prompt_templates()
    # A new row's bootstrap must not erase ownership recorded under the old row ID.
    monkeypatch.setattr(store,'PROMPT_DEFINITIONS',[replace(flow.d,default_text='new code default')])
    await store.ensure_prompt_templates()
    row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    assert row.webManaged is True
    assert row.content==flow.d.default_text and row.defaultContent=='new code default'
    assert all(v['change_type']!='code_sync' for v in await store.list_prompt_versions(flow.key))

async def test_nested_snapshot_reuse_requires_the_same_agent(flow):
    parent=await runtime.bind_agent_context(None)
    try:
        await store.update_prompt_text(flow.key,'new committed content')
        nested=await runtime.bind_agent_context(None,reuse_snapshot=True)
        try: assert await store.get_prompt_text(flow.key)==flow.d.default_text
        finally: runtime.reset_current_agent(nested)
        other=await runtime.bind_agent_context('different-synthetic-agent',reuse_snapshot=True)
        try: assert await store.get_prompt_text(flow.key)=='new committed content'
        finally: runtime.reset_current_agent(other)
        assert await store.get_prompt_text(flow.key)==flow.d.default_text
    finally: runtime.reset_current_agent(parent)

async def test_unauthenticated_prompt_write_has_no_side_effects(flow):
    flow.app.dependency_overrides.clear()
    response=await flow.client.put('/admin-api/prompts/'+flow.key,json={'content':'unauthorized'})
    assert response.status_code==401
    assert len(await store.list_prompt_versions(flow.key))==1

# Public Web publications deliberately do not reuse optimistic-lock revisions.
async def test_publication_default_and_disabled_have_one_content_identity(flow):
    initial=(await store.list_prompts())[0]
    assert initial['content_version_type']=='default' and initial['web_version'] is None
    assert initial['content_version_id']==(await store.list_prompt_versions(flow.key))[0]['id']
    disabled=await store.set_prompt_enabled(flow.key,False)
    assert disabled['content_version_type']=='default' and disabled['web_version'] is None
    assert disabled['content_version_id']==initial['content_version_id']
    assert disabled['is_enabled'] is False and disabled['revision']>initial['revision']

async def test_web_publications_count_content_actions_only(flow):
    first=await store.update_prompt_text(flow.key,'Web one')
    assert first['web_version']==1 and first['revision']==2
    disabled=await store.set_prompt_enabled(flow.key,False)
    noop=await store.update_prompt_text(flow.key,'Web one')
    assert disabled['web_version']==noop['web_version']==1
    assert disabled['content_version_id']==noop['content_version_id']==first['version_id']
    assert noop['version_id'] is None
    reset=await store.reset_prompt_text(flow.key)
    assert reset['web_version']==2 and reset['content_version_type']=='web'
    restored=await store.restore_prompt_version(flow.key,first['version_id'])
    assert restored['web_version']==3 and restored['content']=='Web one'
    assert restored['is_enabled'] is False
    history=await store.list_prompt_versions(flow.key)
    assert [v['web_version'] for v in history if v['web_version'] is not None]==[3,2,1]
    assert next(v for v in history if v['change_type']=='disable')['web_version'] is None

async def test_concurrent_publications_are_contiguous_and_match_current_content(flow):
    results=await asyncio.gather(*(store.update_prompt_text(flow.key,str(i)) for i in range(3)))
    assert sorted(x['web_version'] for x in results)==[1,2,3]
    latest=max(results,key=lambda x:x['web_version'])
    row=(await store.list_prompts())[0]
    assert row['content']==latest['content'] and row['web_version']==3
    assert row['content_version_id']==latest['version_id']

async def test_stale_guard_does_not_consume_publication_number(flow):
    await store.update_prompt_text(flow.key,'one',expected_revision=1)
    with pytest.raises(store.PromptUpdateConflictError):
        await store.update_prompt_text(flow.key,'stale',expected_revision=1,publish_version=True)
    second=await store.update_prompt_text(flow.key,'two',expected_revision=2)
    assert second['web_version']==2

async def test_mapping_failure_rolls_back_counter_content_audit_and_cache(flow):
    await flow.db.execute_raw("CREATE FUNCTION c01_fail_publication() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.prompt_key LIKE 'test.c01.%' THEN RAISE EXCEPTION 'synthetic mapping failure'; END IF; RETURN NEW; END $$")
    await flow.db.execute_raw('CREATE TRIGGER c01_fail_publication BEFORE INSERT ON prompt_publication_versions FOR EACH ROW EXECUTE FUNCTION c01_fail_publication()')
    try:
        with pytest.raises(Exception): await store.update_prompt_text(flow.key,'must roll back')
        row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
        assert row.content==flow.d.default_text and row.revision==1
        assert await flow.db.promptpublicationcounter.find_unique(where={'promptKey':flow.key}) is None
        assert len(await store.list_prompt_versions(flow.key))==1
        assert await flow.redis.get(store._redis_key(flow.key))==row.content
    finally:
        await flow.db.execute_raw('DROP TRIGGER c01_fail_publication ON prompt_publication_versions')
        await flow.db.execute_raw('DROP FUNCTION c01_fail_publication()')
    assert (await store.update_prompt_text(flow.key,'first success'))['web_version']==1

async def test_publication_numbers_survive_history_and_template_deletion(flow):
    await store.update_prompt_text(flow.key,'one')
    await flow.db.prompttemplateversion.delete_many(where={'promptKey':flow.key})
    assert await flow.db.promptpublicationversion.count(where={'promptKey':flow.key})==0
    await flow.db.prompttemplate.delete(where={'key':flow.key})
    recreated=await store.update_prompt_text(flow.key,'two')
    assert recreated['web_version']==2

async def test_publication_pagination_and_enable_revision_keep_current_content_id(flow):
    for i in range(22): await store.update_prompt_text(flow.key,'version '+str(i))
    published=(await store.list_prompts())[0]
    disabled=await store.set_prompt_enabled(flow.key,False)
    history=await store.list_prompt_versions(flow.key,limit=5)
    assert len(history)==5 and history[0]['change_type']=='disable'
    assert [v['web_version'] for v in history[1:]]==[22,21,20,19]
    assert disabled['content_version_id']==published['content_version_id']
    assert disabled['web_version']==22 and disabled['revision']==24

@pytest.mark.parametrize('legacy',['unknown','code_sync_after_web','missing_history'])
async def test_uncertain_history_can_publish_same_content_without_changing_enable(flow,legacy):
    if legacy=='code_sync_after_web':
        await store.update_prompt_text(flow.key,'historical Web')
        row=await flow.db.prompttemplate.update(where={'key':flow.key},data={'content':flow.d.default_text})
        await store._version(flow.db,row,'code_sync','default')
    elif legacy=='unknown':
        row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
        await store._version(flow.db,row,'unknown','db')
    else:
        await flow.db.prompttemplateversion.delete_many(where={'promptKey':flow.key})
    await store.set_prompt_enabled(flow.key,False)
    before=(await store.list_prompts())[0]
    assert before['content_version_type']=='unverified' and before['web_version'] is None
    response=await flow.client.put('/admin-api/prompts/'+flow.key,json={
        'content':before['content'],'publish_version':True,'expected_revision':before['revision'],
        'expected_updated_at':before['updated_at']})
    assert response.status_code==200,response.text
    saved=response.json()
    assert saved['content']==before['content'] and saved['is_enabled'] is False
    assert saved['content_version_type']=='web' and saved['web_version']==(2 if legacy=='code_sync_after_web' else 1)
    assert saved['content_version_id']==saved['version_id']
    latest=(await store.list_prompt_versions(flow.key))[0]
    assert latest['change_type']=='manual_save' and latest['web_version']==saved['web_version']
    assert await flow.redis.get(store._redis_key(flow.key))==before['content']
    assert (await store.list_prompts())[0]['content_version_id']==saved['version_id']

async def test_content_restore_at_same_content_is_still_a_new_web_publication(flow):
    first=await store.update_prompt_text(flow.key,'same')
    second=await store.restore_prompt_version(flow.key,first['version_id'])
    assert second['content']==first['content'] and second['web_version']==2

async def test_old_client_audit_insert_gets_publication_number(flow):
    row=await flow.db.prompttemplate.find_unique(where={'key':flow.key})
    await flow.db.execute_raw("INSERT INTO prompt_template_versions(id,prompt_id,prompt_key,content,source,change_type,created_at) VALUES($1,$2,$3,$4,'db','manual_save',NOW())",uuid4().hex,row.id,row.key,row.content)
    listed=(await store.list_prompts())[0]
    assert listed['content_version_type']=='web' and listed['web_version']==1
    assert listed['revision']==1

async def test_publication_api_requires_admin(flow):
    flow.app.dependency_overrides.clear()
    response=await flow.client.put('/admin-api/prompts/'+flow.key,json={'content':flow.d.default_text,'publish_version':True})
    assert response.status_code==401
    assert await flow.db.promptpublicationcounter.find_unique(where={'promptKey':flow.key}) is None

async def test_same_content_publication_preserves_legacy_surrounding_whitespace(flow):
    raw='\n  historical text {message}\n'
    await flow.db.prompttemplate.update(where={'key':flow.key},data={'content':raw,'webManaged':True})
    result=await store.update_prompt_text(flow.key,raw,publish_version=True)
    assert result['content']==raw and result['web_version']==1
    assert await flow.redis.get(store._redis_key(flow.key))==raw
    edited=await store.update_prompt_text(flow.key,'  changed text  ',publish_version=True)
    assert edited['content']=='changed text' and edited['web_version']==2

async def test_alignment_preflight_apply_and_rerun_are_guarded_and_idempotent(flow,monkeypatch):
    import hashlib
    from scripts import align_prompt_publication_versions as script
    monkeypatch.setattr(script,'db',flow.db)
    monkeypatch.setattr(script,'get_redis',AsyncMock(return_value=flow.redis))
    await flow.db.prompttemplate.update(where={'key':flow.key},data={'webManaged':True})
    entry={'key':flow.key,'expected_sha256':hashlib.sha256(flow.d.default_text.encode()).hexdigest(),'allowed_fields':['message']}
    assert (await script.align([entry]))[0]['action']=='preflight'
    assert len(await store.list_prompt_versions(flow.key))==1
    receipt=(await script.align([entry],apply=True))[0]
    assert receipt['action']=='published' and receipt['web_version']==1
    assert (await script.align([entry],apply=True))[0]['action']=='already_published'
    assert len(await store.list_prompt_versions(flow.key))==2
    entry['expected_sha256']='stale'
    with pytest.raises(ValueError,match='reviewed'):await script.align([entry],apply=True)
    assert len(await store.list_prompt_versions(flow.key))==2
