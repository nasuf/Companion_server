"""Owned PostgreSQL clone writes, source scopes and atomic workspace switching."""
import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from prisma.errors import DataError, RawQueryError

from app.services.agent_template import clone as c
from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME
from app.services.memory import evidence_read as read
from app.services.memory.evidence import content_version
from app.services.memory.storage import entity_repo
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
async def cloning(origins, monkeypatch):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    monkeypatch.setattr(c,"db",db);monkeypatch.setattr(entity_repo,"db",db)
    effects={name: (MagicMock() if name=="_dispatch_day_one_schedule" else AsyncMock()) for name in
             ["assign_random_voice","finalize_archived_workspaces","_dispatch_day_one_schedule"]}
    for name,value in effects.items():monkeypatch.setattr(c,name,value)
    await db.user.update(where={"id":users[-1]},data={"username":TEMPLATE_SYSTEM_USERNAME})
    for agent in agents:await db.aiagent.update(where={"id":agent},data={"status":"active"})
    await db.aiagent.update(where={"id":agents[-1]},data={"city":"Synthetic template city"})
    source_ids=[]
    for level,importance in [(1,.9),(2,.7),(3,.3)]:
        m=await db.aimemory.create(data={"userId":users[-1],"workspaceId":spaces[-1],
            "content":f"Synthetic template level {level}","level":level,"importance":importance,
            "provenance":"profile_seed","mainCategory":"生活","subCategory":"工作",
            "currentScore":.11,"valueUpdatedAt":datetime.now(UTC)-timedelta(days=180)})
        source_ids.append(m.id)
    await db.execute_raw("INSERT INTO memory_embeddings(memory_id,embedding) SELECT x,$2::extensions.vector FROM unnest($1::text[]) x",
        source_ids,"["+",".join(["0.01"]*1024)+"]")
    async def clone(**kwargs):return await c.clone_template_agent_for_user(uid,agents[-1],**kwargs)
    async def inspect(workspace,memory_id):return await read.memory_evidence_detail(
        user_id=uid,workspace_id=workspace,side="ai",memory_id=memory_id)
    try:yield origins,clone,inspect,effects,source_ids
    finally:
        # The parent fixture owns these users; clean new clone resources too.
        all_convs=await db.conversation.find_many(where={"userId":{"in":users}})
        await db.message.delete_many(where={"conversationId":{"in":[r.id for r in all_convs]}})
        await db.conversation.delete_many(where={"userId":{"in":users}})
        for model in (db.usermemory,db.aimemory):await model.delete_many(where={"userId":{"in":users}})
        await db.chatworkspace.delete_many(where={"userId":{"in":users}})
        await db.aiagent.delete_many(where={"userId":{"in":users}})


async def test_clone_copies_all_tiers_vectors_and_each_actual_source_version(cloning):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    await db.aimemory.create(data={"userId":users[-1],"workspaceId":spaces[-1],"content":"Synthetic archived","isArchived":True})
    await db.aimemory.create(data={"userId":uid,"workspaceId":spaces[-1],"content":"Synthetic wrong owner"})
    await db.usermemory.create(data={"userId":users[-1],"workspaceId":spaces[-1],"content":"Synthetic user private"})
    agent,space,conv=await clone()
    rows=await db.aimemory.find_many(where={"workspaceId":space.id})
    assert sorted((r.level,r.importance) for r in rows)==[(1,.9),(2,.7),(3,.3)]
    assert agent.city=="Synthetic template city" and conv.workspaceId==space.id
    assert (await db.aiagent.find_unique(where={"id":agent.id})).sourceTemplateId==agents[-1]
    assert all(r.currentScore is None for r in rows)
    assert all(r.valueUpdatedAt is not None for r in rows)
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",[r.id for r in rows]))[0]['n']==3
    assert await db.memorychangelog.count(where={"workspaceId":space.id,"operation":"insert"})==3
    for row in rows:
        result=await inspect(space.id,row.id);item=result['items'][0]
        assert result['state']=='linked' and item['relation']=='template_copy'
        assert item['source_ref'] in source_ids and item['source_kind']=='memory' and item['source_side']=='ai'
        assert item['source_version']==content_version(row.content) and item['current_content']
        assert item['availability']=='available' and item['extractor_version']=='template-clone-v1'
        assert "Synthetic template level" not in str(result)
    assert (await db.chatworkspace.find_unique(where={"id":wid})).status=='archived'
    assert (await db.conversation.find_unique(where={"id":convs[0]})).isDeleted
    assert (await db.aiagent.find_unique(where={"id":agents[0]})).status=='archived'
    for value in effects.values():assert value.call_count==1
    assert effects['finalize_archived_workspaces'].await_args.args[0][0]['workspace_id']==wid


@pytest.mark.parametrize('failure',['agent','pointer','workspace','memory','vector','audit','evidence','conversation'])
async def test_any_primary_write_failure_preserves_original_agent_workspace_and_conversation(cloning,failure):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    table={'agent':'ai_agents','pointer':'ai_agents','workspace':'chat_workspaces','memory':'memories_ai',
           'vector':'memory_embeddings','audit':'memory_changelogs','evidence':'memory_evidence_links','conversation':'conversations'}[failure]
    predicate=(f"EXISTS(SELECT 1 FROM memories_ai WHERE id=NEW.memory_id AND user_id='{uid}')" if failure=='vector'
               else f"NEW.user_id='{uid}'")
    if failure=='pointer':predicate+=' AND NEW.source_template_id IS NOT NULL'
    name='fail_clone_'+uuid4().hex;event='UPDATE' if failure=='pointer' else 'INSERT'
    await db.execute_raw(f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF {predicate} THEN RAISE EXCEPTION 'synthetic clone failure'; END IF; RETURN NEW; END $$")
    await db.execute_raw(f"CREATE TRIGGER {name} BEFORE {event} ON {table} FOR EACH ROW EXECUTE FUNCTION {name}()")
    try:
        with pytest.raises((DataError,RawQueryError),match='synthetic clone failure'):await clone()
    finally:
        await db.execute_raw(f"DROP TRIGGER {name} ON {table}");await db.execute_raw(f"DROP FUNCTION {name}()")
    assert (await db.chatworkspace.find_unique(where={"id":wid})).status=='active'
    assert not (await db.conversation.find_unique(where={"id":convs[0]})).isDeleted
    assert (await db.aiagent.find_unique(where={"id":agents[0]})).status=='active'
    assert await db.aiagent.count(where={"userId":uid})==2
    assert await db.aimemory.count(where={"userId":uid})==1
    assert await db.memorychangelog.count(where={"userId":uid})==0
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_evidence_links WHERE user_id=$1",uid))[0]['n']==0
    for value in effects.values():assert value.call_count==0


@pytest.mark.parametrize('invalid',['missing','disabled','archived','private_owner','workspace_owner','workspace_inactive','empty','over_limit','missing_user','inactive_user','system_target'])
async def test_invalid_template_or_target_never_creates_a_clone(cloning,invalid,monkeypatch):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    if invalid=='missing':pass
    if invalid=='disabled':await db.execute_raw("UPDATE ai_agents SET template_enabled=false WHERE id=$1",agents[-1])
    if invalid=='archived':await db.aiagent.update(where={"id":agents[-1]},data={"status":"archived"})
    if invalid=='private_owner':await db.user.update(where={"id":users[-1]},data={"username":"synthetic-private-"+uuid4().hex})
    if invalid=='workspace_owner':await db.chatworkspace.update(where={"id":spaces[-1]},data={"userId":uid,"status":"archived"})
    if invalid=='workspace_inactive':await db.chatworkspace.update(where={"id":spaces[-1]},data={"status":"archived"})
    if invalid=='empty':await db.aimemory.delete_many(where={"workspaceId":spaces[-1]})
    if invalid=='over_limit':monkeypatch.setattr(c,'MAX_TEMPLATE_MEMORIES',2)
    if invalid=='inactive_user':await db.user.update(where={"id":uid},data={"status":"disabled"})
    with pytest.raises(ValueError):
        if invalid=='missing':await c.clone_template_agent_for_user(uid,'missing')
        elif invalid=='missing_user':await c.clone_template_agent_for_user('missing',agents[-1])
        elif invalid=='system_target':await c.clone_template_agent_for_user(users[-1],agents[-1])
        else:await clone()
    assert await db.aiagent.count(where={"userId":uid})==2
    assert (await db.chatworkspace.find_unique(where={"id":wid})).status=='active'
    for value in effects.values():assert value.call_count==0


async def test_default_signup_rechecks_under_sql_lock_even_without_redis(cloning):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    with pytest.raises(ValueError,match='already_provisioned'):await clone(only_if_missing=True)
    await db.chatworkspace.update(where={"id":wid},data={"status":"archived"})
    await db.aiagent.update(where={"id":agents[0]},data={"status":"provisioning"})
    with pytest.raises(ValueError,match='already_provisioned'):await clone(only_if_missing=True)
    await db.aiagent.update(where={"id":agents[0]},data={"status":"archived"})
    results=await asyncio.gather(clone(only_if_missing=True),clone(only_if_missing=True),return_exceptions=True)
    assert sum(isinstance(r,tuple) for r in results)==1
    assert sum(isinstance(r,ValueError) for r in results)==1
    assert await db.chatworkspace.count(where={"userId":uid,"status":"active"})==1


async def test_changed_deleted_and_rebound_template_sources_are_distinct(cloning):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    agent,space,conv=await clone();row=await db.aimemory.find_first(where={"workspaceId":space.id})
    item=(await inspect(space.id,row.id))['items'][0];source=item['source_ref'];fingerprint=item['source_version']
    await db.aimemory.update(where={"id":source},data={"content":"Synthetic revised source"})
    assert (await inspect(space.id,row.id))['items'][0]['availability']=='changed'
    await db.aiagent.update(where={"id":agents[-1]},data={"userId":uid})
    hidden=(await inspect(space.id,row.id))['items'][0]
    assert hidden['availability']=='unavailable' and hidden['source_ref'] is None
    await db.aiagent.update(where={"id":agents[-1]},data={"userId":users[-1]})
    await db.aimemory.delete(where={"id":source})
    deleted=(await inspect(space.id,row.id))['items'][0]
    assert deleted['availability']=='deleted' and deleted['source_ref'] is None and deleted['source_version']==fingerprint
    assert await db.aimemory.find_unique(where={"id":row.id}) is not None
    await db.aimemory.update(where={"id":row.id},data={"content":"Synthetic edited clone"})
    assert (await inspect(space.id,row.id))['state']=='current_unlinked'


async def test_clone_waits_for_persona_force_replacement_lock(cloning):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    async with db.tx() as tx:
        await tx.query_raw("SELECT pg_advisory_xact_lock(hashtextextended($1,0))::text AS held",'persona-init:'+spaces[-1])
        task=asyncio.create_task(clone())
        await asyncio.sleep(.15);assert not task.done()
    assert (await asyncio.wait_for(task,10))[1].userId==uid


async def test_post_commit_failures_do_not_undo_a_complete_clone(cloning,monkeypatch):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    for name in ['assign_random_voice','finalize_archived_workspaces']:effects[name].side_effect=RuntimeError('synthetic post commit failure')
    monkeypatch.setattr(c,'_clone_memory_entities',AsyncMock(side_effect=RuntimeError('synthetic entity failure')))
    agent,space,conv=await clone()
    assert (await db.chatworkspace.find_unique(where={"id":space.id})).status=='active'
    assert await db.aimemory.count(where={"workspaceId":space.id})==3
    effects['_dispatch_day_one_schedule'].assert_called_once()


@pytest.mark.parametrize('size,budget',[(250,15),(1000,25)])
async def test_bounded_memory_clone_completes_with_every_origin(cloning,record_property,size,budget):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    ids=[uuid4().hex for _ in range(size-3)]
    await db.aimemory.create_many(data=[{'id':mid,'userId':users[-1],'workspaceId':spaces[-1],
        'content':'Synthetic template batch '+str(i),'level':2,'importance':.7} for i,mid in enumerate(ids)])
    await db.execute_raw("INSERT INTO memory_embeddings(memory_id,embedding) SELECT x,$2::extensions.vector FROM unnest($1::text[]) x",ids,'['+','.join(['.01']*1024)+']')
    start=time.monotonic();agent,space,conv=await clone();elapsed=time.monotonic()-start
    record_property(f'clone_{size}_elapsed_seconds',elapsed)
    assert elapsed<budget
    assert await db.aimemory.count(where={"workspaceId":space.id})==size
    assert (await db.query_raw('SELECT count(*)::int n FROM memory_evidence_links WHERE workspace_id=$1',space.id))[0]['n']==size


async def test_existing_workspace_owner_mismatch_rolls_back_without_runtime_cleanup(cloning):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    await db.aiagent.update(where={'id':agents[0]},data={'userId':users[1]})
    with pytest.raises(ValueError,match='clone_existing_workspace_scope_mismatch'):await clone()
    assert (await db.chatworkspace.find_unique(where={'id':wid})).status=='active'
    assert not (await db.conversation.find_unique(where={'id':convs[0]})).isDeleted
    for value in effects.values():assert value.call_count==0


async def test_missing_template_vector_keeps_memory_and_origin_without_inventing_vector(cloning):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    await db.execute_raw('DELETE FROM memory_embeddings WHERE memory_id=$1',source_ids[0])
    agent,space,conv=await clone()
    rows=await db.aimemory.find_many(where={'workspaceId':space.id})
    assert len(rows)==3
    assert (await db.query_raw('SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])',[r.id for r in rows]))[0]['n']==2
    for row in rows:assert (await inspect(space.id,row.id))['state']=='linked'


async def test_template_entity_copy_is_scoped_and_ai_only(cloning):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    for side,name in [('ai','Synthetic template pet'),('user','Synthetic user private entity')]:
        await entity_repo.record_entities_for_memory(memory_id=source_ids[0],memory_source=side,
            user_id=users[-1],workspace_id=spaces[-1],entities=[{'name':name,'type':'pet','aliases':['Synthetic alias']}])
    agent,space,conv=await clone()
    rows=await db.query_raw('SELECT canonical_name FROM memory_entities WHERE workspace_id=$1',space.id)
    assert [r['canonical_name'] for r in rows]==['Synthetic template pet']


async def test_oversized_template_rows_are_still_copied_verbatim_with_warning(cloning,caplog):
    o,clone,inspect,effects,source_ids=cloning;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    text='合成超长模板内容'*500
    await db.aimemory.update(where={'id':source_ids[0]},data={'content':text})
    agent,space,conv=await clone()
    assert await db.aimemory.count(where={'workspaceId':space.id,'content':text})==1
    assert 'injection limit' in caplog.text
