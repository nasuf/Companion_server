"""Knowledge publication: real owned PostgreSQL, no models or production data."""
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma.errors import DataError, RawQueryError

from app.services.agent_template import knowledge as k
from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME
from app.services.memory.evidence import content_version
from app.services.memory import evidence_read as read
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
async def publishing(origins, monkeypatch):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    monkeypatch.setattr(k,"db",db)
    cache=AsyncMock();monkeypatch.setattr(k,"_bust_knowledge_rows_cache",cache)
    await db.user.update(where={"id":users[-1]},data={"username":TEMPLATE_SYSTEM_USERNAME})
    for agent in agents:await db.aiagent.update(where={"id":agent},data={"status":"active"})
    await db.execute_raw("UPDATE ai_agents SET source_template_id=$2 WHERE id=$1",agents[0],agents[-1])
    rows=[]
    for i in range(2):
        rows.append(await db.aimemory.create(data={"userId":users[-1],"workspaceId":spaces[-1],
            "content":f"Synthetic published knowledge {i}","level":1,"importance":.86,
            "mainCategory":"生活","subCategory":"工作","provenance":"knowledge_seed",
            "currentScore":.1,"valueUpdatedAt":datetime.now(UTC)-timedelta(days=100)}))
    await db.execute_raw("INSERT INTO memory_embeddings(memory_id,embedding) SELECT x,$2::extensions.vector FROM unnest($1::text[]) x",
        [r.id for r in rows],"["+",".join(["0.01"]*1024)+"]")
    target_scope=(await k._resolve_target_agents(agents[-1],[agents[0]]))[0]
    async def publish(**changes):
        args={"template_rows":rows,"template_agent_id":agents[-1],"agent_id":agents[0],"user_id":uid,"target_scope":target_scope}
        return await k._sync_agent(**{**args,**changes})
    async def inspect(memory_id):
        return await read.memory_evidence_detail(user_id=uid,workspace_id=wid,side="ai",memory_id=memory_id)
    return origins,rows,publish,inspect,cache


async def test_publish_pins_each_version_vector_audit_clock_and_replay(publishing):
    o,sources,publish,inspect,cache=publishing;db,uid,wid,mid,*_=o
    before=datetime.now(UTC)
    assert await publish()==2
    copied=await db.aimemory.find_many(where={"workspaceId":wid,"provenance":"knowledge_seed"})
    assert len(copied)==2
    for row in copied:
        assert row.level==1 and row.importance==.86 and row.currentScore is None
        assert row.valueUpdatedAt>=before
        result=await inspect(row.id);item=result['items'][0]
        assert result['state']=='linked' and item['relation']=='template_copy'
        assert item['extractor_version']=='knowledge-sync-v1' and item['source_side']=='ai'
        assert item['availability']=='available' and item['current_content']
        assert item['source_version']==content_version(row.content)
        assert item['source_ref'] in {r.id for r in sources}
        assert row.content not in str(result) and uid not in str(item)
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",[r.id for r in copied]))[0]['n']==2
    assert await db.memorychangelog.count(where={"workspaceId":wid,"operation":"knowledge_sync"})==2
    assert await publish()==0
    assert await db.aimemory.count(where={"workspaceId":wid})==3
    assert (await inspect(mid))['state']=='historical_unknown'
    cache.assert_awaited_once_with(wid)


async def test_full_sync_watermark_is_persisted_and_cannot_move_backwards(publishing, monkeypatch):
    o, sources, publish, inspect, cache = publishing
    db, uid, wid, mid, msg, bind, detail, users, agents, spaces, convs = o
    writes = []

    async def capture(_tid, progress):
        writes.append(dict(progress))

    monkeypatch.setattr(k, "_write_progress", capture)
    targets = await k._resolve_target_agents(agents[-1], [agents[0]])
    await k._run_sync_locked(template_agent_id=agents[-1], template_rows=sources,
                             targets=targets, full_mode=True)
    final = writes[-1]
    assert final["status"] == "done" and final["watermark_advanced"]
    assert final["copied_memories"] == 2
    watermark = max(row.createdAt for row in sources)
    await k._advance_watermark(agents[-1], watermark - timedelta(days=1))
    actual = (await db.query_raw("SELECT knowledge_synced_at AS mark FROM ai_agents WHERE id=$1", agents[-1]))[0]["mark"]
    assert datetime.fromisoformat(actual.replace("Z", "+00:00")) == watermark


@pytest.mark.parametrize('failure',['memory','vector','audit','evidence'])
async def test_write_failure_rolls_back_every_target_output_then_retry_succeeds(publishing,failure):
    o,sources,publish,inspect,cache=publishing;db,uid,wid,mid,*_=o
    table={'memory':'memories_ai','vector':'memory_embeddings','audit':'memory_changelogs','evidence':'memory_evidence_links'}[failure]
    predicate=(f"EXISTS(SELECT 1 FROM memories_ai WHERE id=NEW.memory_id AND user_id='{uid}')" if failure=='vector' else f"NEW.user_id='{uid}'")
    name='fail_knowledge_'+uuid4().hex
    await db.execute_raw(f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF {predicate} THEN RAISE EXCEPTION 'synthetic knowledge failure'; END IF; RETURN NEW; END $$")
    await db.execute_raw(f"CREATE TRIGGER {name} BEFORE INSERT ON {table} FOR EACH ROW EXECUTE FUNCTION {name}()")
    try:
        with pytest.raises((DataError,RawQueryError),match='synthetic knowledge failure'):await publish()
    finally:
        await db.execute_raw(f"DROP TRIGGER {name} ON {table}");await db.execute_raw(f"DROP FUNCTION {name}()")
    assert await db.aimemory.count(where={"workspaceId":wid})==1
    assert await db.memorychangelog.count(where={"workspaceId":wid})==0
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_evidence_links WHERE workspace_id=$1",wid))[0]['n']==0
    cache.assert_not_awaited()
    assert await publish()==2


@pytest.mark.parametrize('invalid',['content','deleted','archived','owner','workspace','provenance'])
async def test_stale_or_foreign_source_snapshot_never_copies(publishing,invalid):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    if invalid=='deleted':await db.aimemory.delete(where={"id":rows[0].id})
    else:
        changes={'content':{'content':'Synthetic revised'},'archived':{'isArchived':True},
                 'owner':{'userId':uid},'workspace':{'workspaceId':wid},'provenance':{'provenance':'profile_seed'}}
        await db.aimemory.update(where={"id":rows[0].id},data=changes[invalid])
    with pytest.raises(ValueError,match='knowledge_source_'):await publish()
    assert await db.aimemory.count(where={"workspaceId":wid,"provenance":"knowledge_seed"})==(1 if invalid=='workspace' else 0)
    cache.assert_not_awaited()


@pytest.mark.parametrize('invalid',['template_owner','template_archived','template_workspace','template_workspace_owner',
    'target_owner','target_archived','target_workspace','target_workspace_owner','target_template','target_user_disabled','template_missing'])
async def test_live_scope_changes_reject_publication(publishing,invalid):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    changes={}
    if invalid=='template_owner':await db.user.update(where={"id":users[-1]},data={"username":"synthetic-private-"+uuid4().hex})
    if invalid=='template_archived':await db.aiagent.update(where={"id":agents[-1]},data={"status":"archived"})
    if invalid=='template_workspace':await db.chatworkspace.update(where={"id":spaces[-1]},data={"status":"archived"})
    if invalid in {'template_workspace_owner','target_workspace_owner'}:
        foreign=await db.user.create(data={"username":"knowledge-foreign-"+uuid4().hex});users.append(foreign.id)
        await db.chatworkspace.update(where={"id":spaces[-1] if invalid=='template_workspace_owner' else wid},data={"userId":foreign.id})
    if invalid=='target_owner':changes['user_id']=users[-1]
    if invalid=='target_archived':await db.aiagent.update(where={"id":agents[0]},data={"archivedAt":datetime.now(UTC)})
    if invalid=='target_workspace':await db.chatworkspace.update(where={"id":wid},data={"status":"archived"})
    if invalid=='target_template':await db.execute_raw("UPDATE ai_agents SET source_template_id=NULL WHERE id=$1",agents[0])
    if invalid=='target_user_disabled':await db.user.update(where={"id":uid},data={"status":"disabled"})
    if invalid=='template_missing':changes['template_agent_id']='missing'
    with pytest.raises(ValueError,match='scope_invalid'):await publish(**changes)
    assert await db.aimemory.count(where={"workspaceId":wid})==1
    cache.assert_not_awaited()


async def test_missing_source_vector_is_explicitly_preserved_not_a_copy_error(publishing):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,*_=o
    await db.execute_raw("DELETE FROM memory_embeddings WHERE memory_id=$1",rows[0].id)
    assert await publish()==2
    copied=await db.aimemory.find_many(where={"workspaceId":wid,"provenance":"knowledge_seed"})
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",[r.id for r in copied]))[0]['n']==1
    assert all([(await inspect(r.id))['state']=='linked' for r in copied])


async def test_parallel_publish_is_idempotent_and_archived_content_is_not_resurrected(publishing):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,*_=o
    assert sorted(await asyncio.wait_for(asyncio.gather(publish(),publish()),15))==[0,2]
    row=await db.aimemory.find_first(where={"workspaceId":wid,"provenance":"knowledge_seed"})
    await db.aimemory.update(where={"id":row.id},data={"isArchived":True})
    assert await publish()==0
    assert await db.aimemory.count(where={"workspaceId":wid,"provenance":"knowledge_seed"})==2


async def test_duplicate_text_in_source_batch_copies_once_and_disabled_enrollment_is_allowed(publishing):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,mid,msg,bind,detail,users,agents,*_=o
    await db.aimemory.update(where={"id":rows[1].id},data={"content":rows[0].content})
    rows[1].content=rows[0].content
    await db.execute_raw("UPDATE ai_agents SET template_enabled=false WHERE id=$1",agents[-1])
    assert await publish()==1  # Enrollment is separate from publishing to existing clones.
    assert await publish()==0


async def test_source_edit_delete_and_owner_rebind_are_readonly_and_redacted(publishing):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,*_=o
    assert await publish()==2
    target=await db.aimemory.find_first(where={"workspaceId":wid,"content":rows[0].content})
    await db.aimemory.update(where={"id":rows[0].id},data={"content":"Synthetic revised"})
    assert (await inspect(target.id))['items'][0]['availability']=='changed'
    await db.aimemory.update(where={"id":rows[0].id},data={"userId":uid})
    hidden=(await inspect(target.id))['items'][0]
    assert hidden['availability']=='unavailable' and hidden['source_ref'] is None
    await db.aimemory.delete(where={"id":rows[0].id})
    deleted=(await inspect(target.id))['items'][0]
    assert deleted['availability']=='deleted' and deleted['source_ref'] is None


@pytest.mark.parametrize('case',['empty','over_limit','duplicate_ids'])
async def test_invalid_batches_fail_before_writes(publishing,monkeypatch,case):
    o,rows,publish,inspect,cache=publishing
    if case=='over_limit':monkeypatch.setattr(k,'MAX_KNOWLEDGE_SYNC_MEMORIES',1)
    data=[] if case=='empty' else [rows[0],rows[0]] if case=='duplicate_ids' else rows
    with pytest.raises(ValueError):await publish(template_rows=data)
    cache.assert_not_awaited()


async def test_scope_recheck_after_waiting_for_persona_replacement(publishing):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=o
    task=None
    try:
        async with db.tx(timeout=timedelta(seconds=10)) as tx:
            await tx.query_raw("SELECT pg_advisory_xact_lock(hashtextextended($1,0))::text AS held","persona-init:"+spaces[-1])
            task=asyncio.create_task(publish())
            async with asyncio.timeout(5):
                while not await db.query_raw("SELECT 1 FROM pg_stat_activity WHERE wait_event='advisory' AND query LIKE '%pg_advisory_xact_lock_shared%'"):
                    if task.done():await task
                    await asyncio.sleep(.02)
            await tx.chatworkspace.update(where={"id":spaces[-1]},data={"status":"archived"})
        with pytest.raises(ValueError,match='template_scope_changed'):await asyncio.wait_for(task,5)
        cache.assert_not_awaited()
    finally:
        if task and not task.done():
            task.cancel();await asyncio.gather(task,return_exceptions=True)


@pytest.mark.parametrize('resource',['users','ai_agents','chat_workspaces'])
async def test_reactivated_scope_generation_cannot_inherit_queued_publication(publishing,resource):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,mid,msg,bind,detail,users,agents,*_=o
    ident={'users':uid,'ai_agents':agents[0],'chat_workspaces':wid}[resource]
    await db.execute_raw(f"UPDATE {resource} SET execution_generation=gen_random_uuid()::text WHERE id=$1",ident)
    with pytest.raises(ValueError,match='target_scope_expired'):await publish()
    assert await db.aimemory.count(where={"workspaceId":wid})==1
    cache.assert_not_awaited()


async def test_upper_bound_batch_copies_exactly_one_thousand_origins(publishing):
    import time
    o,rows,publish,inspect,cache=publishing;db,uid,wid,mid,msg,bind,detail,users,agents,spaces,*_=o
    await db.aimemory.create_many(data=[{'userId':users[-1],'workspaceId':spaces[-1],
        'content':f'Synthetic capacity {i}','provenance':'knowledge_seed','level':1,'importance':.86}
        for i in range(998)])
    rows=await db.aimemory.find_many(where={'workspaceId':spaces[-1],'provenance':'knowledge_seed'})
    started=time.monotonic()
    assert await publish(template_rows=rows)==1000
    assert time.monotonic()-started<30
    assert (await db.query_raw('SELECT count(*)::int n FROM memory_evidence_links WHERE workspace_id=$1',wid))[0]['n']==1000
    assert await publish(template_rows=rows)==0


async def test_resolve_all_and_selected_targets_share_server_bound_scope(publishing):
    o,rows,publish,inspect,cache=publishing;db,uid,wid,mid,msg,bind,detail,users,agents,*_=o
    all_targets=await k._resolve_target_agents(agents[-1],None)
    selected=await k._resolve_target_agents(agents[-1],[agents[0],agents[0]])
    assert all_targets==selected and len(selected)==1
    assert selected[0]['workspace_id']==wid
    assert all(selected[0][f'{v}_generation'] for v in ['user','agent','workspace'])
    with pytest.raises(ValueError,match='有效关联'):
        await k._resolve_target_agents(agents[-1],[agents[1]])
