"""Actual initialization writes, source isolation and atomic replacement on PG."""
import asyncio
from contextlib import asynccontextmanager
import json
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma.errors import DataError, RawQueryError

from app.services import life_story as life
from app.services.memory import evidence as e, evidence_read as read
from app.services.memory.profile_evidence import (
    ProfileOrigin, prepare_profile_origin, persist_profile_origin,
)
from tests.test_memory_evidence_postgres import origins


def sample(content="Synthetic persona", *, embedding=True):
    return {"content": content, "main_category": "生活", "sub_category": "工作",
            "type": "life", "importance": .7, **({"_embedding": [.01]*1024} if embedding else {})}


@pytest.fixture
async def persona(origins, monkeypatch):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    monkeypatch.setattr(life,"db",db)
    for name in ("ensure_connected","set_progress","bump_cache_version","_seed_persona_entities"):
        monkeypatch.setattr(life,name,AsyncMock())
    async def store(memories=None, origin=None, **kwargs):
        return await life.store_memories_batch(agents[0],uid,
            memories if memories is not None else [sample()],workspace_id=wid,origin=origin,**kwargs)
    async def inspect(mid):
        return await read.memory_evidence_detail(user_id=uid,workspace_id=wid,side="ai",memory_id=mid)
    return origins,store,inspect


@pytest.mark.parametrize("kind", ["generated_profile","imported_profile","provided_profile"])
async def test_profile_batch_has_one_frozen_private_origin_and_every_output_link(persona,kind):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    profile={"identity":{"location":"Synthetic home"}}
    inputs={"name":"Synthetic", "personality":{"warmth":75}}
    origin=prepare_profile_origin(profile,{"title":"Synthetic career"},inputs=inputs,kind=kind)
    profile["identity"]["location"]="Caller changed";inputs["name"]="Caller changed"
    ids=await store([sample("Synthetic one"),sample("Synthetic two")],origin,force=True)
    assert len(ids)==2
    rows=await db.query_raw("SELECT * FROM memory_profile_origins WHERE workspace_id=$1",wid)
    assert len(rows)==1 and json.loads(rows[0]["payload_text"])["profile"]["identity"]["location"]=="Synthetic home"
    assert rows[0]["input_status"]=="recorded"
    assert json.loads(rows[0]["payload_text"])["invocation_inputs"]["name"]=="Synthetic"
    for memory_id in ids:
        result=await inspect(memory_id);item=result["items"][0]
        assert result["state"]=="linked" and item["source_kind"]=="profile"
        assert item["availability"]=="recorded_snapshot" and item["source_status"]==kind
        assert item["profile"]=={"input_status":"recorded","format_version":"persona-profile-v1"}
        assert item["source_version"]==origin.version and item["current_content"]
        assert all(private not in str(result) for private in ("Synthetic home","Synthetic career","warmth","payload_text"))
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",ids))[0]["n"]==2
    assert await db.memorychangelog.count(where={"userId":uid,"operation":"insert"})==2
    assert await db.usermemory.find_unique(where={"id":mid}) is not None
    # Origin identity is scoped and replayable, without duplicating its payload.
    source=await persist_profile_origin(db,user_id=uid,workspace_id=wid,agent_id=agents[0],origin=origin)
    assert source.ref==rows[0]["id"]
    assert await e.bind_memory_evidence(memory_id=ids[0],side="ai",user_id=uid,workspace_id=wid,
        sources=(source,),extractor_version="persona-init-v1")==0
    assert len(await db.query_raw("SELECT id FROM memory_profile_origins WHERE workspace_id=$1",wid))==1


@pytest.mark.parametrize("failure", ["memory","vector","audit","evidence"])
async def test_force_failure_preserves_old_memory_vector_origin_audit_and_cache(persona,failure):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    origin=prepare_profile_origin({"identity":{"location":"Synthetic old"}},None)
    previous=await store([sample("Synthetic old")],origin,force=True)
    before=await inspect(previous[0])
    life.bump_cache_version.reset_mock();life._seed_persona_entities.reset_mock()
    table={"memory":"memories_ai","vector":"memory_embeddings","audit":"memory_changelogs","evidence":"memory_evidence_links"}[failure]
    predicate={"memory":f"NEW.user_id='{uid}'", "vector":f"EXISTS(SELECT 1 FROM memories_ai WHERE id=NEW.memory_id AND user_id='{uid}')",
               "audit":f"NEW.user_id='{uid}'", "evidence":f"NEW.workspace_id='{wid}'"}[failure]
    name="fail_persona_origin_"+uuid4().hex
    await db.execute_raw(f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF {predicate} THEN RAISE EXCEPTION 'synthetic persona failure'; END IF; RETURN NEW; END $$")
    await db.execute_raw(f"CREATE TRIGGER {name} BEFORE INSERT ON {table} FOR EACH ROW EXECUTE FUNCTION {name}()")
    try:
        with pytest.raises((DataError, RawQueryError)):
            await store([sample("Synthetic replacement")],prepare_profile_origin({"identity":{"location":"new"}},None),force=True)
    finally:
        await db.execute_raw(f"DROP TRIGGER {name} ON {table}");await db.execute_raw(f"DROP FUNCTION {name}()")
    assert [m.id for m in await db.aimemory.find_many(where={"workspaceId":wid})]==previous
    after=await inspect(previous[0]);assert before["items"]==after["items"]
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_profile_origins WHERE workspace_id=$1",wid))[0]["n"]==1
    assert (await db.query_raw("SELECT count(*)::int n FROM memory_embeddings WHERE memory_id=ANY($1::text[])",previous))[0]["n"]==1
    assert await db.memorychangelog.count(where={"userId":uid,"operation":"insert"})==1
    life.bump_cache_version.assert_not_awaited();life._seed_persona_entities.assert_not_awaited()


async def test_missing_profile_input_stays_explicitly_unlinked_and_nonforce_skips(persona):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    assert await store()==[]
    ids=await store(force=True)
    result=await inspect(ids[0]);assert result["state"]=="current_unlinked"
    assert result["items"][0]["source_kind"]=="unlinked" and result["items"][0]["source_ref"] is None
    assert await db.query_raw("SELECT id FROM memory_profile_origins WHERE workspace_id=$1",wid)==[]
    assert await store(force=False)==[]
    assert await store([],force=True)==[]


async def test_force_vector_cleanup_preserves_same_id_user_and_other_scope(persona):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    old=await db.aimemory.create(data={"userId":uid,"workspaceId":wid,"content":"Synthetic old"})
    other=await db.aimemory.create(data={"userId":uid,"workspaceId":spaces[1],"content":"Synthetic other Agent"})
    for value in (mid,old.id,other.id):
        await db.execute_raw("INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",value,"["+",".join(["0.01"]*1024)+"]")
    await store([sample()],prepare_profile_origin({},None),force=True)
    remaining=await db.query_raw("SELECT memory_id FROM memory_embeddings WHERE memory_id=ANY($1::text[])",[mid,old.id,other.id])
    assert {r["memory_id"] for r in remaining}=={mid,other.id}
    assert await db.aimemory.find_unique(where={"id":other.id}) is not None


async def test_embedding_provider_failure_and_rebound_scope_never_delete_old_persona(persona,monkeypatch):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    model=type("Embedding",(),{})();model.aembed_documents=AsyncMock(side_effect=RuntimeError("synthetic provider failure"))
    monkeypatch.setattr(life,"get_embedding_model",lambda:model)
    with pytest.raises(RuntimeError):await store([sample(embedding=False)],force=True)
    assert await db.aimemory.find_unique(where={"id":mid}) is not None
    async def rebound(_):
        await db.chatworkspace.update(where={"id":wid},data={"agentId":agents[1]})
        return [[.01]*1024]
    model.aembed_documents.side_effect=rebound
    try:
        with pytest.raises(ValueError,match="scope_mismatch"):
            await store([sample(embedding=False)],force=True)
        assert await db.aimemory.find_unique(where={"id":mid}) is not None
    finally:await db.chatworkspace.update(where={"id":wid},data={"agentId":agents[0]})
    with pytest.raises(ValueError,match="scope_mismatch"):
        await life.store_memories_batch(agents[1],uid,[sample()],workspace_id=wid,force=True)
    with pytest.raises(ValueError,match="Invalid trusted persona category"):
        await store([{**sample(),"main_category":"invalid","sub_category":"invalid"}],force=True)


async def test_concurrent_nonforce_initialization_commits_only_one_complete_batch(persona,monkeypatch):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    await db.aimemory.delete(where={"id":mid})
    entered=0;both=asyncio.Event()
    async def embed(_):
        nonlocal entered
        entered+=1
        if entered==2:both.set()
        await asyncio.wait_for(both.wait(),5)
        return [[.01]*1024]
    model=type("Embedding",(),{})();model.aembed_documents=embed
    monkeypatch.setattr(life,"get_embedding_model",lambda:model)
    origin=prepare_profile_origin({},None)
    results=await asyncio.wait_for(asyncio.gather(store([sample(embedding=False)],origin),store([sample(embedding=False)],origin)),15)
    assert sorted(map(len,results))==[0,1]
    assert await db.aimemory.count(where={"workspaceId":wid})==1
    assert (await inspect(next(r[0] for r in results if r)))["state"]=="linked"


async def test_profile_source_scope_version_side_and_live_deletion_are_guarded(persona):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    origin=prepare_profile_origin({"identity":{"name":"Synthetic"}},None)
    ids=await store(origin=origin,force=True);result=await inspect(ids[0]);ref=result["items"][0]["source_ref"]
    async def binding(source,*,side="ai",memory_id=None):
        return await e.bind_memory_evidence(memory_id=memory_id or ids[0],side=side,user_id=uid,workspace_id=wid,
            sources=(source,),extractor_version="persona-init-v1")
    for source,error in [(e.EvidenceSource("profile","missing",relation="derived_from"),"not_found"),
                         (e.EvidenceSource("profile",ref,relation="recorded_from"),"invalid_profile"),
                         (e.EvidenceSource("profile",ref,side="ai",relation="derived_from"),"invalid_profile"),
                         (e.EvidenceSource("profile",ref,expected_role="user",relation="derived_from"),"invalid_profile"),
                         (e.EvidenceSource("profile",ref,expected_version="0"*64,relation="derived_from"),"version_changed")]:
        with pytest.raises(ValueError,match=error):await binding(source)
    for scope,agent,owner in [(spaces[1],agents[1],uid),(spaces[-1],agents[-1],users[-1])]:
        foreign=await persist_profile_origin(db,user_id=owner,workspace_id=scope,agent_id=agent,origin=origin)
        assert foreign.ref!=ref
        with pytest.raises(ValueError,match="scope_mismatch"):await binding(foreign)
    source=e.EvidenceSource("profile",ref,relation="derived_from",expected_version=origin.version)
    with pytest.raises(RawQueryError):await binding(source,side="user",memory_id=mid)
    with pytest.raises(RawQueryError,match="immutable"):
        await db.execute_raw("UPDATE memory_profile_origins SET source_status='provided_profile' WHERE id=$1",ref)
    raw=(await db.query_raw("SELECT * FROM memory_evidence_links WHERE memory_id=$1",ids[0]))[0]
    columns=[k for k in raw if k!="created_at"]
    for changes in [{"source_profile_id":None},{"source_status":"recorded"},{"source_memory_side":"ai"},
                    {"source_role":"user"},{"source_user_id":users[-1]},{"source_workspace_id":spaces[1]}]:
        row={**raw,**changes,"id":uuid4().hex*2}
        with pytest.raises(RawQueryError):
            await db.execute_raw("INSERT INTO memory_evidence_links ("+",".join(columns)+") VALUES ("+",".join(f"${i+1}" for i in range(len(columns)))+")",*[row[k] for k in columns])
    await db.chatworkspace.update(where={"id":wid},data={"agentId":agents[1]})
    unavailable=(await inspect(ids[0]))["items"][0]
    assert unavailable["availability"]=="unavailable" and unavailable["source_ref"] is None and unavailable["profile"] is None
    await db.chatworkspace.update(where={"id":wid},data={"agentId":agents[0]})
    await db.aimemory.update(where={"id":ids[0]},data={"content":"Synthetic edited persona"})
    assert (await inspect(ids[0]))["state"]=="current_unlinked"
    await db.execute_raw("DELETE FROM memory_profile_origins WHERE id=$1",ref)
    deleted=(await inspect(ids[0]))["items"][0]
    assert deleted["availability"]=="deleted" and deleted["source_ref"] is None and deleted["profile"] is None
    with pytest.raises(RawQueryError,match="immutable"):
        await db.execute_raw("UPDATE memory_evidence_links SET source_profile_id=$1 WHERE id=$2",ref,raw["id"])


async def test_origin_guard_rejects_missing_payload_keys_bad_hash_and_foreign_scope(persona):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    valid=prepare_profile_origin({},None)
    for payload in ['{}',json.dumps({"format":"persona-profile-v1","kind":"provided_profile","profile":{}})]:
        bad=ProfileOrigin(payload,e.content_version(payload),"provided_profile",False)
        with pytest.raises(RawQueryError):await persist_profile_origin(db,user_id=uid,workspace_id=wid,agent_id=agents[0],origin=bad)
    with pytest.raises(RawQueryError):
        await persist_profile_origin(db,user_id=uid,workspace_id=wid,agent_id=agents[0],origin=ProfileOrigin(valid.payload,"0"*64,valid.kind,False))
    with pytest.raises(RawQueryError,match="scope"):
        await persist_profile_origin(db,user_id=uid,workspace_id=spaces[-1],agent_id=agents[0],origin=valid)


async def test_generation_conversion_uses_frozen_profile_and_all_outputs_have_origins(persona,monkeypatch):
    from app.services.memory.init_report import InitReport
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    @asynccontextmanager
    async def lock(*args,**kwargs):yield
    @asynccontextmanager
    async def report(*args,**kwargs):yield InitReport(agent_id=agents[0])
    monkeypatch.setattr(life,"memory_generation_lock",lock);monkeypatch.setattr(life,"init_report",report)
    async def identity(memories):return [{**m,"_embedding":[.01]*1024} for m in memories]
    monkeypatch.setattr(life,"_detect_and_resolve_contradictions",identity)
    monkeypatch.setattr(life,"_embed_and_dedupe",identity)
    profile={"identity":{"name":"Synthetic","location":"Synthetic home","gender":"女"},"likes":{"foods":["合成食物"]}}
    count=await life.generate_l1_coverage(agents[0],uid,profile,None,workspace_id=wid)
    memories=await db.aimemory.find_many(where={"workspaceId":wid})
    assert count==len(memories)>3
    for memory in memories:
        item=(await inspect(memory.id))["items"][0]
        assert item["current_content"] and item["profile"]["input_status"]=="uncollected"
    with pytest.raises(ValueError,match="input_mismatch"):
        await life.generate_l1_coverage(agents[0],uid,profile,None,workspace_id=wid,origin=prepare_profile_origin({},None))


@pytest.mark.parametrize("profile,career,inputs,kind", [([],None,None,"provided_profile"),({},[],None,"provided_profile"),
    ({},None,[],"provided_profile"),({},None,None,"verified_fact"),({"x":float("nan")},None,None,"provided_profile"),
    ({"x":"汉"*100000},None,None,"provided_profile")])
def test_invalid_or_oversized_snapshot_rejected(profile,career,inputs,kind):
    with pytest.raises((ValueError,TypeError)):prepare_profile_origin(profile,career,inputs=inputs,kind=kind)


async def test_full_persona_batch_is_atomic_with_bounded_transaction_time(persona,record_property):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    started=time.monotonic()
    ids=await store([sample(f"Synthetic fact {i}") for i in range(250)],prepare_profile_origin({},None),force=True)
    elapsed=time.monotonic()-started
    record_property("persona_batch_250_seconds",elapsed)
    assert elapsed<25  # headroom under the production 30-second transaction
    rows=await db.query_raw("SELECT count(*)::int n FROM memory_evidence_links WHERE memory_id=ANY($1::text[])",ids)
    assert len(ids)==250 and rows[0]["n"]==250
    assert (await inspect(ids[-1]))["items"][0]["source_kind"]=="profile"


async def test_same_id_user_deletion_preserves_ai_dependencies_until_last_side(persona):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    await db.execute_raw("INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",mid,"["+",".join(["0.01"]*1024)+"]")
    await db.memorychangelog.create(data={"userId":uid,"workspaceId":wid,"memoryId":mid,"operation":"access"})
    await db.usermemory.delete(where={"id":mid})
    assert len(await db.query_raw("SELECT memory_id FROM memory_embeddings WHERE memory_id=$1",mid))==1
    assert await db.memorychangelog.count(where={"memoryId":mid})==1
    await db.aimemory.delete(where={"id":mid})
    assert await db.query_raw("SELECT memory_id FROM memory_embeddings WHERE memory_id=$1",mid)==[]
    assert await db.memorychangelog.count(where={"memoryId":mid})==0


async def test_initialization_does_not_block_ordinary_chat_source_binding(persona,monkeypatch):
    (db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs),store,inspect=persona
    entered=asyncio.Event();release=asyncio.Event()
    original=life.bind_memory_evidence
    async def paused(**kwargs):
        entered.set();await asyncio.wait_for(release.wait(),5)
        return await original(**kwargs)
    monkeypatch.setattr(life,'bind_memory_evidence',paused)
    task=asyncio.create_task(store(origin=prepare_profile_origin({},None),force=True))
    try:
        await asyncio.wait_for(entered.wait(),5)
        # This real write takes source/target/workspace SHARE locks. A workspace
        # UPDATE lock in initialization would stall it and permit lock inversion.
        assert await asyncio.wait_for(bind(e.EvidenceSource('message',msg.id)),2)==1
    finally:
        release.set();await task
    assert (await detail())["state"]=="linked"
