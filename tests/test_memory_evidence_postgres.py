"""Real provenance isolation, replay, deletion and rollback on owned fixtures."""
from base64 import urlsafe_b64encode
import os
from urllib.parse import urlsplit
from uuid import uuid4
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prisma import Prisma
from prisma.errors import RawQueryError
import pytest

from app.api.admin import memory_repairs
from app.services.auth import create_jwt
from app.services.memory import evidence as e, evidence_read as read
from app.services.memory.storage import repo


@pytest.fixture
async def origins(monkeypatch):
    url=os.environ.get("PROACTIVE_E2E_DATABASE_URL", "")
    if not url:
        pytest.skip("Memory evidence requires isolated PostgreSQL")
    assert urlsplit(url).hostname in {"localhost","127.0.0.1"}
    assert urlsplit(url).path=="/companion_proactive_e2e"
    db=Prisma(datasource={"url":url},http={"trust_env":False})
    await db.connect()
    from app.config import settings
    monkeypatch.setattr(settings,"jwt_secret","synthetic-memory-evidence-test-secret")
    for mod in (e,read,repo):monkeypatch.setattr(mod,"db",db)
    monkeypatch.setattr(repo,"_invalidate_caches",AsyncMock())
    users=[];agents=[];spaces=[];convs=[]
    try:
        for i in range(3):
            if i!=1:
                user=await db.user.create(data={"username":"evidence-test-"+uuid4().hex});users.append(user.id)
            agent=await db.aiagent.create(data={"userId":user.id,"name":"Synthetic"});agents.append(agent.id)
            space=await db.chatworkspace.create(data={"userId":user.id,"agentId":agent.id,
                "status":"archived" if i==1 else "active"});spaces.append(space.id)
            conv=await db.conversation.create(data={"userId":user.id,"agentId":agent.id,"workspaceId":space.id});convs.append(conv.id)
        uid=users[0];wid=spaces[0];mid=uuid4().hex
        for model in (db.usermemory,db.aimemory):
            await model.create(data={"id":mid,"userId":uid,"workspaceId":wid,"content":"合成地点事实","level":2})
        message=await db.message.create(data={"conversationId":convs[0],"role":"user","content":"合成用户说自己住在甲城"})
        async def bind(*sources,side="user",memory_id=mid,**kwargs):
            return await e.bind_memory_evidence(memory_id=memory_id,side=side,user_id=uid,
                workspace_id=wid,sources=tuple(sources),extractor_version="test-v1",**kwargs)
        async def detail(side="user",**kwargs):
            return await read.memory_evidence_detail(memory_id=mid,side=side,user_id=uid,workspace_id=wid,**kwargs)
        yield db,uid,wid,mid,message,bind,detail,users,agents,spaces,convs
    finally:
        for table in ("memories_user","memories_ai"):
            await db.execute_raw(f"DELETE FROM memory_embeddings WHERE memory_id IN (SELECT id FROM {table} WHERE user_id=ANY($1::text[]))",users)
        for model in (db.usermemory,db.aimemory):await model.delete_many(where={"userId":{"in":users}})
        await db.message.delete_many(where={"conversationId":{"in":convs}})
        await db.conversation.delete_many(where={"id":{"in":convs}})
        await db.chatworkspace.delete_many(where={"id":{"in":spaces}})
        await db.aiagent.delete_many(where={"id":{"in":agents}})
        await db.user.delete_many(where={"id":{"in":users}})
        await db.disconnect()


@pytest.mark.asyncio
async def test_recorded_origins_replay_pagination_and_unknown_are_distinct(origins):
    db,uid,wid,mid,msg,bind,detail,*_=origins
    assert (await detail())["state"]=="historical_unknown"
    origin=e.EvidenceSource("message",msg.id,relation="extracted_from",expected_role="user")
    assert await bind(origin)==1
    assert await bind(origin)==0
    result=await detail()
    assert result["state"]=="linked" and result["items"][0]["availability"]=="available"
    assert result["items"][0]["source_version"]==e.content_version(msg.content)
    assert "合成用户" not in str(result)
    assert (await detail("ai"))["state"]=="historical_unknown"
    for i in range(3):await bind(e.EvidenceSource("unlinked",f"origin-{i}"))
    page=await detail(limit=2);second=await detail(limit=2,cursor=page["next_cursor"])
    assert len(page["items"])+len(second["items"])==4 and second["next_cursor"] is None
    assert not {i["id"] for i in page["items"]}&{i["id"] for i in second["items"]}
    assert second["state"]=="linked"
    assert (await detail(cursor=urlsafe_b64encode(b"f"*64).decode()))["items"]==[]
    await db.usermemory.update(where={"id":mid},data={"content":"合成修改后的事实"})
    changed=await detail()
    assert changed["state"]=="current_unlinked" and not any(i["current_content"] for i in changed["items"])
    report=await read.audit_evidence_page(user_id=uid,workspace_id=wid,side="user",limit=1)
    assert report["dry_run"] and report["checked"]==1 and report["unknown"]==1
    assert report["denominator"]=="this_page_only"
    assert await db.usermemory.count(where={"userId":uid})==1


@pytest.mark.asyncio
async def test_binding_rejects_foreign_owner_agent_role_and_missing_sources(origins):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    for conv in convs[1:]:
        foreign=await db.message.create(data={"conversationId":conv,"role":"user","content":"外部合成"})
        with pytest.raises(ValueError,match="scope_mismatch"):await bind(e.EvidenceSource("message",foreign.id))
    for origin,error in [
        (e.EvidenceSource("message","missing"),"not_found"),
        (e.EvidenceSource("message",msg.id,expected_role="assistant"),"role_mismatch"),
        (e.EvidenceSource("message",msg.id,expected_version="0"*64),"version_changed"),
        (e.EvidenceSource("tool_success",msg.id),"unsupported"),
        (e.EvidenceSource("message",""),"invalid_evidence_reference"),
        (e.EvidenceSource("message",msg.id,relation="confirmed_true"),"invalid_evidence_relation"),
        (e.EvidenceSource("memory",mid,side=None),"invalid_memory_side"),
    ]:
        with pytest.raises(ValueError,match=error):await bind(origin)
    with pytest.raises(ValueError,match="target_scope"):await bind(e.EvidenceSource("message",msg.id),memory_id="missing")
    with pytest.raises(ValueError,match="invalid_evidence_batch"):await bind()
    with pytest.raises(ValueError,match="invalid_evidence_batch"):
        await bind(*(e.EvidenceSource("unlinked",str(i)) for i in range(101)))
    with pytest.raises(ValueError,match="invalid_evidence_batch"):
        await e.bind_memory_evidence(memory_id=mid,side="user",user_id=uid,workspace_id=wid,
            sources=(e.EvidenceSource("message",msg.id),),extractor_version="")
    await db.conversation.update(where={"id":convs[0]},data={"isDeleted":True})
    with pytest.raises(ValueError,match="not_found"):await bind(e.EvidenceSource("message",msg.id))
    assert (await detail())["state"]=="historical_unknown"


@pytest.mark.asyncio
async def test_deleted_changed_rebound_sources_and_target_cascade(origins):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    await bind(e.EvidenceSource("message",msg.id))
    await db.message.update(where={"id":msg.id},data={"content":"合成消息被编辑"})
    assert (await detail())["items"][0]["availability"]=="changed"
    await db.conversation.update(where={"id":convs[0]},data={"isDeleted":True})
    unavailable=(await detail())["items"][0]
    assert unavailable["availability"]=="unavailable" and unavailable["source_ref"] is None
    await db.message.delete(where={"id":msg.id})
    deleted=(await detail())["items"][0]
    assert deleted["availability"]=="deleted" and deleted["source_ref"] is None
    await db.aimemory.delete(where={"id":mid})
    assert len((await detail())["items"])==1
    await db.usermemory.delete(where={"id":mid})
    assert await db.query_raw("SELECT id FROM memory_evidence_links WHERE memory_id=$1",mid)==[]


@pytest.mark.asyncio
async def test_parent_side_scope_copy_and_deleted_parent(origins):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    await bind(e.EvidenceSource("memory",mid,side="ai",relation="derived_from"))
    parent=(await detail())["items"][0]
    assert parent["source_side"]=="ai" and parent["availability"]=="available"
    foreign=await db.aimemory.create(data={"userId":users[-1],"workspaceId":spaces[-1],"content":"合成模板"})
    with pytest.raises(ValueError,match="scope_mismatch"):
        await bind(e.EvidenceSource("memory",foreign.id,side="ai",relation="derived_from"))
    with pytest.raises(ValueError,match="template_scope_mismatch"):
        await bind(e.EvidenceSource("memory",foreign.id,side="ai",relation="template_copy"))
    await db.aiagent.update(where={"id":agents[0]},data={"sourceTemplateId":agents[-1]})
    await bind(e.EvidenceSource("memory",foreign.id,side="ai",relation="template_copy"))
    assert len((await detail())["items"])==2
    await db.aiagent.update(where={"id":agents[0]},data={"sourceTemplateId":None})
    assert any(i["availability"]=="unavailable" for i in (await detail())["items"])
    await db.aimemory.delete(where={"id":mid})
    assert any(i["availability"]=="deleted" for i in (await detail())["items"])


@pytest.mark.asyncio
async def test_write_transaction_rolls_back_and_side_collision_updates_correct_table(origins):
    db,uid,wid,mid,msg,bind,detail,*_=origins
    bad=(e.EvidenceSource("message","missing"),)
    with pytest.raises(ValueError):
        await repo.create(source="ai",userId=uid,workspaceId=wid,content="不得留下",evidence_sources=bad)
    assert await db.aimemory.count(where={"userId":uid})==1
    with pytest.raises(ValueError):
        await repo.update(mid,source="ai",content="失败更新",evidence_sources=bad)
    assert (await db.aimemory.find_unique(where={"id":mid})).content=="合成地点事实"
    await repo.update(mid,source="ai",content="合成 AI 事件",evidence_sources=(e.EvidenceSource("message",msg.id),))
    assert (await db.usermemory.find_unique(where={"id":mid})).content=="合成地点事实"
    assert (await detail("ai"))["items"][0]["current_content"]
    record=await repo.find_unique(mid)
    with pytest.raises(ValueError,match="side_mismatch"):await repo.update(mid,source="ai",record=record,content="错误侧")


@pytest.mark.asyncio
async def test_admin_scope_permissions_validation_and_readonly_preview(origins):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    app=FastAPI();app.include_router(memory_repairs.router)
    token=create_jwt("synthetic-admin",role="admin")
    headers={"Authorization":"Bearer "+token}
    path=f"/admin-api/memory-repairs/evidence/user/{mid}"
    params={"user_id":uid,"workspace_id":wid}
    async with AsyncClient(transport=ASGITransport(app=app),base_url="http://isolated.test") as client:
        assert (await client.get(path,params=params)).status_code==401
        assert (await client.get(path,params=params,headers={"Authorization":"Bearer "+create_jwt(uid,role="user")})).status_code==403
        assert (await client.get(path,params=params,headers=headers)).json()["state"]=="historical_unknown"
        for key,value in [("workspace_id",spaces[1]),("user_id",users[-1])]:
            assert (await client.get(path,params={**params,key:value},headers=headers)).status_code==404
        for cursor in ["!",urlsafe_b64encode(b"wrong").decode(),"x"*299]:
            assert (await client.get(path,params={**params,"cursor":cursor},headers=headers)).status_code==400
        assert (await client.get(path,params={**params,"limit":51},headers=headers)).status_code==422
        assert (await client.get(path.replace("/user/","/other/"),params=params,headers=headers)).status_code==400
        audit=await client.get("/admin-api/memory-repairs/evidence-audit",params={**params,"side":"ai","limit":1},headers=headers)
        assert audit.status_code==200 and audit.json()["dry_run"]
        unknown=await client.get("/admin-api/memory-repairs/evidence-audit",params={**params,"side":"ai","workspace_id":spaces[-1]},headers=headers)
        assert unknown.status_code==404
    for limit in [0,51]:
        with pytest.raises(ValueError):await detail(limit=limit)
    for cursor in ["a"*301,urlsafe_b64encode(b"\xff").decode()]:
        with pytest.raises(ValueError):await detail(cursor=cursor)
    with pytest.raises(ValueError):await read.audit_evidence_page(user_id=uid,workspace_id=wid,side="ai",limit=501)


@pytest.mark.asyncio
async def test_input_snapshot_rechecks_versions_before_binding(origins):
    db,uid,wid,mid,msg,bind,detail,*_=origins
    origins=await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side="user",
        message_ids=[msg.id,msg.id],extraction_input="user: "+msg.content)
    assert len(origins)==1 and origins[0].expected_version==e.content_version(msg.content)
    for ids,text in [([],msg.content),([msg.id],"其他内容"),(["missing"],msg.content),(["x"*201],msg.content)]:
        with pytest.raises(ValueError):
            await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side="user",message_ids=ids,extraction_input=text)
    await db.message.update(where={"id":msg.id},data={"content":"后来编辑的消息"})
    with pytest.raises(ValueError,match="version_changed"):await bind(*origins)
    assert (await detail())["state"]=="historical_unknown"


@pytest.mark.asyncio
async def test_database_guards_scope_version_side_and_immutable_snapshots(origins):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    await bind(e.EvidenceSource("message",msg.id))
    row=(await db.query_raw("SELECT * FROM memory_evidence_links WHERE memory_id=$1",mid))[0]
    # Raw SQL must obey the same guards as the application adapter.
    columns=[key for key in row if key!="created_at"]
    for changes in [{"user_id":users[-1]}, {"workspace_id":spaces[1]}, {"agent_id":agents[-1]},
                    {"content_version":"0"*64}, {"source_version":"0"*64}, {"source_role":"assistant"},
                    {"source_message_id":None}, {"memory_source":"ai"}, {"user_memory_id":None},
                    {"source_kind":"tool_success"}]:
        values={**row,**changes,"id":uuid4().hex*2}
        sql="INSERT INTO memory_evidence_links ("+",".join(columns)+") VALUES ("+",".join(f"${i+1}" for i in range(len(columns)))+")"
        with pytest.raises(RawQueryError):await db.execute_raw(sql,*[values[key] for key in columns])
    with pytest.raises(RawQueryError,match="immutable"):
        await db.execute_raw("UPDATE memory_evidence_links SET source_version=$1 WHERE id=$2","0"*64,row["id"])
    await db.message.delete(where={"id":msg.id})
    assert (await detail())["items"][0]["availability"]=="deleted"
    # Referential nulling is allowed; resurrecting a source pointer is not.
    replacement=await db.message.create(data={"conversationId":convs[0],"role":"user","content":"另一个消息"})
    with pytest.raises(RawQueryError,match="immutable"):
        await db.execute_raw("UPDATE memory_evidence_links SET source_message_id=$1 WHERE id=$2",replacement.id,row["id"])


@pytest.mark.asyncio
async def test_split_storage_links_every_output_and_rolls_back_invalid_origin(origins,monkeypatch):
    from app.services.memory.storage import persistence as p
    from app.services.memory.recording import pipeline as pl
    db,uid,wid,mid,msg,bind,detail,*_=origins
    monkeypatch.setattr(p,"db",db)
    monkeypatch.setattr(p,"generate_embedding",AsyncMock(return_value=[0.0]*1024))
    monkeypatch.setattr(p,"store_embedding",AsyncMock())
    monkeypatch.setattr(p,"log_memory_changelog",AsyncMock())
    text=("我的爱好是公园跑步："+"周末我会沿河跑三公里锻炼身体，留意沿途风景。"*12+
          "；我的爱好是阅读历史："+"晚上我会读历史书籍了解过去的生活故事，再记下感想。"*12)
    sources=await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side="user",
        message_ids=[msg.id],extraction_input=msg.content)
    result=await p.store_memory(user_id=uid,workspace_id=wid,content=text,level=2,
        main_category="偏好",sub_category="兴趣爱好",skip_reconciliation=True,
        evidence_sources=sources,extractor_version="test-split-v1")
    assert result
    rows=await db.query_raw("SELECT m.id,count(e.id)::int AS n FROM memories_user m JOIN memory_evidence_links e ON e.user_memory_id=m.id WHERE m.user_id=$1 GROUP BY m.id",uid)
    assert len(rows)>=2 and all(row["n"]==1 for row in rows)
    assert p.store_embedding.await_count==len(rows)
    page=await read.audit_evidence_page(user_id=uid,workspace_id=wid,side="user",limit=1)
    assert page["next_after_id"] and page["checked"]==1
    next_page=await read.audit_evidence_page(user_id=uid,workspace_id=wid,side="user",limit=1,after_id=page["next_after_id"])
    assert next_page["checked"]==1
    # Exercise the actual extraction adapter without calling a live model.
    monkeypatch.setattr(pl,"resolve_workspace_id",AsyncMock(return_value=wid))
    monkeypatch.setattr(pl,"should_memorize",AsyncMock(return_value=True))
    monkeypatch.setattr(pl,"extract_memories",AsyncMock(return_value={"memories":[{"content":"用户喜爱阅读","importance":.6,"main_category":"偏好","sub_category":"兴趣爱好"}]}))
    store=AsyncMock(return_value="captured-id");monkeypatch.setattr(pl,"store_memory",store)
    monkeypatch.setattr(pl,"log_memory_evidence",AsyncMock())
    monkeypatch.setattr(pl,"record_entities_for_memory",AsyncMock())
    monkeypatch.setattr(pl,"record_topics_for_memory",AsyncMock())
    monkeypatch.setattr(pl,"record_preferences_for_memory",AsyncMock())
    await pl.process_memory_pipeline(user_id=uid,workspace_id=wid,new_conversation="user: "+msg.content,
        evidence_message_ids=[msg.id])
    assert store.await_args.kwargs["evidence_sources"][0].expected_version==e.content_version(msg.content)
    with pytest.raises(pl.MemoryExtractionError):
        await pl.process_memory_pipeline(user_id=uid,workspace_id=wid,new_conversation="user: "+msg.content,evidence_message_ids=["missing"])


@pytest.mark.asyncio
async def test_name_replacement_keeps_old_name_when_origin_validation_fails(origins,monkeypatch):
    from app.services.memory.storage import persistence as p
    db,uid,wid,mid,msg,bind,detail,*_=origins
    await db.usermemory.update(where={"id":mid},data={"level":1,"content":"用户姓名为旧合成名","mainCategory":"身份","subCategory":"姓名"})
    monkeypatch.setattr(p,"db",db)
    monkeypatch.setattr(p,"generate_embedding",AsyncMock(return_value=[0.0]*1024))
    monkeypatch.setattr(p,"store_embedding",AsyncMock())
    monkeypatch.setattr(p,"log_memory_changelog",AsyncMock())
    args=dict(user_id=uid,workspace_id=wid,content="用户姓名为新合成名",level=1,
              main_category="身份",sub_category="姓名",_singleton_locked=True)
    with pytest.raises(ValueError,match="not_found"):
        await p.store_memory(**args,evidence_sources=(e.EvidenceSource("message","missing"),))
    assert not (await db.usermemory.find_unique(where={"id":mid})).isArchived
    from app.services.memory.storage import embedding
    monkeypatch.setattr(p,"store_embedding",embedding.store_embedding)
    monkeypatch.setattr(p,"generate_embedding",AsyncMock(return_value=[0.0]))
    with pytest.raises(RawQueryError):
        await p.store_memory(**args,evidence_sources=(e.EvidenceSource("message",msg.id),))
    assert not (await db.usermemory.find_unique(where={"id":mid})).isArchived
    assert await db.usermemory.count(where={"userId":uid})==1
    monkeypatch.setattr(p,"store_embedding",AsyncMock())
    monkeypatch.setattr(p,"generate_embedding",AsyncMock(return_value=[0.0]*1024))
    result=await p.store_memory(**args,evidence_sources=(e.EvidenceSource("message",msg.id),))
    assert result and result!=mid
    assert (await db.usermemory.find_unique(where={"id":mid})).isArchived
    assert await db.usermemory.count(where={"userId":uid,"level":1,"isArchived":False})==1


@pytest.mark.asyncio
async def test_unlinked_import_and_workspace_rebinding_are_explicit(origins):
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs=origins
    await bind(e.EvidenceSource("import","no-origin-archive"))
    result=await detail()
    assert result["state"]=="current_unlinked"
    assert result["items"][0]["availability"]=="unverified" and result["items"][0]["source_ref"] is None
    await bind(e.EvidenceSource("message",msg.id))
    await db.chatworkspace.update(where={"id":wid},data={"agentId":agents[1]})
    result=await detail()
    assert result["state"]=="current_unlinked" and all(i["availability"]=="unavailable" for i in result["items"])
    audit=await read.audit_evidence_page(user_id=uid,workspace_id=wid,side="user")
    assert audit["linked"]==0 and audit["unknown"]==1


@pytest.mark.asyncio
async def test_reconciliation_vector_and_content_roll_back_together(origins,monkeypatch):
    from app.services.memory.storage import persistence as p, embedding
    from app.services.memory.storage.reconciliation import ReconciliationDecision
    db,uid,wid,mid,msg,bind,detail,*_=origins
    monkeypatch.setattr(p,"db",db)
    monkeypatch.setattr(p,"generate_embedding",AsyncMock(return_value=[0.0,1.0]+[0.0]*1022))
    monkeypatch.setattr(p,"store_embedding",embedding.store_embedding)
    monkeypatch.setattr(p,"log_memory_changelog",AsyncMock())
    record=await repo.find_unique(mid)
    monkeypatch.setattr(p,"resolve_memory_write",AsyncMock(return_value=ReconciliationDecision(
        action="merge_existing",existing_id=mid,existing_record=record,merged_content="合成合并后的事实")))
    await embedding.store_embedding(mid,[1.0]+[0.0]*1023,database=db)
    args=dict(user_id=uid,workspace_id=wid,content="新增合成事实",level=2,
              main_category="生活",sub_category="生活")
    with pytest.raises(ValueError,match="not_found"):
        await p.store_memory(**args,evidence_sources=(e.EvidenceSource("message","missing"),))
    assert (await db.usermemory.find_unique(where={"id":mid})).content==record.content
    vector=(await db.query_raw("SELECT embedding::text AS v FROM memory_embeddings WHERE memory_id=$1",mid))[0]["v"]
    assert vector.startswith("[1,0,")
    await p.store_memory(**args,evidence_sources=(e.EvidenceSource("message",msg.id),))
    assert (await db.usermemory.find_unique(where={"id":mid})).content=="合成合并后的事实"
    assert (await detail())["state"]=="linked"
