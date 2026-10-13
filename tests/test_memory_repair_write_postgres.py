"""Real JWT/HTTP/PG acceptance for atomic repairs; external embeddings are stubbed."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.services.memory import repair_write, evidence_read, repair_queue
from tests.test_memory_evidence_postgres import origins
from tests.test_memory_input_write_api_postgres import editing, auth, reject_write, VECTOR


@pytest.fixture
async def repairing(editing, monkeypatch):
    o,target,queue,client,embedding,admin = editing
    monkeypatch.setattr(repair_write,'db',o[0])
    monkeypatch.setattr(repair_write,'generate_embedding',embedding)
    monkeypatch.setattr(repair_write,'after_memory_changelog',AsyncMock())
    monkeypatch.setattr(repair_write,'invalidate_knowledge_rows',AsyncMock())
    async def submit(item, action='merge_memories', **data):
        return await client.post(f"/admin-api/memory-repairs/{item['id']}/actions",
            headers=auth(admin,'admin'),json={'action':action,'expected_updated_at':item['updated_at'],**data})
    yield (*editing, submit)


@pytest.mark.parametrize('change_during_embedding', [False, True])
async def test_real_prisma_timestamp_only_change_rejects_stale_repair(repairing, change_during_embedding):
    o, target, queue, client, embedding, admin, submit = repairing
    db, uid, wid, *_ = o
    row = await target()
    item = await queue(row)
    assert isinstance(item['updated_at'], str) and item['updated_at']
    listed = await client.get('/admin-api/memory-repairs', headers=auth(admin, 'admin'),
                             params={'user_id': uid, 'workspace_id': wid, 'limit': 2})
    assert listed.status_code == 200
    assert listed.json()['items'][0]['updated_at'] == item['updated_at']

    async def change_version(*_args, **_kwargs):
        await db.execute_raw("UPDATE memory_repair_items SET updated_at=updated_at+INTERVAL '1 second' WHERE id=$1", item['id'])
        return VECTOR

    if change_during_embedding:
        embedding.side_effect = change_version
    else:
        await change_version()
    response = await submit(item, 'insert_replacement_memory', content='合成替代事实')
    assert response.status_code == 409, response.text
    assert not (await db.usermemory.find_unique(where={'id': row.id})).isArchived
    assert (await repair_queue.get_memory_repair_item(item['id']))['status'] == 'open'
    assert await db.query_raw('SELECT id FROM memory_input_origins WHERE workspace_id=$1', wid) == []


@pytest.mark.parametrize('side',['user','ai'])
@pytest.mark.parametrize('action',['merge_memories','insert_replacement_memory','archive_memory','downgrade_memory','mark_verified'])
async def test_success_is_atomic_scoped_and_auditable(repairing,side,action):
    o,target,queue,client,embedding,admin,submit = repairing
    db,uid,wid,mid,msg,bind,detail,users,agents,spaces,convs = o
    row = await target(side); parent = await target(side)
    item = await queue(row,side)
    response = await submit(item,action,content='修复后的合成事实',memory_ids=[parent.id],level=3)
    assert response.status_code==200,response.text
    result=response.json();assert result['repair_item']['status']=='resolved'
    saved=await repair_queue.get_memory_repair_item(item['id'])
    assert saved['evidence']['repair_result']['memory_id']==result['memory_id']
    model=db.aimemory if side=='ai' else db.usermemory
    actual=await model.find_unique(where={'id':result['memory_id']})
    assert (actual.userId,actual.workspaceId)==(uid,wid)
    if action in {'merge_memories','insert_replacement_memory'}:
        assert actual.content=='修复后的合成事实'
        old=await model.find_unique(where={'id':parent.id if action=='merge_memories' else row.id})
        assert old.isArchived
        assert result['memory_id'] != row.id if action=='insert_replacement_memory' else result['memory_id']==row.id
        assert (await db.query_raw('SELECT embedding::text AS v FROM memory_embeddings WHERE memory_id=$1',actual.id))[0]['v'].startswith('[0.012,')
        if action=='insert_replacement_memory':
            assert actual.valueUpdatedAt is not None and actual.statementTime is not None
            audit=await db.memorychangelog.find_first(where={'memoryId':actual.id,'operation':'repair_insert_replacement'})
            assert audit.oldValue is None
    else:
        assert actual.content==row.content
        assert actual.isArchived == (action=='archive_memory')
        assert actual.level == (3 if action=='downgrade_memory' else row.level)
        embedding.assert_not_awaited()
    sources=await evidence_read.memory_evidence_detail(user_id=uid,workspace_id=wid,side=side,memory_id=actual.id)
    input_record=next(x for x in sources['items'] if x['source_kind']=='input')
    assert input_record['source_status']=='admin_repair' and input_record['current_content']
    assert admin not in str(sources) and '修复后的合成事实' not in str(sources)
    raw=(await db.query_raw('SELECT payload_text FROM memory_input_origins WHERE id=$1',input_record['source_ref']))[0]
    snapshot=json.loads(raw['payload_text'])['input'];assert snapshot['before'][0]['content']==row.content
    assert snapshot['action']==action and snapshot['actor_id']==admin
    assert (await submit(item,action,content='再次执行',memory_ids=[parent.id])).status_code==409


@pytest.mark.parametrize('action',['merge_memories','insert_replacement_memory'])
@pytest.mark.parametrize('failure',['memory','vector','origin','evidence','audit','resolve','archive'])
async def test_every_persisted_step_rolls_back_and_retry_works(repairing,action,failure):
    o,target,queue,client,embedding,admin,submit = repairing
    db,uid,wid,*_=o;row=await target();parent=await target();item=await queue(row)
    table,predicate,op={
        'memory':('memories_user',"NEW.content='修复后的合成事实'",'INSERT' if action=='insert_replacement_memory' else 'UPDATE'),
        'vector':('memory_embeddings',"true",'INSERT'),
        'origin':('memory_input_origins',"true",'INSERT'),
        'evidence':('memory_evidence_links',"NEW.source_kind='input'",'INSERT'),
        'audit':('memory_changelogs',"NEW.operation LIKE 'repair_%'",'INSERT'),
        'resolve':('memory_repair_items',"NEW.status='resolved'",'UPDATE'),
        'archive':('memories_user',"NEW.is_archived",'UPDATE'),
    }[failure]
    before=await db.usermemory.count(where={'workspaceId':wid})
    async with reject_write(db,table,predicate,operation=op):
        response=await submit(item,action,content='修复后的合成事实',memory_ids=[parent.id])
        assert response.status_code==500,response.text
    assert await db.usermemory.count(where={'workspaceId':wid})==before
    assert (await db.usermemory.find_unique(where={'id':row.id})).content==row.content
    assert not (await db.usermemory.find_unique(where={'id':parent.id})).isArchived
    assert (await db.query_raw('SELECT embedding::text v FROM memory_embeddings WHERE memory_id=$1',row.id))[0]['v'].startswith('[0.004,')
    assert (await repair_queue.get_memory_repair_item(item['id']))['status']=='open'
    assert await db.query_raw('SELECT id FROM memory_input_origins WHERE workspace_id=$1',wid)==[]
    assert (await submit(item,action,content='修复后的合成事实',memory_ids=[parent.id])).status_code==200


@pytest.mark.parametrize('change',['target','parent','generation','actor','item','delete'])
async def test_rechecks_changes_during_embedding(repairing,change):
    o,target,queue,client,embedding,admin,submit=repairing
    db,uid,wid,_,_,_,_,users,agents,*_=o;row=await target();parent=await target();item=await queue(row)
    async def mutate(_):
        if change in {'target','parent'}:
            await db.usermemory.update(where={'id':row.id if change=='target' else parent.id},data={'content':'并发编辑的新值'})
        elif change=='generation':
            await db.execute_raw('UPDATE chat_workspaces SET execution_generation=md5(execution_generation) WHERE id=$1',wid)
        elif change=='actor':
            await db.user.update(where={'id':admin},data={'role':'user'})
        elif change=='item':
            await repair_queue.update_memory_repair_item_status(item['id'],status='dismissed')
        else:
            await db.usermemory.delete(where={'id':parent.id})
        return VECTOR
    embedding.side_effect=mutate
    response=await submit(item,content='修复后的合成事实',memory_ids=[parent.id])
    assert response.status_code in {403,404,409},response.text
    assert await db.query_raw('SELECT id FROM memory_input_origins WHERE workspace_id=$1',wid)==[]
    assert (await repair_queue.get_memory_repair_item(item['id']))['status'] == ('dismissed' if change=='item' else 'open')


async def test_double_submit_and_opposing_merges_do_not_partially_apply(repairing):
    o,target,queue,client,embedding,admin,submit=repairing
    db,uid,wid,*_=o;row=await target();parent=await target();item=await queue(row);other=await queue(parent)
    ready=asyncio.Event();count=0
    async def embed(_):
        nonlocal count
        count+=1
        if count==2:ready.set()
        await asyncio.wait_for(ready.wait(),5)
        return VECTOR
    embedding.side_effect=embed
    responses=await asyncio.wait_for(asyncio.gather(
        submit(item,content='合并甲',memory_ids=[parent.id]),
        submit(other,content='合并乙',memory_ids=[row.id])),15)
    assert sorted(x.status_code for x in responses)==[200,404]
    assert len(await db.query_raw('SELECT id FROM memory_input_origins WHERE workspace_id=$1',wid))==1


async def test_replacement_scope_archive_context_and_parent_deletion(repairing):
    o,target,queue,client,embedding,admin,submit=repairing
    db,uid,wid,*_=o;row=await target();item=await queue(row)
    for key,value in [('user_id',o[7][-1]),('workspace_id',o[9][-1]),('source','ai')]:
        assert (await submit(item,'insert_replacement_memory',content='新事实',**{key:value})).status_code==403
    embedding.assert_not_awaited()
    await db.usermemory.update(where={'id':row.id},data={'isArchived':True})
    response=await submit(item,'insert_replacement_memory',content='新事实')
    assert response.status_code==200,response.text
    await db.usermemory.delete(where={'id':row.id})
    result=await evidence_read.memory_evidence_detail(user_id=uid,workspace_id=wid,side='user',memory_id=response.json()['memory_id'])
    parent=next(x for x in result['items'] if x['source_kind']=='memory')
    assert parent['availability']=='deleted' and parent['source_ref'] is None


async def test_replacement_refuses_unrelated_duplicate_and_singleton(repairing):
    o,target,queue,client,embedding,admin,submit=repairing
    db,uid,wid,*_=o;row=await target();other=await target();item=await queue(row)
    assert (await submit(item,'insert_replacement_memory',content=other.content)).status_code==409
    await db.usermemory.update(where={'id':other.id},data={'mainCategory':'身份','subCategory':'姓名','level':1})
    response=await submit(item,'insert_replacement_memory',content='新姓名',main_category='身份',sub_category='姓名',level=1)
    assert response.status_code==409,response.text
    assert not (await db.usermemory.find_unique(where={'id':row.id})).isArchived


async def test_merge_limits_stale_ui_and_cross_side_fail_closed(repairing):
    o,target,queue,client,embedding,admin,submit=repairing
    db,uid,wid,*_=o;row=await target();parent=await target('ai');item=await queue(row)
    for kwargs,code in [({'memory_ids':[]},400),({'memory_ids':['x']*21},422),
        ({'memory_ids':[parent.id]},403),({'expected_updated_at':'old'},409)]:
        assert (await submit(item,content='新内容',**kwargs)).status_code==code
    embedding.assert_not_awaited()


async def test_missing_old_target_requires_valid_explicit_scope(repairing):
    o,target,queue,client,embedding,admin,submit=repairing
    db,uid,wid,*_=o;row=await target();item=await queue(row)
    await db.execute_raw('UPDATE memory_repair_items SET memory_id=NULL WHERE id=$1',item['id'])
    item=await repair_queue.get_memory_repair_item(item['id'])
    result=await submit(item,'insert_replacement_memory',content='补录的合成事实',main_category='生活',sub_category='其他')
    assert result.status_code==200,result.text
    assert result.json()['memory_id']!=row.id
