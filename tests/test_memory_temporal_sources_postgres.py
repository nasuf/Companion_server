"""Real DB: temporal origin integrity, atomic rollback, replay and erasure."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from app.services.memory import evidence as e
from app.services.memory.recording.temporal_sources import resolve_fact_clock
from app.services.memory.storage import persistence as store
from tests.test_memory_evidence_postgres import origins

T=datetime(2026,10,20,1,tzinfo=timezone.utc)


async def snapshot(origins):
    db,uid,wid,mid,msg,*_=origins
    await db.message.update(where={'id':msg.id},data={'content':'昨天去看展了','createdAt':T})
    return await e.snapshot_message_sources(user_id=uid,workspace_id=wid,side='user',
        message_ids=[msg.id],extraction_input='user: 昨天去看展了',conversation_id=msg.conversationId)


async def test_origin_time_replay_and_public_metadata(origins):
    db,uid,wid,mid,msg,bind,detail,*_=origins
    sources=await snapshot(origins)
    clock=resolve_fact_clock({'source_indices':[1],'time_quote':'昨天','time_source_index':1},sources)
    bound=clock.bind(sources)
    assert await bind(*bound)==1
    assert await bind(*bound)==0
    item=(await detail())['items'][0]
    assert item['source_statement_time']==T.isoformat()
    assert item['temporal_basis']['event_start'].startswith('2026-10-19')
    assert '昨天' not in str(item) and 'input_text' not in item
    await db.message.update(where={'id':msg.id},data={'createdAt':T+timedelta(days=1)})
    assert (await detail())['items'][0]['availability']=='changed'
    with pytest.raises(ValueError,match='source_time_changed'):await bind(*bound)
    await db.message.delete(where={'id':msg.id})
    item=(await detail())['items'][0]
    assert item['availability']=='deleted' and item['temporal_basis'] is None
    assert item['source_statement_time'] is None


@pytest.mark.parametrize('mutation',['time','delete','content','archive'])
async def test_mutated_source_rolls_back_new_fact_and_clock(origins,mutation):
    db,uid,wid,mid,msg,*_=origins
    sources=await snapshot(origins)
    if mutation=='time':await db.message.update(where={'id':msg.id},data={'createdAt':T+timedelta(days=1)})
    if mutation=='delete':await db.message.delete(where={'id':msg.id})
    if mutation=='content':await db.message.update(where={'id':msg.id},data={'content':'changed'})
    if mutation=='archive':await db.chatworkspace.update(where={'id':wid},data={'status':'archived'})
    from app.services.memory.storage import repo
    before=await db.usermemory.count(where={'userId':uid})
    with pytest.raises(ValueError):
        await repo.create(source='user',userId=uid,workspaceId=wid,content='回滚事实',level=2,
                          evidence_sources=sources,statementTime=T,occurTime=T)
    assert await db.usermemory.count(where={'userId':uid})==before
    assert await db.query_raw('SELECT id FROM memory_evidence_links WHERE memory_id=$1',mid)==[]


@pytest.mark.parametrize('known',[True,False])
async def test_storage_preserves_explicit_unknown_instead_of_inferring_now(origins,monkeypatch,known):
    db,uid,wid,mid,msg,*_=origins
    sources=await snapshot(origins)
    clock=resolve_fact_clock({'source_indices':[1] if known else [],
                             'time_quote':'昨天' if known else None,'time_source_index':1},sources)
    monkeypatch.setattr(store,'db',db)
    monkeypatch.setattr(store,'generate_embedding',AsyncMock(return_value=[0.01]*1024))
    monkeypatch.setattr(store,'store_embedding',AsyncMock())
    monkeypatch.setattr(store,'resolve_memory_write',AsyncMock(return_value=SimpleNamespace(action='insert_new')))
    monkeypatch.setattr(store,'log_memory_changelog',AsyncMock())
    got=await store.store_memory(uid,'用户看了一场展览',workspace_id=wid,level=2,importance=.6,
        main_category='生活',sub_category='其他',statement_time=clock.statement_time,
        occur_time=clock.occur_time,infer_time=False,evidence_sources=clock.bind(sources))
    row=await db.usermemory.find_unique(where={'id':got})
    assert row.statementTime==clock.statement_time
    assert row.occurTime==clock.occur_time
    links=await db.query_raw('SELECT temporal_basis FROM memory_evidence_links WHERE memory_id=$1',got)
    assert links[0]['temporal_basis']['status']==('resolved' if known else 'source_unresolved')


async def test_merge_clears_false_clock_without_legacy_rebinding(origins,monkeypatch):
    from app.services.memory.storage import repo
    from app.services.memory.lifecycle import quality
    db,uid,wid,mid,msg,*_=origins
    sources=await snapshot(origins)
    clock=resolve_fact_clock({'source_indices':[1],'time_quote':'昨天','time_source_index':1},sources)
    old=await repo.find_unique(mid)
    monkeypatch.setattr(store,'db',db)
    monkeypatch.setattr(store,'generate_embedding',AsyncMock(return_value=[0.01]*1024))
    monkeypatch.setattr(store,'store_embedding',AsyncMock())
    monkeypatch.setattr(store,'resolve_memory_write',AsyncMock(return_value=SimpleNamespace(
        action='merge_existing',existing_id=mid,existing_record=old,merged_content='合并了不同日期的两件事情')))
    audit=AsyncMock();monkeypatch.setattr(store,'log_memory_changelog',audit)
    got=await store.store_memory(uid,'用户看了一场展览',workspace_id=wid,level=2,importance=.6,
        main_category='生活',sub_category='其他',statement_time=clock.statement_time,
        occur_time=clock.occur_time,infer_time=False,evidence_sources=clock.bind(sources))
    await quality.log_memory_evidence(user_id=uid,workspace_id=wid,memory_id=got,
        message_ids=[msg.id],evidence_sources=clock.bind(sources),sources_already_bound=True)
    row=await db.usermemory.find_unique(where={'id':got})
    assert row.statementTime is None and row.occurTime is None
    links=await db.query_raw('SELECT temporal_basis FROM memory_evidence_links WHERE memory_id=$1',got)
    assert len(links)==1 and links[0]['temporal_basis']['status']=='merge_unresolved'


async def test_reminder_uses_retained_clock_not_stale_extraction(origins,monkeypatch):
    from app.services.memory.recording import pipeline as p
    import app.services.reminder.scheduling as scheduling
    db,uid,wid,mid,msg,*_=origins
    monkeypatch.setattr(p,'get_active_workspace',AsyncMock(return_value=SimpleNamespace(id=wid,agentId=origins[8][0])))
    create=AsyncMock();monkeypatch.setattr(scheduling,'upsert_reminder_trigger',create)
    clock=datetime(2027,10,23,7,tzinfo=timezone.utc)
    await db.usermemory.update(where={'id':mid},data={'subCategory':'提醒','occurTime':clock})
    await p._create_reminder_timetrigger(user_id=uid,memory_id=mid,summary='提醒',occur_time=clock,
        recurrence='once',side='user',require_stored_clock=True)
    assert create.await_count==1
    await db.usermemory.update(where={'id':mid},data={'occurTime':None})
    await p._create_reminder_timetrigger(user_id=uid,memory_id=mid,summary='提醒',occur_time=clock,
        recurrence='once',side='user',require_stored_clock=True)
    assert create.await_count==1


async def test_split_facts_do_not_broadcast_the_parent_event_date(origins,monkeypatch):
    db,uid,wid,mid,msg,*_=origins
    sources=await snapshot(origins)
    clock=resolve_fact_clock({'source_indices':[1],'time_quote':'昨天','time_source_index':1},sources)
    pieces=['昨天去看了一场展览','还打算在未来学习摄影']
    monkeypatch.setattr(store,'_split_for_storage',lambda *_:pieces)
    monkeypatch.setattr(store,'db',db)
    monkeypatch.setattr(store,'generate_embedding',AsyncMock(return_value=[0.01]*1024))
    monkeypatch.setattr(store,'store_embedding',AsyncMock())
    monkeypatch.setattr(store,'resolve_memory_write',AsyncMock(return_value=SimpleNamespace(action='insert_new')))
    monkeypatch.setattr(store,'log_memory_changelog',AsyncMock())
    got=await store.store_memory(uid,'；'.join(pieces),workspace_id=wid,level=2,importance=.6,
        main_category='生活',sub_category='其他',statement_time=clock.statement_time,
        occur_time=clock.occur_time,infer_time=False,evidence_sources=clock.bind(sources))
    rows=await db.usermemory.find_many(where={'userId':uid,'content':{'in':pieces}})
    assert len(rows)==2 and got in {row.id for row in rows}
    assert all(row.occurTime is None and row.statementTime is None for row in rows)
    links=await db.query_raw('SELECT temporal_basis FROM memory_evidence_links WHERE memory_id=ANY($1::text[])',[row.id for row in rows])
    assert len(links)==2 and all(link['temporal_basis']=={
        'version':1,'status':'split_unresolved','source_role':'batch'} for link in links)
