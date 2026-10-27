"""Real PostgreSQL message outcomes; model output is deliberately synthetic."""
from datetime import timedelta
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.services.memory.recording import pipeline, receipts, receipt_read
from app.services.memory.recording.context_window import history_rows, format_history
from app.services.chat import post_process as post
from tests.test_memory_receipts_postgres import ledger, origins
from tests.test_memory_chat_sources import chat, observations, redis


@pytest.mark.parametrize("mode,expected,reason", [
    ("empty", "no_fact", "empty_extraction"),
    ("heuristic", "policy_skipped", "heuristic_filtered"),
    ("prefilter", "policy_skipped", "prefilter_rejected"),
])
async def test_empty_results_are_durable_and_not_model_replayed(ledger, monkeypatch, mode, expected, reason):
    db, uid, wid, msg, args, extract = ledger
    from app.config import settings
    monkeypatch.setattr(settings, "enable_memory_prefilter", True)
    extract.return_value = {"memories": [], "_provenance": {"stage": "extraction", "template_sha256": "a"*64}}
    if mode == "heuristic": monkeypatch.setattr(pipeline, "should_extract_memory", lambda _: False)
    if mode == "prefilter": monkeypatch.setattr(pipeline, "should_memorize", AsyncMock(return_value=False))
    assert await pipeline.process_memory_pipeline(**args) == []
    assert await pipeline.process_memory_pipeline(**args) == []
    rows = await db.query_raw("SELECT pipeline_version,processing_outcome,outcome_reason,extraction_provenance,extraction FROM memory_extraction_batches WHERE conversation_id=$1", msg.conversationId)
    assert len(rows) == 1
    row = rows[0]
    assert row['pipeline_version'] == receipts.PIPELINE_VERSION
    assert row['processing_outcome'] == expected and row['outcome_reason'] == reason
    assert row['extraction'] is None and row['extraction_provenance']['stage'] == ('extraction' if mode=='empty' else mode)
    assert extract.await_count == (1 if mode == 'empty' else 0)


async def test_completed_source_version_cannot_silently_cover_an_edit(ledger):
    db, _, _, msg, args, extract = ledger
    extract.return_value = {"memories": []}
    await pipeline.process_memory_pipeline(**args)
    await db.message.update(where={'id': msg.id}, data={'content': '后续编辑的不同事实'})
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**dict(args, new_conversation='user: 后续编辑的不同事实'))
    report = await receipt_read.recent_message_processing(limit=50)
    row = next(r for r in report['records'] if r['message_id'] == msg.id)
    assert row['outcome'] == 'needs_reconcile'
    assert extract.await_count == 1


async def test_failed_extraction_is_not_confused_with_no_fact(ledger):
    db, _, _, msg, args, extract = ledger
    extract.return_value = {'_extraction_error': True, '_extraction_error_kind':'PRIVATE_ERROR'}
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    report = await receipt_read.recent_message_processing(limit=50)
    row = next(r for r in report['records'] if r['message_id'] == msg.id)
    assert row['outcome'] == 'retryable_failed'
    assert row['batch_fact_count'] is None and 'PRIVATE' not in str(report)
    extract.return_value = {'memories': []}
    await pipeline.process_memory_pipeline(**args)
    report = await receipt_read.recent_message_processing(limit=50)
    assert next(r for r in report['records'] if r['message_id']==msg.id)['outcome'] == 'no_fact'


async def test_policy_skip_is_recorded_but_cannot_claim_an_eligible_message(chat):
    from prisma import Json
    db, uid, wid, msg = chat
    kwargs=dict(user_id=uid,workspace_id=wid,conversation_id=msg.conversationId,side='user',message_ids=[msg.id])
    with pytest.raises(receipts.MemoryReceiptError):
        await receipts.record_policy_skips(**kwargs)
    await db.message.update(where={'id':msg.id},data={'metadata':Json({'human_takeover':True})})
    await receipts.record_policy_skips(**kwargs)
    await receipts.record_policy_skips(**kwargs)
    rows=await db.query_raw('SELECT b.processing_outcome,b.outcome_reason FROM memory_extraction_sources s JOIN memory_extraction_batches b ON b.id=s.batch_id WHERE s.message_id=$1',msg.id)
    assert rows==[{'processing_outcome':'policy_skipped','outcome_reason':'reserved_messages'}]
    pipeline.extract_memories.assert_not_awaited()
    await db.message.update(where={'id': msg.id}, data={'metadata': Json({})})
    record = next(r for r in (await receipt_read.recent_message_processing(limit=50))['records']
                  if r['message_id'] == msg.id)
    assert record['outcome'] == 'needs_reconcile'
    assert not any(k.startswith('_') for k in record)


async def test_completed_receipt_does_not_cover_a_new_exclusion_flag(ledger):
    from prisma import Json
    db, _, _, msg, args, extract = ledger
    extract.return_value = {'memories': []}
    await pipeline.process_memory_pipeline(**args)
    await db.message.update(where={'id': msg.id}, data={'metadata': Json({'real_world_type': 'gift'})})
    record = next(r for r in (await receipt_read.recent_message_processing(limit=50))['records']
                  if r['message_id'] == msg.id)
    assert record['outcome'] == 'needs_reconcile'


async def test_expanded_bubble_history_does_not_expand_fact_eligibility(chat):
    db,uid,wid,msg=chat
    rows=[{'id':msg.id,'role':'user','content':msg.content,'createdAt':msg.createdAt}]
    for role,content in [('assistant','先问你一个问题'),('assistant','你最近喜欢哪本书？'),
                         ('user','我喜欢合成书'),('assistant','知道了'),('assistant','这本书很有趣'),
                         ('user','今天开始重读'),('assistant','慢慢看'),('assistant','读完再聊')]:
        row=await db.message.create(data={'conversationId':msg.conversationId,'role':role,'content':content,
                                         'createdAt':msg.createdAt+timedelta(seconds=len(rows))})
        rows.append({'id':row.id,'role':role,'content':content,'createdAt':row.createdAt})
    await post._bg_memory_pipeline(uid,rows,msg.conversationId,wid)
    user_call=next(c for c in pipeline.extract_memories.await_args_list if c.kwargs['side']=='user')
    assert msg.content not in user_call.args[0]
    assert '你最近喜欢哪本书' in user_call.kwargs['context_conversation']
    assert 'user: 我喜欢合成书' in user_call.args[0]
    assert 'user: 我喜欢合成书' not in user_call.kwargs['context_conversation']
    assert await db.query_raw('SELECT message_id FROM memory_extraction_sources WHERE message_id=$1',msg.id)==[]


@pytest.mark.parametrize('proactive,gap',[(True,1),(False,601)])
def test_history_and_previous_question_honor_existing_turn_boundaries(proactive,gap):
    from datetime import datetime,timezone
    base=datetime(2026,10,26,tzinfo=timezone.utc)
    rows=[{'id':'a','role':'assistant','content':'旧提问？','createdAt':base},
          {'id':'b','role':'assistant','content':'新话题','createdAt':base+timedelta(seconds=gap),'metadata':{'proactive':proactive}},
          {'id':'u','role':'user','content':'好','createdAt':base+timedelta(seconds=gap+1)}]
    assert post._ai_line_before(rows,rows[-1])=='新话题'
    assert format_history(rows).count('assistant:')==2
    assert post._ai_line_before(rows,{'id':'not-in-window'})==''


async def test_coverage_admin_permissions_bounds_and_failure_are_explicit(ledger,monkeypatch):
    from app.api.admin.runtime_jobs import router
    from app.services.auth import create_jwt
    db, uid, _, msg, args, extract=ledger
    app=FastAPI();app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app),base_url='http://fixture') as client:
        path='/admin-api/runtime-jobs/memory-message-processing'
        assert (await client.get(path)).status_code==401
        assert (await client.get(path,headers={'Authorization':'Bearer '+create_jwt(uid,role='user')})).status_code==403
        await db.user.update(where={'id':uid},data={'role':'admin'})
        h={'Authorization':'Bearer '+create_jwt(uid,role='admin')}
        for query in ('?limit=0','?limit=51','?hours=169'):
            assert (await client.get(path+query,headers=h)).status_code==422
        row=(await client.get(path,headers=h)).json()['records'][0]
        assert row['outcome']=='unrecorded' and row['pipeline_version'] is None
        assert 'content' not in row
        monkeypatch.setattr(receipt_read,'recent_message_processing',AsyncMock(side_effect=RuntimeError('SECRET')))
        response=await client.get(path,headers=h)
        assert response.status_code==503 and 'SECRET' not in response.text


def test_history_is_bounded_without_mutating_source_rows():
    rows=[{'id':str(i),'role':'user' if i%4==0 else 'assistant','content':'x'*10000} for i in range(1000)]
    selected=history_rows(rows)
    assert len(selected)==12
    assert len(format_history(selected))<=8000
    assert rows[-1]['content']=='x'*10000


async def test_same_timestamp_late_commit_is_discovered_without_historic_backfill(chat):
    from app.services.memory.recording import watermark as wm
    db,uid,wid,msg=chat
    # The first admitted window establishes a floor; older untouched history
    # must never be pulled into this repair.
    old=await db.message.create(data={'conversationId':msg.conversationId,'role':'user',
        'content':'此前未纳入的新旧历史边界','createdAt':msg.createdAt-timedelta(days=10)})
    await post._bg_memory_pipeline(uid,[],msg.conversationId,wid,source_message_ids=[msg.id])
    late=await db.message.create(data={'id':'000-late-'+msg.id,'conversationId':msg.conversationId,
        'role':'user','content':'同时间戳迟提交的另一条事实','createdAt':msg.createdAt})
    result=await post._pipeline_with_watermark(uid,[],msg.conversationId,side='user',workspace_id=wid)
    assert result.status=='completed' and result.source_count==1
    assert result.progress=='unchanged'  # SQL receipt, not cursor advancement, proves it.
    assert pipeline.extract_memories.await_args.args[0]=='user: '+late.content
    claimed=await db.query_raw('SELECT message_id FROM memory_extraction_sources WHERE message_id=ANY($1::text[])',[old.id,late.id])
    assert claimed==[{'message_id':late.id}]
    again=await post._pipeline_with_watermark(uid,[],msg.conversationId,side='user',workspace_id=wid)
    assert again.status=='no_new_messages'


@pytest.mark.parametrize('count',[10,30,31,100,1000])
async def test_bounded_pages_cover_each_source_once_with_durable_receipts(chat,count):
    from app.services.memory.recording import watermark as wm
    db,uid,wid,msg=chat
    stamp=msg.createdAt+timedelta(seconds=1)
    rows=[{'id':f'{i:04}-{msg.id}','conversationId':msg.conversationId,'role':'user',
           'content':f'合成事实 {i}','createdAt':stamp} for i in range(count)]
    await db.message.create_many(data=rows)
    await wm.set_watermark(msg.conversationId,'user',msg.createdAt)
    for n in range((count+29)//30):
        result=await post._pipeline_with_watermark(uid,[],msg.conversationId,side='user',workspace_id=wid)
        assert result.status=='completed' and 1<=result.source_count<=30
    result=await post._pipeline_with_watermark(uid,[],msg.conversationId,side='user',workspace_id=wid)
    assert result.status=='no_new_messages'
    claimed=await db.query_raw('''SELECT s.message_id FROM memory_extraction_sources s
        JOIN memory_extraction_batches b ON b.id=s.batch_id WHERE b.conversation_id=$1 AND b.side='user' ''',msg.conversationId)
    assert len(claimed)==count and {r['message_id'] for r in claimed}=={r['id'] for r in rows}
    assert pipeline.extract_memories.await_count==(count+29)//30


async def test_concurrent_filter_rejection_cannot_discard_the_winning_frozen_facts(ledger,monkeypatch):
    import asyncio
    from app.config import settings
    monkeypatch.setattr(settings,'enable_memory_prefilter',True)
    db,_,_,msg,args,extract=ledger
    second_entered=asyncio.Event(); frozen=asyncio.Event(); calls=0
    async def prefilter(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==1: return True
        second_entered.set();await frozen.wait();return False
    payload=extract.return_value
    async def extraction(*a,**kw):
        await second_entered.wait();return payload
    freeze=receipts.freeze
    async def freeze_candidate(*a,**kw):
        result=await freeze(*a,**kw);frozen.set();return result
    monkeypatch.setattr(pipeline,'should_memorize',prefilter)
    extract.side_effect=extraction
    monkeypatch.setattr(receipts,'freeze',freeze_candidate)
    async with asyncio.timeout(15):
        first,second=await asyncio.gather(pipeline.process_memory_pipeline(**args),pipeline.process_memory_pipeline(**args))
    assert len(first)==2 and set(first)==set(second)
    rows=await db.query_raw('SELECT processing_outcome FROM memory_extraction_batches WHERE conversation_id=$1',msg.conversationId)
    assert rows==[{'processing_outcome':'processed'}]
    assert extract.await_count==1


async def test_late_commit_failure_remains_recoverable_behind_the_fast_cursor(chat):
    db, uid, wid, msg = chat
    await post._bg_memory_pipeline(uid, [], msg.conversationId, wid, source_message_ids=[msg.id])
    late = await db.message.create(data={
        'id': '000-retry-' + msg.id, 'conversationId': msg.conversationId,
        'role': 'user', 'content': '迟提交后第一次抽取失败', 'createdAt': msg.createdAt,
    })
    pipeline.extract_memories.return_value = {'_extraction_error': True}
    failed = await post._pipeline_with_watermark(uid, [], msg.conversationId, side='user', workspace_id=wid)
    assert failed.status == 'failed' and failed.progress == 'unchanged'
    pipeline.extract_memories.return_value = {'memories': []}
    outcome = await post._pipeline_with_watermark(uid, [], msg.conversationId, side='user', workspace_id=wid)
    assert outcome.status == 'completed' and outcome.source_count == 1
    rows = await db.query_raw('''SELECT b.processing_outcome FROM memory_extraction_sources s
        JOIN memory_extraction_batches b ON b.id=s.batch_id WHERE s.message_id=$1''', late.id)
    assert rows == [{'processing_outcome': 'no_fact'}]


async def test_audit_rotates_bounded_pages_and_revisits_late_commits(chat):
    from app.services.memory.recording.message_audit import unrecorded_since_admission, SCAN_PAGE_SIZE
    db, uid, wid, msg = chat
    await post._bg_memory_pipeline(uid, [], msg.conversationId, wid, source_message_ids=[msg.id])
    # An opposite-side backlog must not make the user-side scan unbounded.
    await db.message.create_many(data=[{
        'id': f'audit-{i:04}-{msg.id}', 'conversationId': msg.conversationId,
        'role': 'assistant', 'content': '合成对侧消息',
        'createdAt': msg.createdAt + timedelta(seconds=i+1),
    } for i in range(SCAN_PAGE_SIZE*2)])
    assert await unrecorded_since_admission(msg.conversationId, 'user', database=db) == []
    cursor = (await db.query_raw('SELECT after_at FROM memory_message_audit_cursors'))[0]
    assert post._parse_ts({'createdAt': cursor['after_at']}) == msg.createdAt + timedelta(seconds=SCAN_PAGE_SIZE-1)
    late = await db.message.create(data={
        'id': '000-audit-' + msg.id, 'conversationId': msg.conversationId,
        'role': 'user', 'content': '落在补查位置之前的迟提交', 'createdAt': msg.createdAt,
    })
    found = []
    for _ in range(4):
        found = await unrecorded_since_admission(msg.conversationId, 'user', database=db)
        if found:
            break
    assert [row['id'] for row in found] == [late.id]
    # A returned-but-not-yet-processed hole stays discoverable.
    assert [row['id'] for row in await unrecorded_since_admission(
        msg.conversationId, 'user', database=db)] == [late.id]


async def test_legacy_receipt_endpoint_keeps_its_error_contract(ledger):
    db, _, _, msg, args, extract = ledger
    extract.return_value = {'_extraction_error': True, '_extraction_error_kind': 'prompt_disabled'}
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    records = (await receipt_read.recent_receipts(limit=50))['records']
    row = next(row for row in records if row['conversation_id'] == msg.conversationId)
    assert row['last_error'] == 'processing_failed'
    detail = (await receipt_read.recent_message_processing(limit=50))['records']
    assert next(row for row in detail if row['message_id'] == msg.id)['last_error'] == 'prompt_disabled'


async def test_stalled_monitor_counts_scope_age_and_bounds(ledger):
    db, uid, wid, msg, args, extract = ledger
    extract.return_value = {'_extraction_error': True}
    with pytest.raises(pipeline.MemoryExtractionError):
        await pipeline.process_memory_pipeline(**args)
    batch = (await db.query_raw('SELECT id FROM memory_extraction_batches WHERE conversation_id=$1',msg.conversationId))[0]['id']
    assert (await receipt_read.stalled_batch_monitor())['stale_failed_batches'] == 0
    for i in range(9):
        await db.execute_raw('''INSERT INTO memory_extraction_batches
            (id,user_id,workspace_id,conversation_id,side,scope,sources,status,last_error,updated_at)
            SELECT id||$2,user_id,workspace_id,conversation_id,CASE WHEN $3%2=0 THEN 'user' ELSE 'ai' END,scope,sources,
                CASE WHEN $3%2=0 THEN 'preparing' ELSE 'processing' END,'PRIVATE_SECRET',
                CURRENT_TIMESTAMP-(($3+3)::int*INTERVAL '1 hour')
            FROM memory_extraction_batches WHERE id=$1''',batch,f'-{i}',i)
    result = await receipt_read.stalled_batch_monitor()
    assert result['status']=='measured'
    assert result['unfinished_batches']==result['failed_batches']==10
    assert result['stale_failed_batches']==9 and len(result['oldest'])==5
    assert [r['batch_id'] for r in result['oldest']]==[batch+f'-{i}' for i in range(8,3,-1)]
    assert all(r['reason']=='processing_failed' and r['updated_at'].endswith('+00:00') for r in result['oldest'])
    assert 'PRIVATE' not in str(result) and msg.content not in str(result)
    await db.execute_raw("UPDATE memory_extraction_batches SET status='blocked' WHERE id=$1",batch+'-8')
    await db.execute_raw("UPDATE memory_extraction_batches SET status='completed' WHERE id=$1",batch+'-7')
    await db.execute_raw("UPDATE memory_extraction_batches SET last_error=NULL WHERE id=$1",batch+'-6')
    result = await receipt_read.stalled_batch_monitor()
    assert (result['unfinished_batches'],result['failed_batches'],result['stale_failed_batches'])==(8,7,6)
    await db.conversation.update(where={'id':msg.conversationId},data={'isDeleted':True})
    assert (await receipt_read.stalled_batch_monitor())['unfinished_batches']==0
    await db.conversation.update(where={'id':msg.conversationId},data={'isDeleted':False})
    await db.chatworkspace.update(where={'id':wid},data={'status':'archived'})
    assert (await receipt_read.stalled_batch_monitor())['unfinished_batches']==0


async def test_monitor_failure_does_not_fake_zero_or_hide_message_coverage(ledger,monkeypatch):
    from unittest.mock import Mock
    db, _, _, msg, *_ = ledger
    proxy=Mock(query_raw=db.query_raw)
    proxy.tx.side_effect=RuntimeError('PRIVATE_SQL_CONNECTION')
    monkeypatch.setattr(receipt_read,'db',proxy)
    report=await receipt_read.recent_message_processing()
    assert report['records'][0]['message_id']==msg.id
    assert report['stalled_batches']['status']=='unavailable'
    assert 'stale_failed_batches' not in report['stalled_batches'] and 'PRIVATE' not in str(report)


async def test_audit_and_message_coverage_do_not_depend_on_pg_timezone(ledger,monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from app.services.memory.recording.message_audit import unrecorded_since_admission, SCAN_PAGE_SIZE
    db, _, _, msg, args, extract = ledger
    extract.return_value={'memories': []}
    await pipeline.process_memory_pipeline(**args)
    await db.message.create_many(data=[dict(id=f'tz-{i:04}-{msg.id}',conversationId=msg.conversationId,
        role='assistant',content='合成对侧',createdAt=msg.createdAt+timedelta(seconds=i+1)) for i in range(SCAN_PAGE_SIZE)])
    old=await db.message.create(data=dict(conversationId=msg.conversationId,role='user',content='25小时前',createdAt=msg.createdAt-timedelta(hours=25)))
    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        @asynccontextmanager
        async def same_transaction(): yield tx
        database=SimpleNamespace(tx=same_transaction,query_raw=tx.query_raw)
        monkeypatch.setattr(receipt_read,'db',database)
        results=[]
        for zone in ('UTC','Asia/Shanghai'):
            await tx.execute_raw("SELECT set_config('TimeZone',$1,true)",zone)
            await tx.execute_raw('DELETE FROM memory_message_audit_cursors')
            await unrecorded_since_admission(msg.conversationId,'user',database=database)
            cursor=(await tx.query_raw('SELECT after_at FROM memory_message_audit_cursors'))[0]
            assert receipt_read._utc_iso(cursor['after_at'])==receipt_read._utc_iso(msg.createdAt+timedelta(seconds=SCAN_PAGE_SIZE-1))
            # Exercise cursor comparison as well as the admission floor.
            assert await unrecorded_since_admission(msg.conversationId,'user',database=database)==[]
            report=await receipt_read.recent_message_processing(limit=50)
            assert old.id not in {r['message_id'] for r in report['records']}
            # A one-hour sample with late rows removed includes the exact receipt.
            await tx.execute_raw("DELETE FROM messages WHERE id LIKE $1",'tz-%-'+msg.id)
            report=await receipt_read.recent_message_processing(limit=50)
            row=next(r for r in report['records'] if r['message_id']==msg.id)
            assert row['source_matches'] and row['outcome']=='no_fact'
            assert row['message_at']==receipt_read._utc_iso(msg.createdAt)
            results.append(row)
            if zone=='UTC':
                await tx.execute_raw('''INSERT INTO messages(id,conversation_id,role,content,created_at)
                    SELECT 'tz-'||lpad(i::text,4,'0')||'-'||$1,$2,'assistant','合成对侧',
                    ($3::timestamptz AT TIME ZONE 'UTC')+((i+1)*INTERVAL '1 second') FROM generate_series(0,$4-1) i''',msg.id,msg.conversationId,msg.createdAt,SCAN_PAGE_SIZE)
        assert results[0]==results[1]


async def test_cursor_erasure_resets_discovery_not_completion(chat):
    from app.services.memory.recording.message_audit import unrecorded_since_admission
    db,uid,wid,msg=chat
    await post._bg_memory_pipeline(uid, [], msg.conversationId, wid, source_message_ids=[msg.id])
    await unrecorded_since_admission(msg.conversationId,'user',database=db)
    assert len(await db.query_raw('SELECT * FROM memory_message_audit_cursors'))==1
    await db.execute_raw('DELETE FROM memory_extraction_batches WHERE conversation_id=$1',msg.conversationId)
    assert await db.query_raw('SELECT * FROM memory_message_audit_cursors')==[]
    # No surviving admission is not permission to backfill history.
    assert await unrecorded_since_admission(msg.conversationId,'user',database=db)==[]
    assert (await receipt_read.recent_message_processing())['records'][0]['outcome']=='unrecorded'
