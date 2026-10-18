"""Failure evidence must not masquerade as successful empty memory extraction."""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.chat import post_process as p
from app.services.memory.recording import diagnostics as d, watermark as wm
from app.services.memory.recording.outcomes import MemoryProgressError, MemorySideOutcome
from app.services.memory.recording.pipeline import MemoryExtractionError
from tests.test_memory_evidence_postgres import origins


@pytest.fixture
def unit(monkeypatch):
    monkeypatch.setattr(d, '_write', AsyncMock())
    monkeypatch.setattr(p, '_resolve_workspace_id_for_conversation', AsyncMock(return_value='workspace'))
    monkeypatch.setattr(p, 'get_watermark', AsyncMock(return_value=None))
    monkeypatch.setattr(p, 'set_watermark', AsyncMock())
    monkeypatch.setattr(p, 'process_memory_pipeline', AsyncMock(return_value=[]))
    monkeypatch.setattr(p.manager, 'send_event', AsyncMock())
    return [{'id': 'message', 'role': 'user', 'content': 'Synthetic fact', 'createdAt': datetime.now(UTC)}]


@pytest.mark.parametrize('kind,status', [('extraction_failed','failed'), ('prompt_disabled','disabled'), ('source_invalid','failed')])
async def test_failure_and_disabled_are_not_successful_empty(unit, monkeypatch, kind, status):
    p.process_memory_pipeline.side_effect = MemoryExtractionError('PRIVATE_CHAT_AND_TOKEN', kind=kind)
    result = await p._bg_memory_pipeline('user', unit, 'conversation', 'workspace')
    assert result[0] == MemorySideOutcome('user', status, kind)
    assert result[1].status == 'no_new_messages'
    p.set_watermark.assert_not_awaited()
    assert 'PRIVATE' not in str(d._write.await_args_list)
    assert d._write.await_args.args[0]['status'] == 'incomplete'


async def test_successful_empty_is_completed_and_advances(unit):
    result = await p._bg_memory_pipeline('user', unit, 'conversation', 'workspace')
    assert result[0] == MemorySideOutcome('user', 'completed', 'processed', 0, 'advanced')
    p.set_watermark.assert_awaited_once()
    assert d._write.await_args.args[0]['status'] == 'completed'


async def test_write_then_progress_failure_preserves_known_count(unit):
    p.process_memory_pipeline.return_value = ['one', 'two']
    p.set_watermark.side_effect = MemoryProgressError('watermark_write_unavailable')
    result = await p._bg_memory_pipeline('user', unit, 'conversation', 'workspace')
    assert result[0] == MemorySideOutcome('user','failed','watermark_write_unavailable',2,'unknown')
    p.process_memory_pipeline.assert_awaited_once()
    p.manager.send_event.assert_awaited_once()


async def test_partial_storage_failure_has_unknown_count_and_no_advancement(unit):
    p.process_memory_pipeline.side_effect = RuntimeError('PARTIAL_WRITE_PRIVATE')
    result = await p._bg_memory_pipeline('user', unit, 'conversation', 'workspace')
    assert result[0].memory_count is None and result[0].reason == 'processing_failed'
    p.set_watermark.assert_not_awaited()
    assert 'PRIVATE' not in str(d._write.await_args_list)


async def test_side_failure_keeps_conversation_lock_until_other_side_finishes(unit, monkeypatch):
    started=asyncio.Event(); release=asyncio.Event(); calls=[]
    async def side(*args, side, **kwargs):
        calls.append(side)
        if side == 'ai':
            started.set(); await release.wait()
            return MemorySideOutcome('ai','completed','processed',1)
        await started.wait()
        raise RuntimeError('one side failed')
    monkeypatch.setattr(p,'_pipeline_with_watermark',side)
    a=asyncio.create_task(p._bg_memory_pipeline('user',unit,'serial-conv','workspace'))
    await started.wait()
    b=asyncio.create_task(p._bg_memory_pipeline('user',unit,'serial-conv','workspace'))
    await asyncio.sleep(.01)
    assert not a.done() and not b.done() and calls == ['user','ai']
    release.set(); first,second=await asyncio.gather(a,b)
    assert first[0].status=='failed' and first[1].status=='completed'
    assert len(calls)==4


async def test_cancellation_drains_children_before_releasing_lock(unit,monkeypatch):
    started=asyncio.Event(); cleanup=asyncio.Event(); cleaned=asyncio.Event()
    async def side(*args,side,**kwargs):
        if side=='user':return MemorySideOutcome(side,'no_new_messages','no_new_messages',0)
        started.set()
        try:await asyncio.Future()
        finally:
            await cleanup.wait(); cleaned.set()
    monkeypatch.setattr(p,'_pipeline_with_watermark',side)
    task=asyncio.create_task(p._bg_memory_pipeline('user',unit,'cancel-conv','workspace'))
    await started.wait();task.cancel();await asyncio.sleep(.01)
    assert p._get_pipeline_lock('cancel-conv').locked() and not cleaned.is_set()
    cleanup.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert cleaned.is_set() and not p._get_pipeline_lock('cancel-conv').locked()
    assert d._write.await_args.args[0]['status']=='cancelled'


@pytest.mark.parametrize('raw',[b'bad-time',object()])
async def test_corrupt_watermark_is_not_missing(monkeypatch,raw):
    monkeypatch.setattr(wm,'get_redis',AsyncMock(return_value=SimpleNamespace(get=AsyncMock(return_value=raw))))
    with pytest.raises(MemoryProgressError,match='watermark_invalid'):await wm.get_watermark('conv','user')


async def test_missing_watermark_is_distinct_from_read_failure(monkeypatch):
    monkeypatch.setattr(wm,'get_redis',AsyncMock(return_value=SimpleNamespace(get=AsyncMock(return_value=None))))
    assert await wm.get_watermark('conv','user') is None


@pytest.mark.parametrize('mutation,reason',[('user','scope_invalid'),('workspace','scope_invalid'),('missing','scope_invalid'),('deleted','scope_invalid'),('unavailable','scope_unavailable')])
async def test_real_conversation_scope_cannot_fall_back_to_another_agent(origins,monkeypatch,mutation,reason):
    db,uid,wid,_,msg,*_=origins
    monkeypatch.setattr(p,'db',db)
    monkeypatch.setattr(d,'_write',AsyncMock())
    pipeline=AsyncMock();monkeypatch.setattr(p,'process_memory_pipeline',pipeline)
    if mutation=='deleted':await db.conversation.update(where={'id':msg.conversationId},data={'isDeleted':True})
    if mutation=='unavailable':monkeypatch.setattr(p,'db',SimpleNamespace(conversation=SimpleNamespace(find_unique=AsyncMock(side_effect=RuntimeError('private')))))
    results=await p._bg_memory_pipeline('different-user' if mutation=='user' else uid,
        [{'role':'user','content':'private'}], 'missing' if mutation=='missing' else msg.conversationId,
        'different-workspace' if mutation=='workspace' else wid)
    assert all(r.reason==reason and r.status=='failed' for r in results)
    pipeline.assert_not_awaited()


async def test_real_scope_is_resolved_and_validated(origins,monkeypatch):
    db,uid,wid,_,msg,*_=origins;monkeypatch.setattr(p,'db',db)
    assert await p._resolve_workspace_id_for_conversation(msg.conversationId,uid,None)==wid
    assert await p._resolve_workspace_id_for_conversation(msg.conversationId,uid,wid)==wid


async def test_bounded_page_never_advances_over_an_incomplete_timestamp_cohort(origins,monkeypatch):
    db,uid,wid,_,msg,*_=origins;monkeypatch.setattr(p,'db',db)
    await db.message.delete_many(where={'conversationId':msg.conversationId})
    base=datetime(2026,10,1,tzinfo=UTC)
    for i in range(31):
        await db.message.create(data={'conversationId':msg.conversationId,'role':'user','content':'synthetic',
            'createdAt':base+timedelta(seconds=0 if i<20 else 1)})
    rows=await p._fetch_new_side_messages(msg.conversationId,base-timedelta(seconds=1),'user')
    assert len(rows)==20 and all(p._parse_ts(r)==base for r in rows)
    rows=await p._fetch_new_side_messages(msg.conversationId,base,'user')
    assert len(rows)==11
    await db.message.update_many(where={'conversationId':msg.conversationId},data={'createdAt':base})
    with pytest.raises(MemoryProgressError,match='timestamp_page_boundary'):
        await p._fetch_new_side_messages(msg.conversationId,base-timedelta(seconds=1),'user')


async def test_recent_synthetic_reply_cannot_jump_over_pending_page(unit,monkeypatch):
    base=datetime.now(UTC)-timedelta(minutes=10)
    monkeypatch.setattr(p,'get_watermark',AsyncMock(return_value=base-timedelta(minutes=1)))
    monkeypatch.setattr(p,'_fetch_new_side_messages',AsyncMock(return_value=[
        {'id':str(i),'role':'assistant','content':f'older {i}','createdAt':base+timedelta(seconds=i)} for i in range(30)]))
    result=await p._pipeline_with_watermark('user',[{'role':'assistant','content':'LATEST_SYNTHETIC'}],'conv',side='ai',workspace_id='workspace')
    assert 'LATEST_SYNTHETIC' not in p.process_memory_pipeline.await_args.kwargs['new_conversation']
    assert p.set_watermark.await_args.args[2]==base+timedelta(seconds=29)
    assert result.progress=='advanced'


async def test_redis_outage_retains_safe_log_and_never_reports_zero_success(unit,monkeypatch,caplog):
    p.get_watermark.side_effect=MemoryProgressError('watermark_read_unavailable')
    with caplog.at_level('WARNING'):
        result=await p._bg_memory_pipeline('user',unit,'conv','workspace')
    row=next(r for r in caplog.records if getattr(r,'kind',None)=='memory_pipeline')
    assert row.outcome=='incomplete' and row.n_stored_total is None
    assert all(r['reason']=='watermark_read_unavailable' for r in row.side_outcomes)
    p.process_memory_pipeline.assert_not_awaited()


async def test_intentionally_skipped_progress_write_failure_is_unknown(unit):
    unit.append({'role':'assistant','content':'synthetic skipped'})
    p.set_watermark.side_effect=MemoryProgressError('watermark_write_unavailable')
    result=await p._bg_memory_pipeline('user',unit,'conv','workspace',skip_ai_side=True)
    assert result[1]==MemorySideOutcome('ai','failed','watermark_write_unavailable',0,'unknown')


async def test_no_conversation_side_runs_and_refresh_failure_does_not_change_success(unit):
    result=await p._bg_memory_pipeline('user',unit,None,'workspace',skip_ai_side=True)
    assert result[0].status=='completed' and result[0].progress=='not_applicable'
    assert result[1].status=='excluded' and result[1].progress=='not_applicable'
    p.set_watermark.assert_not_awaited()
    p.process_memory_pipeline.return_value=['memory']
    p.manager.send_event.side_effect=RuntimeError('PRIVATE_NOTIFICATION')
    result=await p._bg_memory_pipeline('user',unit,'conv','workspace')
    assert result[0].status=='completed' and result[0].memory_count==1


async def test_distributed_lock_failure_is_observed_and_does_not_process(unit,monkeypatch,caplog):
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def unavailable(*args,**kwargs):
        assert kwargs['fail_open'] is False
        raise p.DistributedLockUnavailable('PRIVATE_REDIS')
        yield
    monkeypatch.setattr(p,'settings',SimpleNamespace(is_production=lambda:True))
    monkeypatch.setattr(p,'distributed_lock',unavailable)
    with caplog.at_level('WARNING'):
        results=await p._bg_memory_pipeline('user',unit,'conv','workspace')
    assert all(r.reason=='lock_unavailable' for r in results)
    assert 'PRIVATE' not in caplog.text
    assert any(getattr(r,'outcome',None)=='incomplete' for r in caplog.records)
    p.process_memory_pipeline.assert_not_awaited()
