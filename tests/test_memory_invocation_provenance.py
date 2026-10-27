"""Memory audit captures successful routes, without altering model responses."""
import asyncio
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.services.llm import models, resilience
from app.services.memory.recording import pre_filter, extraction, pipeline
from tests.test_memory_receipts_postgres import ledger, origins


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('fail_primary,fail_fallback', [(False,False),(True,False),(True,True)])
async def test_only_successful_route_is_recorded(monkeypatch,stream,fail_primary,fail_fallback):
    resilience.reset_breakers_for_testing()
    monkeypatch.setattr(models.settings,'llm_resilience_enabled',True)
    profile=resilience.CallProfile(timeout_s=1, max_retries=0,retry_backoff_s=(), stream_mode=stream,
                                   allow_ollama_fallback=True)
    seen=[]
    def model(name,fail):
        async def invoke(messages,**kwargs):
            seen.append(kwargs)
            if fail: raise RuntimeError('synthetic provider failure')
            return AIMessage(content='{"memories":[]}')
        async def chunks(messages,**kwargs):
            seen.append(kwargs)
            if fail: raise RuntimeError('synthetic provider failure')
            yield AIMessageChunk(content='{"memories":[]}')
        return SimpleNamespace(model_name=name,ainvoke=invoke,astream=chunks)
    primary,fallback=model('primary',fail_primary),model('fallback',fail_fallback)
    monkeypatch.setattr(models,'_provider_and_profile',lambda *a:('test',profile))
    monkeypatch.setattr(models,'get_fallback_chat_model',lambda:fallback)
    monkeypatch.setattr(models,'_resolve_usage_model_key',lambda m:'test/'+m.model_name)
    route={'actual_model':'stale'}
    try:
        if fail_primary and fail_fallback:
            with pytest.raises(resilience.LLMFailedError):
                await models.invoke_json(primary,'test',invocation_metadata=route)
            assert route=={}
        else:
            assert await models.invoke_json(primary,'test',invocation_metadata=route)=={'memories':[]}
            assert route=={'actual_model':'test/'+('fallback' if fail_primary else 'primary')}
        assert all('invocation_metadata' not in kwargs for kwargs in seen)
    finally:
        resilience.reset_breakers_for_testing()
        resilience.reset_slots_for_testing()


async def test_concurrent_invocations_do_not_share_metadata(monkeypatch):
    monkeypatch.setattr(models.settings,'llm_resilience_enabled',True)
    profile=resilience.CallProfile(timeout_s=1,max_retries=0,retry_backoff_s=(),allow_ollama_fallback=False)
    monkeypatch.setattr(models,'_provider_and_profile',lambda *a:('ollama',profile))
    monkeypatch.setattr(models,'_resolve_usage_model_key',lambda m:m.model_name)
    barrier=asyncio.Event()
    async def invoke(messages,**kwargs):
        await barrier.wait()
        return AIMessage(content='记')
    first,second={},{}
    calls=[asyncio.create_task(models.invoke_text(SimpleNamespace(model_name=n,ainvoke=invoke),
        'test',invocation_metadata=out)) for n,out in [('one',first),('two',second)]]
    barrier.set()
    assert await asyncio.gather(*calls)==['记','记']
    assert first=={'actual_model':'one'} and second=={'actual_model':'two'}


@pytest.mark.parametrize('answer,decision,result',[('不记','do_not_memorize',False),('记','memorize',True),('unknown','unrecognized_fail_open',True)])
async def test_prefilter_records_prompt_and_actual_route(monkeypatch,answer,decision,result):
    template='判定 {message} 前文 {prev_ai}'
    monkeypatch.setattr(pre_filter,'get_prompt_text',AsyncMock(return_value=template))
    monkeypatch.setattr(pre_filter,'get_utility_model',lambda:SimpleNamespace(model_name='primary'))
    async def invoke(model,prompt,**kwargs):
        kwargs['invocation_metadata']['actual_model']='ollama/fallback'
        return answer
    monkeypatch.setattr(pre_filter,'invoke_text',invoke)
    audit={}
    assert await pre_filter.should_memorize('合成用户消息',provenance=audit) is result
    assert audit['actual_model']=='ollama/fallback'
    assert audit['template_sha256']==sha256(template.encode()).hexdigest()
    assert audit['decision']==decision and audit['model_called'] is True
    assert '合成用户消息' not in str(audit)


async def test_prefilter_rules_and_failure_are_explicit(monkeypatch):
    audit={}
    invoke=AsyncMock(side_effect=RuntimeError('SECRET'))
    monkeypatch.setattr(pre_filter,'invoke_text',invoke)
    monkeypatch.setattr(pre_filter,'get_prompt_text',AsyncMock(return_value='{message}'))
    monkeypatch.setattr(pre_filter,'get_utility_model',lambda:SimpleNamespace(model_name='primary'))
    assert not await pre_filter.should_memorize('我记住了你的名字',side='ai',provenance=audit)
    assert audit['model_called'] is False and audit['decision']=='user_fact_acknowledgement'
    assert await pre_filter.should_memorize('我小时候经常听音乐',side='ai',provenance=audit)
    assert audit['model_called'] is False and audit['decision']=='stable_ai_self_memory'
    assert await pre_filter.should_memorize('这是一个合成事实',provenance=audit)
    assert audit['decision']=='error_fail_open' and audit['actual_model'] is None
    assert 'SECRET' not in str(audit)
    assert invoke.await_count==1


async def test_extraction_cannot_spoof_successful_model(monkeypatch):
    monkeypatch.setattr(extraction,'get_chat_model',lambda:SimpleNamespace(model_name='primary'))
    monkeypatch.setattr(extraction,'get_prompt_text',AsyncMock(return_value='{new_conversation}'))
    async def invoke(model,prompt,**kwargs):
        kwargs['invocation_metadata']['actual_model']='ollama/fallback'
        return {'memories':[],'_provenance':{'actual_model':'spoof'}}
    monkeypatch.setattr(extraction,'invoke_json',invoke)
    result=await extraction.extract_memories('user: 合成事实')
    assert result['_provenance']['actual_model']=='ollama/fallback'


@pytest.mark.parametrize('memorize',[True,False])
async def test_prefilter_provenance_survives_empty_completion(ledger,monkeypatch,memorize):
    from app.config import settings
    db,_,_,msg,args,extract=ledger
    monkeypatch.setattr(settings,'enable_memory_prefilter',True)
    async def judge(*a,**kwargs):
        kwargs['provenance'].update(stage='prefilter',schema_version=1,actual_model='test/utility',template_sha256='a'*64)
        return memorize
    monkeypatch.setattr(pipeline,'should_memorize',judge)
    extract.return_value={'memories':[],'_provenance':{'stage':'extraction','actual_model':'test/chat'}}
    assert await pipeline.process_memory_pipeline(**args)==[]
    record=(await db.query_raw('SELECT extraction_provenance FROM memory_extraction_batches WHERE conversation_id=$1',msg.conversationId))[0]['extraction_provenance']
    audit=record['prefilter'] if memorize else record
    assert audit['actual_model']=='test/utility' and audit['template_sha256']=='a'*64
    assert await pipeline.process_memory_pipeline(**args)==[]
    assert extract.await_count==(1 if memorize else 0)


@pytest.mark.parametrize('failure',[RuntimeError,asyncio.CancelledError])
async def test_partial_stream_does_not_claim_successful_route(monkeypatch,failure):
    resilience.reset_breakers_for_testing()
    monkeypatch.setattr(models.settings,'llm_resilience_enabled',True)
    profile=resilience.CallProfile(timeout_s=1,max_retries=0,retry_backoff_s=(),stream_mode=True,allow_ollama_fallback=False)
    async def chunks(messages,**kwargs):
        yield AIMessageChunk(content='partial')
        raise failure()
    model=SimpleNamespace(model_name='primary',astream=chunks)
    monkeypatch.setattr(models,'_provider_and_profile',lambda *a:('test',profile))
    monkeypatch.setattr(models,'_resolve_usage_model_key',lambda m:m.model_name)
    route={}
    with pytest.raises(asyncio.CancelledError if failure is asyncio.CancelledError else resilience.LLMFailedError):
        await models.invoke_text(model,'test',invocation_metadata=route)
    assert route=={}
    resilience.reset_breakers_for_testing()
    resilience.reset_slots_for_testing()
