"""Cloud failover through real LangChain clients, with synthetic HTTP responses."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from app.config import settings
from app.services.llm import models, resilience, usage_tracker
from app.services import runtime_config as runtime


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(settings, 'llm_resilience_enabled', True)
    resilience.reset_breakers_for_testing()
    resilience.reset_slots_for_testing()
    models._build_fallback_chat_model.cache_clear()
    yield
    resilience.reset_breakers_for_testing()
    resilience.reset_slots_for_testing()
    models._build_fallback_chat_model.cache_clear()


class Stream(httpx.AsyncByteStream):
    def __init__(self, name, behavior):
        self.name, self.behavior, self.closed = name, behavior, False

    async def __aiter__(self):
        if self.behavior == 'timeout':
            await asyncio.sleep(10)
        def event(delta, usage=None):
            return ('data: '+json.dumps({'id':'synthetic','object':'chat.completion.chunk', 'created':1,
                'model':self.name, 'choices':[{'index':0,'delta':delta,'finish_reason':None}],
                **({'usage': usage} if usage else {})})+'\n\n').encode()
        yield event({'role':'assistant', 'content':''})
        if self.behavior == 'empty_fail':
            raise httpx.ReadError('synthetic empty failure')
        yield event({'content':'{"reply":"合成回复"}'})
        if self.behavior == 'mid_fail':
            raise httpx.ReadError('synthetic mid failure')
        yield event({}, {'prompt_tokens':12,'completion_tokens':5,'total_tokens':17,
                         'prompt_tokens_details':{'cached_tokens':4}})
        yield b'data: [DONE]\n\n'

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def clients(monkeypatch, behavior='ok', backup_behavior='ok'):
    calls, streams = [], []
    async def handler(request):
        body=json.loads(request.content); name=body['model']; calls.append(body)
        mode=behavior if name=='primary' else backup_behavior
        if mode=='fail':
            return httpx.Response(503,json={'error':{'message':'synthetic outage','type':'unavailable'}})
        if body.get('stream'):
            stream=Stream(name,mode); streams.append(stream)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=stream)
        return httpx.Response(200,json={'id':'synthetic','object':'chat.completion','created':1,'model':name,
            'choices':[{'index':0,'message':{'role':'assistant','content':'{"reply":"合成回复"}'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':12,'completion_tokens':5,'total_tokens':17,'prompt_tokens_details':{'cached_tokens':4}}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        def model(name, provider):
            value=ChatOpenAI(model=name, api_key='synthetic-only', base_url='http://127.0.0.1:1/v1',
                             http_async_client=client, max_retries=0, stream_usage=True)
            object.__setattr__(value,'_companion_provider',provider)
            return value
        primary,backup=model('primary','ark'),model('backup','dashscope')
        monkeypatch.setattr(models, 'get_fallback_chat_model', lambda:backup)
        yield primary,backup,calls,streams


def profile(stream=False, allow=True):
    return resilience.CallProfile(timeout_s=1,max_retries=0,retry_backoff_s=(),
                                  first_chunk_timeout_s=.05, stream_mode=stream, allow_ollama_fallback=allow)


@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('behavior',['ok','fail'])
async def test_actual_route_usage_and_metadata_in_four_combinations(monkeypatch,stream,behavior):
    async with clients(monkeypatch,behavior) as (primary,backup,calls,streams):
        token=usage_tracker.start_session(); metadata={'actual_model':'stale'}
        try:
            result,usage=await models.invoke_json_with_usage(primary,'synthetic prompt', profile=profile(stream),
                                                            invocation_metadata=metadata,temperature=0)
        finally:
            summary=usage_tracker.flush_session(token)
        actual='ark/primary' if behavior=='ok' else 'dashscope/backup'
        assert result=={'reply':'合成回复'}
        assert metadata=={'actual_model':actual}
        assert summary['tokens_by_model']=={actual:{'input':12,'output':5,'cached_input':4}}
        assert summary['call_count']==1
        assert summary['fallback_count']==(behavior=='fail')
        assert all('invocation_metadata' not in call and 'format' not in call for call in calls)
        assert all(stream.closed for stream in streams)
        assert usage==({} if stream else {'input_tokens':12,'output_tokens':5})


@pytest.mark.parametrize('behavior',['timeout','empty_fail'])
async def test_no_visible_output_switches_to_cloud_and_closes_source(monkeypatch,behavior):
    async with clients(monkeypatch,behavior) as (primary,backup,calls,streams):
        metadata={}
        assert await models.invoke_text(primary,'synthetic',profile=profile(True),invocation_metadata=metadata)
        assert metadata['actual_model']=='dashscope/backup'
        assert [c['model'] for c in calls]==['primary','backup']
        assert all(s.closed for s in streams)
        assert 'dashscope' in resilience._breakers and 'ollama' not in resilience._breakers


@pytest.mark.parametrize('behavior,backup_behavior,allow',[
    ('mid_fail','ok',True),('fail','fail',True),('fail','ok',False),
])
async def test_failure_never_reports_success_or_duplicates_visible_output(monkeypatch,behavior,backup_behavior,allow):
    async with clients(monkeypatch,behavior,backup_behavior) as (primary,backup,calls,streams):
        metadata={'actual_model':'stale'}
        with pytest.raises(resilience.LLMFailedError):
            await models.invoke_text(primary,'synthetic',profile=profile(True,allow),invocation_metadata=metadata)
        assert metadata=={}
        assert [c['model'] for c in calls]==(['primary','backup'] if backup_behavior=='fail' else ['primary'])
        assert all(s.closed for s in streams)


@pytest.mark.parametrize('stream',[False,True])
async def test_cancellation_does_not_call_backup(monkeypatch,stream):
    async with clients(monkeypatch,'timeout') as (primary,backup,calls,streams):
        # Unary cancellation is tested through a blocking primary factory too.
        async def blocked(*args,**kwargs):
            await asyncio.sleep(10)
        if not stream:
            monkeypatch.setattr(type(primary),'ainvoke',blocked)
        metadata={}
        task=asyncio.create_task(models.invoke_text(primary,'synthetic',profile=profile(stream),invocation_metadata=metadata))
        await asyncio.sleep(.01); task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert metadata=={}
        assert not any(c['model']=='backup' for c in calls)
        assert all(s.closed for s in streams)


async def test_main_reply_collect_stream_uses_cloud_provider_and_usage(monkeypatch):
    from app.services.chat import reply_generate
    async with clients(monkeypatch,'fail') as (primary,backup,calls,streams):
        monkeypatch.setattr(reply_generate,'get_chat_model',lambda:primary)
        monkeypatch.setattr(reply_generate,'get_profile',lambda _:profile(True))
        token=usage_tracker.start_session()
        try: text,failed=await reply_generate._run_main_llm([{'role':'user','content':'synthetic'}])
        finally: summary=usage_tracker.flush_session(token)
        assert text and failed is False
        assert set(summary['tokens_by_model'])=={'dashscope/backup'}
        assert 'ollama' not in resilience._breakers


def test_fallback_factory_and_offline_identity_guards(monkeypatch):
    monkeypatch.setattr(settings,'dashscope_api_key','synthetic-only')
    monkeypatch.setattr(settings,'fallback_chat_provider','dashscope')
    monkeypatch.setattr(settings,'fallback_chat_model','qwen3.5-plus')
    fallback=models.get_fallback_chat_model()
    assert isinstance(fallback,ChatOpenAI)
    assert models._resolve_usage_model_key(fallback)=='dashscope/qwen3.5-plus'
    assert models.get_fallback_for(fallback) is None
    assert models.get_fallback_for(ChatOllama(model='synthetic')) is None
    models._build_fallback_chat_model.cache_clear()
    monkeypatch.setattr(settings,'dashscope_api_key','')
    assert models.get_fallback_for(fallback) is None


async def test_missing_backup_configuration_keeps_healthy_primary(monkeypatch):
    async with clients(monkeypatch) as (primary,backup,calls,streams):
        def broken(): raise ValueError('PRIVATE_CONFIGURATION')
        monkeypatch.setattr(models,'get_fallback_chat_model',broken)
        assert await models.invoke_text(primary,'synthetic',profile=profile())
        assert len(calls)==1


@pytest.mark.parametrize('cached',[False,True])
async def test_first_config_failure_blocks_but_refresh_preserves_snapshot(monkeypatch,cached,caplog):
    monkeypatch.setattr(runtime,'_CACHE_LOADED',cached)
    monkeypatch.setattr(runtime,'_GLOBAL_CACHE',{'remoteChatModel':'safe-model'})
    monkeypatch.setattr(runtime,'db',SimpleNamespace(systemconfig=SimpleNamespace(
        find_unique=AsyncMock(side_effect=RuntimeError('PRIVATE_CONFIGURATION')))))
    if cached:
        await runtime.load_caches()
        assert runtime._GLOBAL_CACHE=={'remoteChatModel':'safe-model'}
    else:
        with pytest.raises(runtime.RuntimeConfigurationUnavailable): await runtime.ensure_loaded()
        assert not runtime._CACHE_LOADED
    assert 'PRIVATE_CONFIGURATION' not in caplog.text


def test_production_sync_access_cannot_use_environment_before_preload(monkeypatch):
    monkeypatch.setattr(runtime,'_CACHE_LOADED',False)
    monkeypatch.setattr(settings,'app_env','production')
    token=runtime._current_snapshot.set(None)
    try:
        with pytest.raises(runtime.RuntimeConfigurationUnavailable):runtime.resolve_config_sync()
    finally:runtime._current_snapshot.reset(token)


async def test_initial_config_failure_prevents_api_scheduler_start(monkeypatch):
    from app import main
    from jobs import scheduler
    @asynccontextmanager
    async def lock(*args,**kwargs):yield
    for name in ['connect_db','disconnect_db','close_redis','ensure_prompt_templates',
                 'ensure_default_careers','ensure_default_names','reload_holiday_cache']:
        monkeypatch.setattr(main,name,AsyncMock())
    monkeypatch.setattr(main,'_warn_if_embedding_model_uncalibrated',lambda:None)
    monkeypatch.setattr(main,'get_redis',AsyncMock(return_value=SimpleNamespace(ping=AsyncMock())))
    monkeypatch.setattr(main,'distributed_lock',lock)
    monkeypatch.setattr(main,'mark_redis_healthy',lambda _:None)
    monkeypatch.setattr(type(settings),'validate_security_config',lambda self:None)
    monkeypatch.setattr(settings,'app_runtime_role','integrated')
    monkeypatch.setattr(scheduler,'drain_delayed_replies',AsyncMock())
    start=MagicMock();monkeypatch.setattr(main,'setup_scheduler',start)
    monkeypatch.setattr(runtime,'load_caches',AsyncMock(side_effect=runtime.RuntimeConfigurationUnavailable('synthetic')))
    with pytest.raises(runtime.RuntimeConfigurationUnavailable):
        await anext(main.lifespan(main.app))
    start.assert_not_called()
    main.disconnect_db.assert_awaited_once()


@pytest.mark.parametrize('stream',[False,True])
async def test_langchain_trace_identifies_the_actual_fallback_model(monkeypatch,stream):
    from langchain_core.callbacks import AsyncCallbackHandler
    class Audit(AsyncCallbackHandler):
        def __init__(self):self.models=[];self.completed=[]
        async def on_chat_model_start(self,serialized,messages,**kwargs):
            self.models.append(kwargs['invocation_params']['model'])
        async def on_llm_end(self,response,**kwargs):
            self.completed.append(response.generations[0][0].text)
    audit=Audit()
    async with clients(monkeypatch,'fail') as (primary,backup,calls,streams):
        result=await models.invoke_text(primary,'synthetic',profile=profile(stream),config={'callbacks':[audit]})
        assert audit.models==['primary','backup']
        assert audit.completed==[result]


async def test_unary_failure_keeps_no_success_metadata(monkeypatch):
    async with clients(monkeypatch,'fail','fail') as (primary,backup,calls,streams):
        metadata={'actual_model':'stale'}
        with pytest.raises(resilience.LLMFailedError):
            await models.invoke_text(primary,'synthetic',profile=profile(),invocation_metadata=metadata)
        assert metadata=={}
        assert [c['model'] for c in calls]==['primary','backup']
        assert 'dashscope' in resilience._breakers and 'ollama' not in resilience._breakers
