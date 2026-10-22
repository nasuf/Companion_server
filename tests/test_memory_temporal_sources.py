"""Per-fact event time must never inherit a batch clock or unrelated date."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.memory.evidence import EvidenceSource, message_statement_time
from app.services.memory.recording.temporal_sources import (
    resolve_fact_clock, source_table, public_temporal_basis, uncertain_composite_sources,
)

TZ = timezone(timedelta(hours=8))
T = datetime(2026, 10, 20, 9, 0, tzinfo=TZ)


def origin(text, time=T, ref='m1'):
    return EvidenceSource('message', ref, expected_statement_time=time, input_text=text)


def mem(indices, quote=None, index=None):
    return dict(source_indices=indices, time_quote=quote, time_source_index=index,
                occur_time='2099-01-01T12:00:00Z')


def test_cross_day_backlog_anchors_each_fact_and_does_not_broadcast():
    sources=(origin('昨天去看展了'), origin('最近喜欢摄影', T+timedelta(days=1),'m2'))
    event=resolve_fact_clock(mem([1], '昨天', 1), sources)
    preference=resolve_fact_clock(mem([2]), sources)
    assert event.occur_time.date().isoformat()=='2026-10-19'
    assert event.statement_time==T
    assert preference.occur_time is None and preference.statement_time==T+timedelta(days=1)
    assert preference.basis['status']=='time_not_stated'
    assert [s.temporal_basis['source_role'] for s in event.bind(sources)]==['fact','batch']
    assert 'm1' not in source_table(sources)
    assert '昨天去看展了' in source_table(sources)


def test_two_events_in_one_message_keep_their_own_dates():
    sources=(origin('昨天去看展了，明天下午3点去面试'),)
    first=resolve_fact_clock(mem([1],'昨天',1),sources)
    second=resolve_fact_clock(mem([1],'明天下午3点',1),sources)
    assert first.occur_time.date().isoformat()=='2026-10-19'
    assert second.occur_time == datetime(2026,10,21,15,tzinfo=TZ)


def test_minutes_use_original_message_clock_and_offset():
    sources=(origin('十分钟前到公司了'),origin('喜欢摄影',T+timedelta(hours=5),'m2'))
    result=resolve_fact_clock(mem([1],'十分钟前',1),sources)
    assert result.occur_time==T-timedelta(minutes=10)


@pytest.mark.parametrize('indices',[None,[],[0],[3],[-1],[True],['1'],[1,1],{},'1'])
def test_invalid_attribution_never_falls_back_to_llm_or_batch(indices):
    result=resolve_fact_clock(mem(indices,'昨天',1),(origin('昨天到家'),origin('今天出门')))
    assert result.occur_time is None and result.statement_time is None
    assert result.basis['status']=='source_unresolved'


@pytest.mark.parametrize('quote,index',[('明天',1),('昨天',2),('昨天',True),('昨天','1'),(' ',1),('x'*201,1),(7,1)])
def test_invalid_quote_is_unknown(quote,index):
    result=resolve_fact_clock(mem([1],quote,index),(origin('昨天到家'),))
    assert result.occur_time is None and result.basis['status']=='time_quote_invalid'


@pytest.mark.parametrize('quote',['以前','昨天和明天'])
def test_unresolved_or_multiple_times_remain_ambiguous(quote):
    result=resolve_fact_clock(mem([1],quote,1),(origin(quote+'的事'),))
    assert result.occur_time is None and result.basis['status']=='time_ambiguous'


def test_multiple_statement_times_preserve_range_not_a_false_point():
    sources=(origin('喜欢摄影'),origin('也喜欢看展',T+timedelta(days=1),'m2'))
    result=resolve_fact_clock(mem([2,1]),sources)
    assert result.statement_time is None and result.occur_time is None
    assert result.basis['statement_start']==T.isoformat()
    assert result.basis['statement_end']==(T+timedelta(days=1)).isoformat()
    assert resolve_fact_clock(mem([1]),(replace(sources[0],expected_statement_time=None),)).statement_time is None


@pytest.mark.parametrize('value',[None,3,'bad date'])
def test_invalid_persisted_timestamp_is_rejected(value):
    with pytest.raises(ValueError,match='evidence_source_time_invalid'):message_statement_time(value)


def test_native_sql_and_orm_timestamps_agree():
    assert message_statement_time('2026-10-20T01:00:00Z')==T
    assert message_statement_time(datetime(2026,10,20,1))==T
    assert message_statement_time(T)==T


@pytest.mark.parametrize('value',[{},[],{'version':True},{'version':1,'status':[]},
    {'version':1,'status':'resolved','source_role':[]},
    {'version':1,'status':'resolved','source_role':'fact'},
    {'version':1,'status':'time_not_stated','source_role':'fact','statement_start':'bad'},
    {'version':1,'status':'time_not_stated','source_role':'fact','statement_start':'2026-10-20','statement_end':'2026-10-21'},
    {'version':1,'status':'time_not_stated','source_role':'fact','statement_start':T.isoformat(),'statement_end':(T-timedelta(days=1)).isoformat()}])
def test_public_metadata_never_trusts_malformed_storage(value):
    assert public_temporal_basis(value)=={'status':'invalid_metadata'}


def test_public_metadata_whitelist_and_composite_uncertainty():
    assert public_temporal_basis(None) is None
    sources=(origin('昨天到家'),)
    clock=resolve_fact_clock(mem([1],'昨天',1),sources)
    basis={**clock.bind(sources)[0].temporal_basis, 'private_text':'secret'}
    assert 'private_text' not in public_temporal_basis(basis)
    split=uncertain_composite_sources(clock.bind(sources),'split_unresolved')
    assert split[0].expected_statement_time==T
    assert split[0].temporal_basis=={'version':1,'status':'split_unresolved','source_role':'batch'}


@pytest.mark.asyncio
async def test_pipeline_keeps_facts_but_does_not_accept_model_invented_time(monkeypatch):
    from app.services.memory.recording import pipeline as p
    import app.services.memory.evidence as evidence
    sources=(origin('昨天去看展了'),origin('喜欢摄影',T+timedelta(days=1),'m2'))
    monkeypatch.setattr(p,'resolve_workspace_id',AsyncMock(return_value='w'))
    monkeypatch.setattr(evidence,'snapshot_message_sources',AsyncMock(return_value=sources))
    monkeypatch.setattr(p,'should_extract_memory',lambda _:True)
    monkeypatch.setattr(p,'should_memorize',AsyncMock(return_value=True))
    records=[{**mem([1],'昨天',1),'content':'用户去看展','importance':.65,'main_category':'生活','sub_category':'其他'},
             {**mem([2]),'content':'用户喜欢摄影','importance':.65,'main_category':'偏好','sub_category':'爱好'},
             {**mem([]),'content':'用户看展后感觉开心','importance':.6,'main_category':'情绪','sub_category':'其他'}]
    extractor=AsyncMock(return_value={'memories':records})
    monkeypatch.setattr(p,'extract_memories',extractor)
    store=AsyncMock(side_effect=['fact1','fact2','fact3']);monkeypatch.setattr(p,'store_memory',store)
    audit=AsyncMock();monkeypatch.setattr(p,'log_memory_evidence',audit)
    monkeypatch.setattr(p,'record_entities_for_memory',AsyncMock())
    monkeypatch.setattr(p,'record_topics_for_memory',AsyncMock())
    await p.process_memory_pipeline('u','user: 昨天去看展了\nuser: 喜欢摄影',workspace_id='w',
        statement_time=T+timedelta(days=1),evidence_message_ids=['m1','m2'],evidence_conversation_id='c')
    calls=[c.kwargs for c in store.await_args_list]
    assert len(calls)==3 and all(c['infer_time'] is False for c in calls)
    assert calls[0]['occur_time'].date().isoformat()=='2026-10-19'
    assert calls[1]['occur_time'] is None and calls[2]['occur_time'] is None
    assert calls[2]['statement_time'] is None
    assert all(len(c['evidence_sources'])==2 for c in calls) # entire model input remains guarded
    assert 'source_messages' in extractor.await_args.kwargs
    assert [c.kwargs['message_ids'] for c in audit.await_args_list]==[['m1'],['m2'],[]]
    assert all(c.kwargs['sources_already_bound'] for c in audit.await_args_list)


@pytest.mark.parametrize('quote,expected',[
    ('明天下午三点',datetime(2026,10,21,15,tzinfo=TZ)),
    ('明天下午三点半',datetime(2026,10,21,15,30,tzinfo=TZ)),
    ('昨天十点二十分',datetime(2026,10,19,10,20,tzinfo=TZ)),
])
def test_chinese_clock_keeps_precision(quote,expected):
    clock=resolve_fact_clock(mem([1],quote,1),(origin(quote+'去面试'),))
    assert clock.occur_time==expected


def test_midnight_utc_source_resolves_in_configured_calendar():
    stamp=datetime(2026,10,20,16,15,tzinfo=timezone.utc) # local Oct 21
    clock=resolve_fact_clock(mem([1],'昨天',1),(origin('昨天去看展了',stamp),))
    assert clock.occur_time.date().isoformat()=='2026-10-20'


@pytest.mark.asyncio
async def test_source_contract_rendering_and_disabled_template(monkeypatch):
    from app.services.memory.recording import extraction as ex
    from app.services.prompting.defaults import MEMORY_EXTRACTION_SOURCE_CONTRACT_PROMPT
    from app.services.prompting.store import PromptDisabledError
    async def template(key):
        if key=='memory.extraction_source_contract':return MEMORY_EXTRACTION_SOURCE_CONTRACT_PROMPT
        return '当前时间 {current_time} 消息 {new_conversation} 上下文 {context_conversation} {taxonomy_list}'
    monkeypatch.setattr(ex,'get_prompt_text',template)
    monkeypatch.setattr(ex,'get_chat_model',lambda:None)
    invoke=AsyncMock(return_value={'memories':[]});monkeypatch.setattr(ex,'invoke_json',invoke)
    await ex.extract_memories('user: 昨天去看展了',source_messages=source_table((origin('昨天去看展了'),)))
    prompt=invoke.await_args.args[1]
    assert 'source_indices' in prompt and '2026-10-20T09:00:00+08:00' in prompt
    async def disabled(key):
        if key=='memory.extraction_source_contract':raise PromptDisabledError(key)
        return await template(key)
    monkeypatch.setattr(ex,'get_prompt_text',disabled)
    result=await ex.extract_memories('user: x',source_messages='[]')
    assert result['_extraction_error_kind']=='prompt_disabled'
    assert invoke.await_count==1


@pytest.mark.parametrize('quote,date',[('10月23日15点','2026-10-23'),('1月23日15点','2027-01-23')])
def test_yearless_reminder_calendar_assumption_is_visible(quote,date):
    result=resolve_fact_clock({**mem([1],quote,1),'sub_category':'提醒'},(origin(quote+'提醒我复查'),))
    assert result.basis['status']=='calendar_inferred'
    assert result.occur_time.date().isoformat()==date
    assert public_temporal_basis(result.bind((origin(quote),))[0].temporal_basis)['status']=='calendar_inferred'
    ordinary=resolve_fact_clock(mem([1],quote,1),(origin(quote+'去过医院'),))
    assert ordinary.occur_time is None and ordinary.basis['status']=='time_ambiguous'


def test_yearless_invalid_calendar_is_never_scheduled():
    q='2月30日15点'
    result=resolve_fact_clock({**mem([1],q,1),'sub_category':'提醒'},(origin(q+'提醒我复查'),))
    assert result.occur_time is None


@pytest.mark.asyncio
async def test_source_contract_is_editable_in_the_actual_extraction_trace(monkeypatch):
    from app.services.memory.recording import extraction as ex
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    from app.services.prompting.trace_components import (
        ManagedPromptText, start_prompt_render_trace, reset_prompt_render_trace,
        snapshot_prompt_render_traces,
    )
    from app.services.chat import trace_enrich
    from tests.test_trace_enrich import _fake_llm_step

    async def template(key):
        return ManagedPromptText(PROMPT_DEFINITION_MAP[key].default_text,key)
    monkeypatch.setattr(ex,'get_prompt_text',template)
    monkeypatch.setattr(ex,'get_chat_model',lambda:None)
    invoke=AsyncMock(return_value={'memories':[]})
    monkeypatch.setattr(ex,'invoke_json',invoke)
    token=start_prompt_render_trace()
    try:
        await ex.extract_memories('user: 昨天去看展了',source_messages=source_table((origin('昨天去看展了'),)))
        step=trace_enrich.enrich_step(_fake_llm_step(invoke.await_args.args[1]))
        trace_enrich.apply_prompt_render_traces([step],snapshot_prompt_render_traces())
    finally:
        reset_prompt_render_trace(token)
    components={item['prompt_key']:item for item in step['prompt_components']}
    assert step['prompt_key']=='memory.extraction_user'
    assert components['memory.extraction_source_contract']['editable'] is True
