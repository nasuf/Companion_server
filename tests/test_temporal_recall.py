from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.chat import temporal_recall as recall
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.store import PromptDisabledError


def memory(text='在白鹭诊所看牙', source='user', date='2026-10-20T04:00:00Z'):
    return SimpleNamespace(text=text, source=source, occur_time=date, statement_time='2026-10-20T12:00:00Z')


def diagnostic():
    return {'temporal_context': {'query': {'anchor_at': '2026-10-21T04:00:00Z',
        'anchor_source': 'persisted_message', 'timezone': 'Asia/Shanghai', 'operation': {'type': 'unspecified'}}}}


def answer():
    return {'memory_indices': [0], 'reply': '是白鹭诊所。'}


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(recall, 'build_reply_prefix', AsyncMock(return_value=''))
    monkeypatch.setattr(recall, 'get_grounding_model', lambda: 'grounding')
    monkeypatch.setattr(recall, 'get_prompt_text', AsyncMock(side_effect=lambda key: PROMPT_DEFINITION_MAP[key].default_text))
    invoke = AsyncMock(side_effect=[answer(), {'grounded': True}])
    monkeypatch.setattr(recall, 'invoke_json', invoke)
    return invoke


@pytest.mark.asyncio
async def test_answer_preserves_original_evidence_and_records_offsets(model):
    state = diagnostic()
    memories = [memory()]
    original = deepcopy(memories[0].__dict__)
    reply = await recall.answer_temporal_recall('昨天去了哪家诊所？', memories, state, max_chars=150)
    assert reply == '是白鹭诊所。'
    assert memories[0].__dict__ == original
    row = state['temporal_context']['recall']
    assert row['status'] == 'accepted'
    assert row['candidate_count'] == 1
    assert row['matches'] == [{'index': 0, 'source': 'user'}]
    assert 'content' not in row and 'reply' not in row
    assert model.await_count == 2
    assert model.await_args_list[0].kwargs['profile'].allow_ollama_fallback is False


@pytest.mark.asyncio
@pytest.mark.parametrize('question', ['好久不见', '上个月总共去过几次？', '上个月去了多久？', '现在还住那里吗？', '昨晚几点回来的？', '今天吃什么？'])
async def test_unqualified_question_never_calls_model(question, model):
    state = diagnostic()
    assert await recall.answer_temporal_recall(question, [memory()], state, max_chars=150) is None
    model.assert_not_awaited()
    assert state['temporal_context']['recall']['status'] == 'not_requested'


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['missing_anchor', 'empty', 'too_many', 'too_long'])
async def test_missing_or_over_budget_evidence_is_not_silently_cut(reason, model):
    state, records = diagnostic(), [memory()]
    if reason == 'missing_anchor': state['temporal_context']['query']['anchor_at'] = None
    if reason == 'empty': records = []
    if reason == 'too_many': records = [memory()] * 21
    if reason == 'too_long': records = [memory('x'*1001)]
    assert await recall.answer_temporal_recall('昨天去过哪里？', records, state, max_chars=150) is None
    model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [False, 0, {}, {'grounded': 'true'}, {'grounded': True, 'replacement': '编造的替代故事'}])
async def test_invalid_verdict_never_delivers_model_text(bad, model):
    model.side_effect = [answer(), bad]
    state = diagnostic()
    reply = await recall.answer_temporal_recall('昨天去了哪家？', [memory()], state, max_chars=150)
    assert reply == recall._FALLBACK
    assert state['temporal_context']['recall']['status'] == 'unavailable'


@pytest.mark.asyncio
async def test_rejected_answer_does_not_deliver_or_attempt_unverified_rewrite(model):
    model.side_effect = [answer(), {'grounded': False}]
    state = diagnostic()
    assert await recall.answer_temporal_recall('昨天去了哪家？', [memory()], state, max_chars=150) == recall._FALLBACK
    assert state['temporal_context']['recall']['status'] == 'rejected'
    assert model.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['chat.temporal_recall', 'chat.temporal_recall_check'])
async def test_disabling_either_prompt_disables_entire_route(key, model, monkeypatch):
    async def load(k):
        if k == key: raise PromptDisabledError(k)
        return PROMPT_DEFINITION_MAP[k].default_text
    monkeypatch.setattr(recall, 'get_prompt_text', load)
    state = diagnostic()
    assert await recall.answer_temporal_recall('昨天去了哪家？', [memory()], state, max_chars=150) is None
    model.assert_not_awaited()
    assert state['temporal_context']['recall']['status'] == 'disabled'


@pytest.mark.asyncio
async def test_cancellation_propagates(model):
    import asyncio
    model.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await recall.answer_temporal_recall('昨天去了哪家？', [memory()], diagnostic(), max_chars=150)


@pytest.mark.asyncio
async def test_service_failure_explicitly_degrades(model):
    model.side_effect = TimeoutError
    state = diagnostic()
    assert await recall.answer_temporal_recall('昨天去了哪家？', [memory()], state, max_chars=150) == recall._FALLBACK
    assert state['temporal_context']['recall']['reason'] == 'service_failed'


@pytest.mark.parametrize('index', [True, -1, 1, '0'])
def test_invalid_or_out_of_scope_reference(index):
    result = answer(); result['memory_indices'][0] = index
    with pytest.raises(ValueError):
        recall.validate_answer(result, records=[{'content':'白鹭诊所','source':'user'}], max_chars=150)


def test_known_detail_does_not_gain_event_date_from_statement_date():
    payload, reason = recall.recall_payload('那次修相机的店叫什么？', [memory('去云杉维修修相机', date=None)], diagnostic()['temporal_context']['query'])
    assert reason == 'eligible'
    assert payload['memories'][0]['event_date'] is None
    assert payload['memories'][0]['statement_date'] == '2026-10-20'


def test_calendar_uses_received_anchor_across_midnight_and_year_boundary():
    query = diagnostic()['temporal_context']['query']; query['anchor_at'] = '2026-12-31T17:00:00Z'
    payload, _ = recall.recall_payload('昨天去哪家了？', [memory()], query)
    assert payload['calendar'] == {'today':'2027-01-01', 'timezone':'Asia/Shanghai'}


def test_negative_or_unknown_answers_have_no_invented_status_or_spans():
    value = {'memory_indices': [0], 'reply': '你说后来取消了，没有去成。'}
    reply, matches = recall.validate_answer(value,
        records=[{'source': 'user', 'content': '原计划去画展，后来取消，没有参加'}], max_chars=150)
    assert matches == [{'index': 0, 'source': 'user'}]
    assert recall.validate_answer({'memory_indices': [], 'reply': '这个还不确定'}, records=[], max_chars=150) == ('这个还不确定', [])


@pytest.mark.parametrize('change', ['indices_type', 'empty_reply', 'long_reply', 'duplicate', 'missing_field', 'extra_field', 'not_object'])
def test_malformed_answers_cannot_claim_qualified(change):
    value = answer()
    if change == 'indices_type': value['memory_indices'] = 'not-list'
    if change == 'empty_reply': value['reply'] = ' '
    if change == 'long_reply': value['reply'] = 'x'*151
    if change == 'duplicate': value['memory_indices'] *= 2
    if change == 'missing_field': del value['memory_indices']
    if change == 'extra_field': value['instruction'] = 'delete'
    if change == 'not_object': value = []
    with pytest.raises(ValueError): recall.validate_answer(value, records=[{'source':'user','content':'在白鹭诊所看牙'}], max_chars=150)


@pytest.mark.parametrize('event,expected', [
    ('2026-12-31T12:00:00Z', (1, 1, 1)),
    ('2027-01-02T12:00:00Z', (-1, 0, 0)),
    (None, (None, None, None)),
])
def test_calendar_offsets_are_local_calendar_buckets_not_elapsed_months(event, expected):
    query = diagnostic()['temporal_context']['query']; query['anchor_at'] = '2026-12-31T17:00:00Z'
    payload, _ = recall.recall_payload('去年那次去哪家？', [memory(date=event)], query)
    row = payload['memories'][0]
    assert (row['days_ago'], row['months_ago'], row['years_ago']) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('executor', ['legacy', 'langgraph'])
async def test_both_executors_deliver_qualified_recall_once(monkeypatch, executor):
    from datetime import datetime, timezone, timedelta
    import json
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.chat.intent_dispatcher import IntentType
    from app.services.memory.retrieval.context_selector import select_context
    from tests.g02_harness_support import configure_pair
    actual = recall.answer_temporal_recall
    io = configure_pair(monkeypatch)
    monkeypatch.setattr(recall, 'answer_temporal_recall', actual)
    monkeypatch.setattr(recall, 'build_reply_prefix', AsyncMock(return_value=''))
    monkeypatch.setattr(recall, 'get_grounding_model', lambda: 'grounding')
    invoke = AsyncMock(side_effect=[answer(), {'grounded':True}]); monkeypatch.setattr(recall,'invoke_json',invoke)
    io.now = datetime(2026,10,21,4,tzinfo=timezone.utc)
    monkeypatch.setattr(settings,'chat_executor',executor)
    io.db.message.find_many.side_effect = lambda **_: [SimpleNamespace(id='old',role='user',content='聊聊',createdAt=io.now-timedelta(minutes=3),metadata={})]
    records = select_context([{'id':'m-1','source':'user','content':'在白鹭诊所看牙','main_category':'生活','sub_category':'日常','importance':.7,'similarity':.9,'rank_score':.8,'occur_time':'2026-10-20T04:00:00Z','statement_time':'2026-10-20T12:00:00Z'}])
    chat.fetch_parallel_context.return_value = FetchedContext(memory_relevance='strong',classified_memories=records)
    events = [e async for e in chat.stream_chat_response('c-1','昨天去了哪家诊所？',io.agent,'u-1',forced_intent=IntentType.NONE,reply_context={'received_at':io.now.isoformat()})]
    texts = [json.loads(e['data'])['text'] for e in events if e['event']=='reply']
    assert texts == ['是白鹭诊所。']
    assert sum(e['event']=='done' for e in events)==1
    assert invoke.await_count==2 and not io.main_calls and not io.tier_calls
    payload = json.loads(invoke.await_args_list[0].args[1][1].content)
    assert payload['memories'][0]['event_date']=='2026-10-20'
    assert invoke.await_args_list[0].args[1][0].type=='system'


@pytest.mark.asyncio
async def test_legacy_missing_diagnostics_stays_on_original_path(model):
    assert await recall.answer_temporal_recall('昨天去了哪里？', [memory()], {}, max_chars=150) is None
    model.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_enabled_template_disables_whole_route(model, monkeypatch):
    monkeypatch.setattr(recall,'get_prompt_text',AsyncMock(return_value=''))
    state=diagnostic()
    assert await recall.answer_temporal_recall('昨天去了哪家？',[memory()],state,max_chars=150) is None
    assert state['temporal_context']['recall']['status']=='disabled'
    model.assert_not_awaited()


def test_validity_and_future_operations_are_not_answered_by_point_recall():
    query=diagnostic()['temporal_context']['query']
    for operation in ('planned','current','as_of','order','duration','count'):
        query['operation']={'type':operation}
        assert recall.recall_payload('今年那个活动在哪里？',[memory()],query)==(None,'unsupported_operation')


@pytest.mark.asyncio
async def test_new_prompt_roles_have_exact_editable_trace_components(monkeypatch):
    from app.services.prompting import store, trace_components
    monkeypatch.setattr(store,'is_prompt_enabled',AsyncMock(return_value=True))
    token=store._prompt_snapshot.set({k:(v.default_text,True) for k,v in PROMPT_DEFINITION_MAP.items()})
    trace_token=trace_components.start_prompt_render_trace()
    monkeypatch.setattr(recall,'build_reply_prefix',AsyncMock(return_value=''))
    monkeypatch.setattr(recall,'get_grounding_model',lambda:None)
    call=AsyncMock(side_effect=[answer(),{'grounded':True}]);monkeypatch.setattr(recall,'invoke_json',call)
    try:
        assert await recall.answer_temporal_recall('昨天去了哪家？',[memory()],diagnostic(),max_chars=150)=='是白鹭诊所。'
        traces=trace_components.snapshot_prompt_render_traces()
        for i,key in enumerate(('chat.temporal_recall','chat.temporal_recall_check')):
            actual=call.await_args_list[i].args[1][0].content
            trace=next(t for t in traces if t.get('prompt_hash')==trace_components.prompt_hash(actual))
            assert trace['prompt_key']==key
            component=next(c for c in trace['components'] if c['prompt_key']==key)
            assert component['editable'] and actual[component['start']:component['end']]==actual
            assert call.await_args_list[i].args[1][1].type=='human'
    finally:
        store.reset_prompt_snapshot(token);trace_components.reset_prompt_render_trace(trace_token)


@pytest.mark.asyncio
async def test_existing_reply_style_recent_correction_and_character_reach_answer_and_check(model, monkeypatch):
    import json
    monkeypatch.setattr(recall, 'build_reply_prefix', AsyncMock(return_value='保持后台定制口吻。'))
    await recall.answer_temporal_recall('之前买的笔是什么牌子？', [memory()], diagnostic(), max_chars=150,
        recent_context='user: 更正一下，是雨竹牌。', personality='你叫小岚，现居普洱。')
    prompts = [c.args[1] for c in model.await_args_list]
    assert prompts[0][0].type == 'system' and prompts[0][1].type == 'human'
    material = json.loads(prompts[0][-1].content)
    assert material.pop('style_preferences') == '保持后台定制口吻。'
    assert material['recent_context'] == 'user: 更正一下，是雨竹牌。'
    assert material['character'] == '你叫小岚，现居普洱。'
    assert json.loads(prompts[1][-1].content)['input'] == material
    assert '保持后台定制口吻' not in prompts[1][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['recent_context', 'personality', 'style'])
async def test_oversized_shared_context_is_not_partially_hidden_from_recall(field, model, monkeypatch):
    kwargs = {field: 'x' * 4001} if field != 'style' else {}
    if field == 'style': monkeypatch.setattr(recall, 'build_reply_prefix', AsyncMock(return_value='x'*8001))
    state = diagnostic()
    assert await recall.answer_temporal_recall('昨天去了哪家？', [memory()], state, max_chars=150, **kwargs) is None
    assert state['temporal_context']['recall']['status'] == 'not_requested'
    assert state['temporal_context']['recall']['reason'] == 'input_limit'
    model.assert_not_awaited()


@pytest.mark.parametrize('value', ['unknown', ''])
def test_unknown_source_or_empty_content_does_not_enter_recall(value):
    row = memory(source=value) if value else memory(text='')
    assert recall.recall_payload('昨天去了哪里？', [row], diagnostic()['temporal_context']['query']) == (None, 'invalid_source')


def test_unicode_and_local_midnight_are_kept_in_original_record():
    query = diagnostic()['temporal_context']['query']; query['anchor_at'] = '2024-03-01T00:00:00Z'
    row = memory('😺看了《星空》', date='2024-02-28T17:00:00Z')
    payload, _ = recall.recall_payload('昨天看了什么？', [row], query)
    record = payload['memories'][0]
    assert record['content'] == '😺看了《星空》'
    assert record['event_date'] == '2024-02-29' and record['days_ago'] == 1
    assert record['months_ago'] == 1 and record['years_ago'] == 0


def test_duplicate_indices_are_rejected_even_when_candidate_limit_not_exceeded():
    with pytest.raises(ValueError, match='invalid_index'):
        recall.validate_answer({'memory_indices':[0,0], 'reply':'some reply'},
            records=[{'content':'one','source':'user'},{'content':'two','source':'ai'}], max_chars=80)


@pytest.mark.parametrize('reply,record', [
    ('我没去过那个公园。', '用户在松涛公园散步'),
    ('好像没真去过。', '用户希望有机会出行，尚无成行信息'),
    ('好像没成行。', '用户希望有机会出行，尚无成行信息'),
    ('你未曾参加过。', '用户考虑报名'),
    ('你之前没跟我提过店名。', '用户九月修了手表'),
    ('还没真的办手续呢。', '用户正在考虑换工作，尚未决定是否离职'),
    ('昨天没有下单。', '用户想买一架相机'),
    ('你还未参加那个活动。', '用户在考虑参加展览'),
])
def test_lack_of_an_event_record_cannot_support_a_negative_event_claim(reply, record):
    assert recall.has_unsupported_denial(reply, [{'content':record}], '')


@pytest.mark.parametrize('reply,record,context', [
    ('我还不确定你去没去。', '用户在考虑出行', ''),
    ('你之前说没有参加。', '用户明确说没有参加展览', ''),
    ('没去成，你后来取消了。', '用户后来取消，没有参加', ''),
    ('你说过没订。', '用户说没有下单', ''),
    ('你刚说还没下单。', '用户考虑买相机', '我还没下单。'),
    ('没问题，是墨川牌。', '买了墨川牌钢笔', ''),
])
def test_explicit_user_absence_or_uncertainty_is_not_an_unsupported_denial(reply, record, context):
    assert not recall.has_unsupported_denial(reply, [{'content':record}], context)


@pytest.mark.asyncio
async def test_unsubstantiated_denial_is_blocked_before_model_checker(model):
    model.side_effect = [{'memory_indices':[0], 'reply':'你没去过。'}]
    state = diagnostic()
    assert await recall.answer_temporal_recall('昨天去了哪里？', [memory()], state, max_chars=150,
        recent_context='assistant: 你没去过。', personality='现居杭州。') == recall._FALLBACK
    assert state['temporal_context']['recall']['reason'] == 'unsupported_absence'
    assert model.await_count == 1


def test_question_or_conditional_is_not_proof_of_absence_and_uncertainty_does_not_cover_next_clause():
    for text in ('不知道有没有去', '如果没有去就好了', '你没去过吗', '昨天没去？', '昨天没去?'):
        assert recall.has_unsupported_denial('你没去过。', [{'content':text}], '')
    assert recall.has_unsupported_denial('我不确定哪天，但你没去过。', [], '')


@pytest.mark.parametrize('query_change,expected', [
    ({}, [True, False, None]),
    ({'status': 'ambiguous'}, [None, None, None]),
    ({'date_basis': {'event_range_eligible': False}}, [None, None, None]),
    ({'ranges': [{'start':'invalid', 'end':'2026-10-31'}]}, [None, None, None]),
    ({'ranges': [{'start':'2026-11-01', 'end':'2026-10-31'}]}, [None, None, None]),
])
def test_bound_range_is_computed_in_code_and_never_uses_statement_date(query_change, expected):
    query=diagnostic()['temporal_context']['query']
    query.update(status='resolved',date_basis={'event_range_eligible':True},
        ranges=[{'start':'2026-10-01T00:00:00+08:00','end':'2026-10-31T23:59:59+08:00'}])
    query.update(query_change)
    rows=[memory(date='2026-10-31T15:59:59Z'),memory(date='2026-10-31T16:00:00Z'),memory(date=None)]
    payload,_=recall.recall_payload('上个月去哪家？',rows,query)
    assert [r['in_question_date_range'] for r in payload['memories']]==expected
    assert len(payload['memories'])==3  # No retrieval/deletion change.
    assert bool(payload['question_date_ranges']) == (expected[0] is True)


@pytest.mark.asyncio
async def test_checker_rejects_wrong_subject_even_with_same_negative_predicate(model):
    model.side_effect=[{'memory_indices':[0],'reply':'你没参加过。'},{'grounded':False}]
    state=diagnostic()
    assert await recall.answer_temporal_recall('去年我参加过吗？',
        [memory('用户的姐姐没有参加过那个活动')],state,max_chars=150)==recall._FALLBACK
    assert model.await_count==2  # Lexical support is never sufficient to accept.
    assert state['temporal_context']['recall']['reason']=='answer_not_grounded'


@pytest.mark.parametrize('key', ['chat.temporal_recall', 'chat.temporal_recall_check'])
def test_json_prompt_examples_have_no_unfilled_format_fields(key, caplog):
    from string import Formatter
    from app.services.prompting.utils import render_template
    text=PROMPT_DEFINITION_MAP[key].default_text
    assert not [field for _,field,_,_ in Formatter().parse(text) if field is not None]
    rendered=render_template(text, {}, safe=False)
    assert '{"' in rendered and '{{' not in rendered
    assert 'render failed' not in caplog.text


@pytest.mark.parametrize('reply,event,statement,rejected', [
    ('是2018年，那天你跟我说的。','2018-08-02','2026-10-20',True),
    ('那天你告诉我的。',None,'2026-10-20',True),
    ('当天提过这件事。','2026-10-20',None,True),
    ('那天你跟我说的。','2026-10-20','2026-10-20',False),
    ('你之前提过。','2018-08-02','2026-10-20',False),
    ('是2018年8月2日收养的。','2018-08-02','2026-10-20',False),
])
def test_event_day_is_not_mention_day(reply,event,statement,rejected):
    assert recall.has_unsupported_same_day_mention(reply,[{'event_date':event,'statement_date':statement}]) is rejected


def test_deictic_mention_without_any_record_is_unknown():
    assert recall.has_unsupported_same_day_mention('那天你跟我说的。',[])


@pytest.mark.asyncio
async def test_confused_mention_day_is_rejected_before_model_check(model):
    model.side_effect=[{'memory_indices':[0],'reply':'是2018年8月2日，那天你跟我说的。'}]
    state=diagnostic()
    reply=await recall.answer_temporal_recall('之前收养猫是什么时候？',
        [memory('2018年8月2日收养猫',date='2018-08-02T04:00:00Z')],state,max_chars=150)
    assert reply==recall._FALLBACK
    assert state['temporal_context']['recall']['reason']=='unsupported_mention_time'
    assert model.await_count==1
