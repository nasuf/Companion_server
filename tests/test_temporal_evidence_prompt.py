"""Real registry rendering/Trace wiring; semantic quality uses live paired cases."""
from datetime import datetime, timezone
import json
from pathlib import Path
from string import Formatter
from unittest.mock import AsyncMock

import pytest

from app.services.chat.temporal_context import build_turn_clock, render_temporal_context
from app.services.prompting import store, trace_components
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.temporal_query import resolve_temporal_query

NOW = datetime(2026, 10, 21, 4, tzinfo=timezone.utc)


@pytest.fixture
def prompt_snapshot(monkeypatch):
    monkeypatch.setattr(store, 'is_prompt_enabled', AsyncMock(return_value=True))
    token = store._prompt_snapshot.set({k: (v.default_text, True)
                                       for k, v in PROMPT_DEFINITION_MAP.items()})
    try:
        yield
    finally:
        store.reset_prompt_snapshot(token)


def test_prompt_placeholders_keep_existing_web_render_contract():
    text = PROMPT_DEFINITION_MAP['chat.temporal_context'].default_text
    names = {name for _, name, _, _ in Formatter().parse(text) if name}
    assert names == {'now', 'statement_at', 'query_context', 'last_user',
                     'last_dialogue', 'last_outbound', 'user_gap_seconds',
                     'dialogue_gap_seconds', 'followup_policy'}


@pytest.mark.asyncio
@pytest.mark.parametrize('message,status', [
    ('昨天去哪里', 'resolved'), ('假如昨天去了就好了', 'ambiguous'),
    ('你好', 'not_requested'), ('x' * 4001, 'unresolved'),
])
async def test_runtime_renders_uncertainty_without_fabricating_ranges(prompt_snapshot, message, status):
    query = resolve_temporal_query(message, anchor=NOW, anchor_source='persisted_message')
    assert query.status == status
    text = await render_temporal_context(build_turn_clock([], now=NOW),
                                        'optional_questions_allowed=false', query)
    rendered_query = {k: v for k, v in query.prompt_context().items()
                      if k in {'status', 'timezone', 'ranges', 'reason'}}
    assert json.dumps(rendered_query, ensure_ascii=False) in text
    assert 'optional_questions_allowed=false' in text
    assert '{query_context}' not in text and '{now}' not in text
    if status != 'resolved':
        assert not query.parsed_times()


@pytest.mark.asyncio
async def test_count_execution_metadata_stays_in_trace_not_reply_prompt(prompt_snapshot):
    from app.services.chat.temporal_context import prepare_reply_time
    question = '我上周一共游泳几次？'
    query = resolve_temporal_query(question, anchor=NOW, anchor_source='persisted_message')
    before = query.diagnostics()
    diagnostics = {}
    _, _, text = await prepare_reply_time([], exclude_ids=set(), current_message=question,
                                          diagnostics=diagnostics, now=NOW, temporal_query=query)
    assert 'not_executed' not in text
    assert 'complete_scope' not in text
    assert json.dumps(before['ranges'], ensure_ascii=False) in text
    assert diagnostics['temporal_context']['query']['operation'] == before['operation']
    assert query.diagnostics() == before


@pytest.mark.asyncio
@pytest.mark.parametrize('gap,expected', [(30, '30 秒'), (600, '10 分钟'), (5400, '1 小时 30 分钟'), (7200, '2 小时'),
                                          (90000, '1 天 1 小时'), (172800, '2 天')])
async def test_dialogue_gap_is_rendered_as_known_duration_not_model_arithmetic(prompt_snapshot, gap, expected):
    from datetime import timedelta
    rows = [{'role': 'user', 'content': '先去忙', 'createdAt': NOW-timedelta(seconds=gap)}]
    text = await render_temporal_context(build_turn_clock(rows, now=NOW), '')
    assert f'距今约 {expected}' in text


@pytest.mark.asyncio
async def test_unknown_dialogue_gap_is_not_rendered_as_zero(prompt_snapshot):
    text = await render_temporal_context(build_turn_clock([], now=NOW), '')
    assert '距今约' not in text


@pytest.mark.asyncio
async def test_proactive_without_query_does_not_claim_resolved_date(prompt_snapshot):
    text = await render_temporal_context(build_turn_clock([], now=NOW),
                                        'optional_questions_allowed=true')
    assert '未请求日期解析' in text
    assert '"status": "resolved"' not in text
    assert '2026-10-21 12:00' in text


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['weak', 'medium', 'strong', 'l3'])
async def test_all_tiers_include_exact_temporal_text_in_editable_trace(prompt_snapshot, monkeypatch, kind):
    from app.services.chat import intent_replies
    invoke = AsyncMock(return_value='有依据的回答')
    monkeypatch.setattr(intent_replies, 'invoke_text', invoke)
    monkeypatch.setattr(intent_replies, 'get_chat_model', lambda: None)
    temporal = await render_temporal_context(build_turn_clock([], now=NOW),
                                            'optional_questions_allowed=true')
    token = trace_components.start_prompt_render_trace()
    try:
        result = await getattr(intent_replies, f'memory_{kind}_reply')(
            message='之前提过的那家叫什么', temporal_instructions=temporal)
        assert result == '有依据的回答'
        actual = invoke.await_args.args[1]
        assert actual.count(temporal) == 1
        trace = trace_components.snapshot_prompt_render_traces()[-1]
        component = next(c for c in trace['components'] if c['prompt_key']=='chat.temporal_context')
        body = next(c for c in trace['components'] if c['prompt_key']==f'memory.{kind}_reply')
        prefix = next(c for c in trace['components'] if c['prompt_key']=='chat.anti_hallucination_hard_rule')
        assert prefix['end'] < body['start'] < body['end'] < component['start']
        assert component['editable'] is True
        assert actual[component['start']:component['end']] == temporal
    finally:
        trace_components.reset_prompt_render_trace(token)


def test_generalization_corpus_keeps_negative_controls_and_unknown_event_times():
    cases = json.loads((Path(__file__).parents[1] / 'evals/temporal_evidence/cases.json').read_text())
    assert len(cases) == len({c['id'] for c in cases}) == len({c['message'] for c in cases}) == 24
    assert {k:sum(c['category']==k for c in cases) for k in ('positive','negative','control')} == {
        'positive':10, 'negative':10, 'control':4}
    assert any(not c['memories'] for c in cases)
    assert any(len(c['memories']) > 1 for c in cases)
    assert any(m['occur_time'] is None for c in cases for m in c['memories'])
    assert all(c['criterion'] for c in cases)
