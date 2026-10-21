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
    assert json.dumps(query.prompt_context(), ensure_ascii=False) in text
    assert 'optional_questions_allowed=false' in text
    assert '{query_context}' not in text and '{now}' not in text
    if status != 'resolved':
        assert not query.parsed_times()


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
        assert actual.endswith(temporal)
        trace = trace_components.snapshot_prompt_render_traces()[-1]
        component = next(c for c in trace['components'] if c['prompt_key']=='chat.temporal_context')
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
