from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.services.prompting import store
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.reply_prefix import PROACTIVE_COMMON_KEY, PROACTIVE_REPLY_PROMPT_KEYS
from app.services.prompting.trace_components import (
    ManagedPromptText, start_prompt_render_trace, reset_prompt_render_trace,
    snapshot_prompt_render_traces,
)
from app.services.prompting.utils import render_template, render_prompt
from app.services.proactive.dialogue import format_recent_turns


@pytest.mark.asyncio
async def test_shared_prefix_allowlist_and_disabled_semantics(monkeypatch):
    assert PROACTIVE_REPLY_PROMPT_KEYS <= PROMPT_DEFINITION_MAP.keys()
    assert PROACTIVE_COMMON_KEY not in PROACTIVE_REPLY_PROMPT_KEYS
    assert not PROACTIVE_REPLY_PROMPT_KEYS.intersection({
        'offline.activity_card', 'offline.activity_recommendation_copy',
        'offline.activity_companion_topic', 'offline.activity_companion_decision',
        'proactive.reminder_pre_check', 'proactive.memory_topic_rerank',
    })
    cache = AsyncMock()
    cache.get.side_effect = lambda key: '共同规则' if key.endswith(PROACTIVE_COMMON_KEY) else '场景 {name}'
    from contextvars import ContextVar
    monkeypatch.setattr(store, '_prompt_snapshot', ContextVar('test_proactive_prompts', default={key: ('共同规则' if key == PROACTIVE_COMMON_KEY else '场景 {name}', True) for key in PROACTIVE_REPLY_PROMPT_KEYS | {PROACTIVE_COMMON_KEY}}))
    enabled = AsyncMock(return_value=True)
    monkeypatch.setattr(store, 'is_prompt_enabled', enabled)
    for key in PROACTIVE_REPLY_PROMPT_KEYS:
        text = await store.get_prompt_text(key)
        assert text.format(name='内容') == '共同规则\n\n场景 内容'
        assert text.parts[0][0] == PROACTIVE_COMMON_KEY
    enabled.side_effect = lambda key: key != PROACTIVE_COMMON_KEY
    assert str(await store.get_prompt_text('offline.arrival_guide')) == '场景 {name}'
    enabled.return_value = False
    enabled.side_effect = None
    with pytest.raises(store.PromptDisabledError):
        await store.get_prompt_text('offline.arrival_guide')


@pytest.mark.parametrize('renderer', ['format', 'format_map', 'helper'])
def test_composed_trace_preserves_editable_source_boundaries(renderer):
    template = ManagedPromptText.compose([
        (PROACTIVE_COMMON_KEY, '规则'), ('offline.arrival_guide', '喜好：{optional}\n{{"text":"{name}"}}'),
    ], 'offline.arrival_guide')
    token = start_prompt_render_trace()
    try:
        params = {'name': '湖边', 'optional': '（无）'}
        if renderer == 'helper':
            result = render_template(template, params, optional_keys={'optional'})
            assert '喜好' not in result
        else:
            result = template.format(**params) if renderer == 'format' else template.format_map(params)
        traces = snapshot_prompt_render_traces()
        assert len(traces) == 1
        components = traces[0]['components']
        assert [c['prompt_key'] for c in components] == [PROACTIVE_COMMON_KEY, 'offline.arrival_guide']
        assert result[components[0]['start']:components[0]['end']] == '规则'
        assert result[components[1]['start']:components[1]['end']].endswith('{"text":"湖边"}')
        from app.services.chat.trace_enrich import apply_prompt_render_traces
        steps = [{'run_type':'llm','inputs':{'messages':[[{
            'kwargs':{'content':result,'type':'human'},
        }]]}}]
        enriched = apply_prompt_render_traces(steps, traces)[0]
        assert [c['prompt_key'] for c in enriched['prompt_components']] == [PROACTIVE_COMMON_KEY, 'offline.arrival_guide']
    finally:
        reset_prompt_render_trace(token)


def test_ten_turns_keep_multiple_bubbles_timestamps_and_roles():
    messages = []
    for index in range(12):
        for role, content in [('user', f'第{index}轮'), ('assistant', '前半句'), ('assistant', '后半句')]:
            messages.append(dict(role=role, content=content, created_at=datetime(2026, 10, 5, 0, index, tzinfo=UTC)))
    rendered = format_recent_turns(messages)
    assert '第1轮' not in rendered and '第2轮' in rendered and '第11轮' in rendered
    assert rendered.count('用户：') == 10
    assert rendered.count('AI：') == 20
    assert rendered.startswith('[2026-10-05 08:02] 用户：')
    assert format_recent_turns([dict(role='tool', content='不应该进入')]) == ''


@pytest.mark.asyncio
async def test_proactive_render_preserves_every_clause(monkeypatch):
    monkeypatch.setattr('app.services.prompting.utils.get_prompt_text', AsyncMock(return_value='规则'))
    result = await render_prompt('proactive.special_reminder', {}, AsyncMock(return_value='记得带伞||下午要出门'))
    assert result == '记得带伞 下午要出门'


def test_release_manifest_validates_all_actual_placeholder_contracts():
    import json
    from scripts.publish_proactive_common_prompts import MANIFEST, validate_entry
    for entry in json.loads(MANIFEST.read_text())['prompts']:
        validate_entry(entry)
    with pytest.raises(ValueError, match='unsupported placeholders'):
        validate_entry({'key':'offline.arrival_guide', 'content':'{missing_runtime_value}'})
