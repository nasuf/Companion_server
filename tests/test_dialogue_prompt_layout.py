from contextvars import ContextVar
from unittest.mock import AsyncMock

import pytest

from app.services.prompting import store
from app.services.prompting.defaults import DIALOGUE_HISTORY_HEADER, DIALOGUE_CURRENT_MESSAGE_HEADER, DIALOGUE_CONTINUATION_HINT
from app.services.prompting.dialogue_layout import (
    REPLY_KEYS, HISTORY_KEY, CURRENT_KEY, CONTINUATION_KEY, CONTINUATION_HINT_KEYS,
    DialoguePromptText, supports_layout,
)
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.trace_components import ManagedPromptText, start_prompt_render_trace, reset_prompt_render_trace, snapshot_prompt_render_traces
from app.services.prompting.utils import render_prompt, render_template


@pytest.fixture
def prompt_store(monkeypatch):
    snapshot = {k: (d.default_text, d.default_enabled) for k, d in PROMPT_DEFINITION_MAP.items()}
    monkeypatch.setattr(store, '_prompt_snapshot', ContextVar('layout_test', default=snapshot))
    monkeypatch.setattr(store, 'is_prompt_enabled', AsyncMock(side_effect=lambda k: snapshot[k][1]))
    return snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize('key', sorted(REPLY_KEYS))
async def test_all_eight_templates_keep_body_and_move_current_after_history(key, prompt_store):
    original = prompt_store[key][0]
    template = await store.get_prompt_text(key)
    assert isinstance(template, DialoguePromptText)
    assert prompt_store[key][0] == original
    result = render_template(template, {'message': '本轮问题独一份', 'context': '用户: 老话\nAI: 老答',
                                        'received_at': '2026-10-26T01:02:00Z'},
                             context_parts=(('chat.temporal_context', '时间规则 {literal}'),))
    assert result.count('本轮问题独一份') == 1
    assert result.index('时间规则 {literal}') < result.index(DIALOGUE_HISTORY_HEADER)
    assert result.index('AI: 老答') < result.index(DIALOGUE_CURRENT_MESSAGE_HEADER)
    current = '[10-26 09:02] 用户: 本轮问题独一份'
    if key in CONTINUATION_HINT_KEYS:  # selected variant B: the hint follows the current message
        assert result.endswith(current + '\n\n' + DIALOGUE_CONTINUATION_HINT)
    else:
        assert result.endswith(current) and DIALOGUE_CONTINUATION_HINT not in result


@pytest.mark.asyncio
async def test_disabled_headers_restore_slots_and_disabled_hint_selects_a(prompt_store):
    for disabled in (HISTORY_KEY, CURRENT_KEY):
        text, _ = prompt_store[disabled]
        prompt_store[disabled] = text, False
        template = await store.get_prompt_text('memory.weak_reply')
        assert not isinstance(template, DialoguePromptText)
        result = render_template(template, {'message': '最新', 'context': '旧对话'})
        assert '最新' in result and '旧对话' in result and DIALOGUE_CURRENT_MESSAGE_HEADER not in result
        prompt_store[disabled] = text, True
    prompt_store[CONTINUATION_KEY] = DIALOGUE_CONTINUATION_HINT, True
    result = (await store.get_prompt_text('memory.weak_reply')).format(message='最新', context='旧对话')
    assert result.endswith(DIALOGUE_CONTINUATION_HINT)


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['intent.unified', 'memory.relevance', 'intent.deletion_confirm', 'chat.system_base'])
async def test_exclusions_use_original_rendering(key, prompt_store):
    assert not isinstance(await store.get_prompt_text(key), DialoguePromptText)


@pytest.mark.parametrize('renderer', ['format', 'format_map', 'helper'])
def test_empty_history_braces_and_literal_current_message(renderer):
    original = ManagedPromptText('- 用户刚才说：{message}\n- 最近对话：{context}\n输出 {{"reply":"..."}}', 'intent.schedule_adjust_reply')
    template = DialoguePromptText(original, history=DIALOGUE_HISTORY_HEADER, current=DIALOGUE_CURRENT_MESSAGE_HEADER)
    params = {'message': '原样 {不要格式化}\n第二条', 'context': '（无）'}
    if renderer == 'helper':
        result = render_template(template, params, optional_keys={'context'})
    else:
        result = template.format(**params) if renderer == 'format' else template.format_map(params)
    assert '最近对话：（无）' in result and DIALOGUE_HISTORY_HEADER not in result
    assert '{"reply":"..."}' in result
    assert result.endswith('用户: 原样 {不要格式化}\n第二条')


@pytest.mark.asyncio
async def test_trace_headers_are_editable_but_history_and_message_are_not(prompt_store):
    token = start_prompt_render_trace()
    try:
        template = await store.get_prompt_text('memory.weak_reply')
        result = render_template(template, {'message': '当前', 'context': '历史内容'})
        trace = snapshot_prompt_render_traces()[-1]
        headers = [c for c in trace['components'] if c.get('prompt_key') in {HISTORY_KEY, CURRENT_KEY}]
        assert len(headers) == 2 and all(c['editable'] for c in headers)
        assert [result[c['start']:c['end']] for c in headers] == [DIALOGUE_HISTORY_HEADER, DIALOGUE_CURRENT_MESSAGE_HEADER]
        literal = [result[c['start']:c['end']] for c in trace['components'] if not c['editable']]
        assert literal == ['历史内容', '用户: 当前']
        from app.services.chat.trace_enrich import apply_prompt_render_traces
        enriched = apply_prompt_render_traces([{'run_type': 'llm', 'inputs': {'messages': [[{
            'kwargs': {'content': result, 'type': 'human'},
        }]]}}], [trace])[0]
        visible = {c['prompt_key']: c for c in enriched['prompt_components']}
        for key in (HISTORY_KEY, CURRENT_KEY):
            component = visible[key]
            assert component['editable']
            assert result[component['start']:component['end']] == prompt_store[key][0]
    finally:
        reset_prompt_render_trace(token)


@pytest.mark.asyncio
async def test_adjustment_json_contract_survives_unified_render(prompt_store, monkeypatch):
    from app.services.chat import intent_replies
    invoke = AsyncMock(return_value={'reply': '那就九点吧', 'adjustment': '21:00'})
    monkeypatch.setattr(intent_replies, 'invoke_json', invoke)
    monkeypatch.setattr(intent_replies, 'get_chat_model', lambda: None)
    # The caller still receives its structured payload, not a flattened reply.
    result = await intent_replies.schedule_adjust_reply(message='改成九点', context='用户: 八点回家',
                                                        personality_brief='小岚')
    assert result == {'reply': '那就九点吧', 'adjustment': '21:00'}
    rendered = invoke.call_args.args[1]
    assert 'JSON' in rendered and rendered.endswith('用户: 改成九点')


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['intent.schedule_adjust_reply', 'intent.end_reply', 'intent.schedule_query_reply'])
async def test_enabled_hint_skips_json_goodbye_and_unmeasured_templates(prompt_store, key):
    prompt_store[CONTINUATION_KEY] = DIALOGUE_CONTINUATION_HINT, True
    rendered = (await store.get_prompt_text(key)).format(
        message='改成九点', context='用户: 八点回家', personality_brief='小岚')
    assert DIALOGUE_CONTINUATION_HINT not in rendered and rendered.endswith('用户: 改成九点')
    reply = (await store.get_prompt_text('intent.current_state_reply')).format(
        message='在干嘛', context='用户: 晚上好')
    assert reply.endswith(DIALOGUE_CONTINUATION_HINT)


def test_malformed_or_missing_placeholders_do_not_guess():
    assert not supports_layout('文字 {context}')
    assert not supports_layout('{context} {message} {')
    assert supports_layout('{context}\n{message}')


@pytest.mark.asyncio
async def test_render_prompt_places_temporal_components_before_both_dialogue_blocks(prompt_store):
    invoke = AsyncMock(return_value='回复')
    result = await render_prompt('memory.weak_reply', {'message': '新问题', 'context': '旧历史'}, invoke,
                                 context_parts=(('chat.temporal_evidence', '证据 {"literal": 1}'),))
    assert result == '回复'
    text = invoke.call_args.args[0]
    assert text.index('证据 {"literal": 1}') < text.index(DIALOGUE_HISTORY_HEADER)
    assert text.endswith('用户: 新问题\n\n' + DIALOGUE_CONTINUATION_HINT)


@pytest.mark.asyncio
async def test_admin_single_step_replays_the_exact_unified_layout_once(prompt_store, monkeypatch):
    from app.api.admin import prompts
    from app.models.prompt_template import PromptTemplateReplayRequest
    invoke = AsyncMock(return_value='单步结果')
    monkeypatch.setattr(prompts, 'invoke_text', invoke)
    monkeypatch.setattr(prompts, 'get_chat_model', lambda: None)
    monkeypatch.setattr('app.services.runtime_config.ensure_loaded', AsyncMock())
    text = render_template(await store.get_prompt_text('memory.weak_reply'),
                           {'message': '本轮唯一', 'context': '上轮信息'})
    result = await prompts.replay_prompt_step(PromptTemplateReplayRequest(
        prompt_key='memory.weak_reply', rendered_prompt=text, model_kind='chat'))
    assert result.output == '单步结果'
    assert invoke.call_args.args[1] == text
    assert text.count('用户: 本轮唯一') == 1


@pytest.mark.asyncio
async def test_new_keys_follow_their_definition_default_without_a_live_row(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    monkeypatch.setattr(store, 'db', SimpleNamespace(prompttemplate=SimpleNamespace(
        find_unique=AsyncMock(return_value=None))))
    token = store.bind_prompt_snapshot(None)
    try:
        assert await store.is_prompt_enabled(CONTINUATION_KEY)  # variant B ships enabled
        assert store._new_data(PROMPT_DEFINITION_MAP[CONTINUATION_KEY])['isEnabled'] is True
        disabled = replace(PROMPT_DEFINITION_MAP[CONTINUATION_KEY], default_enabled=False)
        monkeypatch.setitem(PROMPT_DEFINITION_MAP, CONTINUATION_KEY, disabled)
        assert not await store.is_prompt_enabled(CONTINUATION_KEY)
        assert store._new_data(disabled)['isEnabled'] is False
    finally:
        store.reset_prompt_snapshot(token)
