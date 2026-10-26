from datetime import UTC, datetime, timedelta

import pytest
from app.services.chat import context_policy as policy
from app.services.chat.dialogue_context import render_dialogue
from app.services.chat.message_utils import previous_assistant_turn
from app.services import runtime_config


def rows(turns=12, bubbles=4):
    now = datetime(2026,10,26,0,tzinfo=UTC)
    result=[]
    for n in range(turns):
        result.append(dict(id=f'u{n}',role='user',content=f'用户选择{n}',createdAt=now+timedelta(minutes=n)))
        for b in range(bubbles):
            result.append(dict(id=f'a{n}-{b}',role='assistant',content=f'答复{n}-{b}',createdAt=now+timedelta(minutes=n,seconds=b+1),metadata={}))
    return result


@pytest.mark.parametrize('bad',[0,21,-1,True,5.0,'5',float('nan')])
def test_policy_invalid_values_cannot_be_saved(bad):
    with pytest.raises(ValueError): policy.apply_patch({}, {'rounds':bad})


def test_policy_reset_and_readonly_main_mode():
    assert policy.apply_patch({'rounds':12},{'rounds':None})=={}
    assert policy.resolve({}).rounds==5
    with pytest.raises(ValueError):policy.apply_patch({}, {'main_history_mode':'rounds'})
    assert policy.resolve({'rounds':0}).valid is False


@pytest.mark.parametrize('purpose', ['intent', 'relevance'])
def test_judgement_budget_does_not_grow_with_twenty_round_setting(purpose):
    from app.services.chat.data_fetch_phase import format_recent_context
    history = rows(20, 1)
    for row in history:
        row['content'] = row['content'] + '长文本' * 350
    text = format_recent_context(history, turns=20, max_chars=6000, purpose=purpose)
    assert len(text) <= policy.JUDGEMENT_MAX_CHARS
    assert all(len(line) <= policy.JUDGEMENT_LINE_CHARS + 30 for line in text.splitlines())


def test_same_turn_policy_survives_worker_reload(monkeypatch):
    snapshot=runtime_config.ConfigurationSnapshot({'chatContextPolicy':{'rounds':3}}, {}, {}, {})
    token=runtime_config._current_snapshot.set(snapshot)
    try:
        monkeypatch.setattr(runtime_config,'_GLOBAL_CACHE',{'chatContextPolicy':{'rounds':12}})
        assert policy.current().rounds==3
        assert render_dialogue(rows()).metrics['injected_rounds']==3
    finally:runtime_config._current_snapshot.reset(token)
    assert policy.current().rounds==12


def test_five_turns_with_four_bubbles_keeps_fifth_user_and_excludes_current():
    history=rows()
    history.append(dict(id='current',role='user',content='当前问题唯一'))
    result=render_dialogue(history, turns=5, exclude_ids={'current'})
    assert '用户选择7' in result.text and '用户选择6' not in result.text
    assert '当前问题唯一' not in result.text
    assert result.metrics['injected_rounds']==5
    assert result.text.count('AI:')==5
    assert '[10-26 08:07]' in result.text
    assert '用户选择7' in result.user_text and '答复' not in result.user_text


def test_previous_identity_and_retrieval_are_not_changed_by_formatter():
    history=rows(1)
    history[1]['metadata']={'memory_retrievals':[{'id':'m'}]}
    turn=previous_assistant_turn(history,None)
    assert turn.id=='a0-3' and turn.retrieval_view().id=='a0-0'
    assert turn.bubble_count==4
    render_dialogue(history)
    assert previous_assistant_turn(history,None)==turn


def test_proactive_and_long_gap_split_in_shared_helper():
    history=rows(1,1)
    history += [dict(id='p',role='assistant',content='主动',metadata={'proactive':True},createdAt=datetime(2026,10,26,0,2,tzinfo=UTC)),
                dict(id='r',role='assistant',content='回复',metadata={},createdAt=datetime(2026,10,26,0,3,tzinfo=UTC)),
                dict(id='later',role='assistant',content='后来',metadata={},createdAt=datetime(2026,10,26,1,tzinfo=UTC))]
    result=render_dialogue(history)
    assert result.text.count('AI:')==4


def test_strict_character_bound_and_no_role_confusion():
    history=[dict(role='system',content='不应进入'),dict(role='user',content='头'+'中'*1000+'尾'),dict(role='assistant',content='前'+'中'*1000+'问题？')]
    result=render_dialogue(history,line_chars=30,max_chars=100)
    assert '不应进入' not in result.text
    assert '用户: 头' in result.text and '尾' not in result.text
    assert 'AI: …' in result.text and result.text.endswith('问题？')
    assert len(result.text)<=100 and result.metrics['truncated']


def test_parallel_consumers_share_bounded_observation_and_reset():
    token=policy.start_observation()
    try:
        render_dialogue(rows(2),purpose='intent')
        render_dialogue(rows(2),purpose='tier')
        assert set(policy.observation()['windows'])=={'intent','tier'}
        assert policy.observation()['main_history_mode']=='budget'
        assert policy.observation()['windows']['tier']['injected_tokens_estimate']>0
    finally:policy.reset_observation(token)
    assert policy.observation() is None


async def test_real_graph_persists_snapshot_and_resets_between_turns(monkeypatch):
    from tests.graph_harness_support import configure_chat
    from app.services.chat import orchestrator as chat
    io = configure_chat(monkeypatch)
    monkeypatch.setattr(runtime_config, '_GLOBAL_CACHE', {'chatContextPolicy': {'rounds': 8}})
    metadata = []
    def create_tracer(*args, **kwargs):
        metadata.append(kwargs['chat_context_policy'])
        return io.tracer
    monkeypatch.setattr(chat, 'create_tracer', create_tracer)
    async def reply(**kwargs):
        result = render_dialogue(rows(), purpose='tier')
        assert result.metrics['effective_rounds'] == 8
        assert 'u-new' in kwargs['exclude_message_ids']
        return ['synthetic reply'], 'synthetic reply', False, {'emotion': '中性'}
    monkeypatch.setattr(chat, '_generate_reply', reply)
    events = [e async for e in chat.stream_chat_response('c-1', '合成消息', io.agent, 'u-1')]
    assert [e['event'] for e in events] == ['reply', 'done']
    saved = chat._save_replies.await_args.args[1][0]['response_diagnostics']['chat_context_policy']
    assert saved['effective_rounds'] == 8
    assert saved['windows']['tier']['injected_rounds'] == 8
    assert chat.run_crisis_guard.await_args.kwargs['exclude_message_ids'] == {'u-new'}
    assert metadata[0]['windows']['tier']['effective_rounds'] == 8
    assert policy.observation() is None
    # Persisted metadata is a copy: later callback mutation cannot rewrite it.
    metadata[0]['windows']['tier']['injected_rounds'] = 1
    assert saved['windows']['tier']['injected_rounds'] == 8


async def test_crisis_classifier_context_excludes_all_current_fragments(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services.chat import crisis_guard_phase as guard
    monkeypatch.setattr(guard, 'should_semantic_crisis_check', lambda _: True)
    monkeypatch.setattr(guard, 'load_crisis_care_state', AsyncMock(return_value=None))
    classifier = AsyncMock(return_value=False)
    messages = [dict(id='old', role='user', content='昨天去过公园'),
                dict(id='a', role='assistant', content='听起来很放松'),
                dict(id='f1', role='user', content='片段一'),
                dict(id='f2', role='user', content='片段二')]
    await guard.run_crisis_guard(
        conversation_id='synthetic', user_id='synthetic', workspace_id=None,
        agent_id=None, user_message='片段一片段二', sub_intent_mode=False,
        messages_dicts=messages, user_message_id='f2',
        exclude_message_ids={'f1', 'f2'}, semantic_classify_fn=classifier)
    assert classifier.await_args.kwargs['message'] == '片段一片段二'
    context = classifier.await_args.kwargs['context']
    assert '昨天去过公园' in context and '听起来很放松' in context
    assert '片段一' not in context and '片段二' not in context
