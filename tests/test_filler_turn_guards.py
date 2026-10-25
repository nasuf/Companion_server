"""Whole-turn guard contract, both live executors and bound runtime configuration."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.services.chat.filler_reply import build_filler_emoji_reply, is_question_like
from app.services.chat.message_utils import previous_assistant_turn
from app.services.chat.intent_dispatcher import previous_turn_offered_schedule_change
from app.services import runtime_config

NOW = datetime(2026, 10, 25, 10, tzinfo=UTC)


def message(id, role, content, *, at=NOW, metadata=None):
    return SimpleNamespace(id=id, role=role, content=content, createdAt=at, metadata=metadata or {}, conversationId='c-1')


@pytest.mark.parametrize('as_dict', [False, True])
def test_whole_turn_keeps_earlier_question_and_last_identity(as_dict):
    rows = [message('u0', 'user', '聊聊'), message('a0', 'assistant', '是这样，对吧？'),
            message('a1', 'assistant', '我听着'), message('u1', 'user', '是的'),
            message('future', 'assistant', 'later')]
    if as_dict:
        rows = [vars(row) for row in rows]
    turn = previous_assistant_turn(rows, 'u1')
    assert turn.content == '是这样，对吧？\n我听着'
    assert turn.id == 'a1'
    assert turn.conversationId == 'c-1'
    assert turn.source_message_ids == ('a0', 'a1')
    assert turn.previous_user_text == '聊聊'
    diagnostics = {}
    assert build_filler_emoji_reply('是的', previous_assistant=turn, probability=1, diagnostics=diagnostics) is None
    assert diagnostics['filler_emoji']['reason'] == 'previous_turn_question'


@pytest.mark.parametrize('text', ['好吗 😊', '对吧？🙂', '去不去嘛！✨', '你觉得呢\n晚安', '要继续吗？\n那我等你'])
def test_question_guard_handles_emoji_and_all_bubbles(text):
    assert is_question_like(text)


@pytest.mark.parametrize('text', ['', '晚安，做个好梦😊', '你好！', '不', '我不去了'])
def test_question_guard_conservative_non_questions(text):
    if text == '不':
        assert is_question_like(text)  # preserve the existing conservative suffix guard
    else:
        assert not is_question_like(text)


@pytest.mark.parametrize('seconds,repeat', [(0, True), (60, True), (61, False), (-1, None)])
def test_repeat_window_and_prior_emoji_are_independent_guards(seconds, repeat):
    turn = previous_assistant_turn([message('u0', 'user', '好'), message('a0', 'assistant', '👌')], None)
    diagnostics = {}
    assert build_filler_emoji_reply('好', previous_assistant=turn, probability=1,
                                   received_at=NOW + timedelta(seconds=seconds), diagnostics=diagnostics) is None
    guards = diagnostics['filler_emoji']['guards']
    assert guards['previous_turn_is_emoji_only'] is True
    assert guards['repeated_after_emoji_within_60s'] is repeat


def test_missing_timestamps_do_not_claim_repeat_evidence():
    diagnostics = {}
    assert build_filler_emoji_reply('嗯', previous_assistant_text='🙂', probability=1, diagnostics=diagnostics) is None
    assert diagnostics['filler_emoji']['seconds_since_previous_reply'] is None
    assert diagnostics['filler_emoji']['guards']['repeated_after_emoji_within_60s'] is None


def test_zero_default_never_shortcuts_and_bound_turn_is_stable(monkeypatch):
    monkeypatch.setattr(runtime_config, '_GLOBAL_CACHE', {})
    for text in ['好', '嗯', '是的', '哈哈']:
        assert build_filler_emoji_reply(text) is None
    snapshot = runtime_config.ConfigurationSnapshot({'fillerEmojiProbability': 0}, {}, {}, {})
    token = runtime_config._current_snapshot.set(snapshot)
    try:
        monkeypatch.setattr(runtime_config, '_GLOBAL_CACHE', {'fillerEmojiProbability': 1})
        assert build_filler_emoji_reply('好') is None
    finally:
        runtime_config._current_snapshot.reset(token)
    assert build_filler_emoji_reply('好') is not None


def test_schedule_offer_in_earlier_bubble_is_preserved():
    turn = previous_assistant_turn([message('a0', 'assistant', '我可以晚点睡'),
                                    message('a1', 'assistant', '我等你'), message('u1', 'user', '好')], 'u1')
    assert previous_turn_offered_schedule_change(turn.content)
    assert build_filler_emoji_reply('好', previous_assistant=turn, probability=1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('executor', ['langgraph', 'legacy'])
@pytest.mark.parametrize('prior,probability,selected', [(['晚安'], 1, True), (['晚安'], 0, False),
                                                      (['对吧？', '我听着'], 1, False), (['👌'], 1, False)])
async def test_live_paths_persist_decisions_and_do_not_drop_reply(monkeypatch, executor, prior, probability, selected):
    from tests.graph_harness_support import configure_chat
    from app.services.chat import orchestrator as chat
    from app.config import settings
    io = configure_chat(monkeypatch)
    monkeypatch.setattr(settings, 'chat_executor', executor)
    monkeypatch.setattr(chat, 'build_filler_emoji_reply', build_filler_emoji_reply)
    monkeypatch.setattr(runtime_config, '_GLOBAL_CACHE', {'fillerEmojiProbability': probability})
    history = [message('u0', 'user', '好')] + [message(f'a{i}', 'assistant', text) for i,text in enumerate(prior)]
    io.db.message.find_many.side_effect = lambda **_: history.copy()
    events = [e async for e in chat.stream_chat_response('c-1', '好', io.agent, 'u-1')]
    assert [e['event'] for e in events] == ['reply', 'done']
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()
    persisted = chat._save_replies.await_args.args[1][0]
    diag = persisted['response_diagnostics']
    assert diag['filler_emoji']['probability'] == probability
    assert (diag['reply_path'] == 'filler_emoji') is selected
    assert chat._generate_reply.await_count == (0 if selected else 1)
    if len(prior) > 1:
        assert diag['filler_emoji']['previous_turn_bubble_count'] == len(prior)


def test_feedback_uses_first_bubble_trace_but_whole_text_and_last_time():
    trace = {'memory_retrievals': [{'selected': [{'id':'m1'}]}]}
    first = message('a0', 'assistant', '你喜欢芒果', metadata=trace)
    last = message('a1', 'assistant', '我记得', at=NOW + timedelta(seconds=4))
    turn = previous_assistant_turn([first, last], None)
    assert turn.id == 'a0' and turn.metadata is trace
    assert turn.content == '你喜欢芒果\n我记得'
    assert turn.createdAt == last.createdAt
    assert last.metadata == {}  # never mutate or copy metadata onto the last bubble


@pytest.mark.asyncio
async def test_background_feedback_does_not_replace_whole_text_with_one_bubble(monkeypatch):
    from unittest.mock import AsyncMock, Mock
    from app.services.chat import orchestrator as chat
    from app.services.memory.retrieval import trace
    db = SimpleNamespace(message=SimpleNamespace(find_unique=AsyncMock(return_value=message('a0','assistant','first'))))
    monkeypatch.setattr(chat, 'db', db)
    builder = Mock(return_value=None)
    monkeypatch.setattr(trace, 'build_memory_retrieval_feedback', builder)
    await chat._record_memory_retrieval_feedback(assistant_message_id='a0', assistant_reply='first\nsecond', user_message='不对', user_message_id='u1')
    assert builder.call_args.kwargs['previous_assistant_reply'] == 'first\nsecond'
