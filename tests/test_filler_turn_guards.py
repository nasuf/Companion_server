"""Whole-turn guard contract, sole live graph and bound runtime configuration."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.services.chat.filler_reply import build_filler_emoji_reply, is_question_like
from app.services.chat.message_utils import previous_assistant_turn
from app.services.chat.intent_dispatcher import previous_turn_offered_schedule_change
from app.services.interaction.topic_continuity import (
    FOLLOWUP_TRIGGER_TYPE,
    ContinuityState,
    TopicVerdict,
    resolve_cue,
)
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
@pytest.mark.parametrize('prior,probability,selected', [(['晚安'], 1, True), (['晚安'], 0, False),
                                                      (['对吧？', '我听着'], 1, False), (['👌'], 1, False)])
async def test_live_graph_persists_decisions_and_does_not_drop_reply(monkeypatch, prior, probability, selected):
    from tests.graph_harness_support import configure_chat
    from app.services.chat import orchestrator as chat
    io = configure_chat(monkeypatch)
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


def test_feedback_targets_trace_bubble_while_turn_identity_stays_last():
    trace = {'memory_retrievals': [{'selected': [{'id':'m1'}]}]}
    first = message('a0', 'assistant', '你喜欢芒果', metadata=trace)
    last = message('a1', 'assistant', '我记得', at=NOW + timedelta(seconds=4))
    turn = previous_assistant_turn([first, last], None)
    assert turn.id == 'a1' and turn.metadata == {}
    assert turn.content == '你喜欢芒果\n我记得'
    assert turn.createdAt == last.createdAt
    view = turn.retrieval_view()
    assert view.id == 'a0' and view.metadata is trace
    assert view.conversationId == 'c-1' and view.content == turn.content
    assert last.metadata == {}  # never mutate or copy metadata onto the last bubble


def test_topic_verdict_on_last_bubble_matches_retrieval_turn():
    # The followup judge anchors its verdict on the last AI message, while the
    # retrieval snapshot sits on the first one (81% of retrieval turns, 2026-10).
    rows = [message('u0', 'user', '今天面试'),
            message('a0', 'assistant', '怎么啦', metadata={'memory_retrievals': [{}]}),
            message('a1', 'assistant', '准备得怎么样？', at=NOW + timedelta(seconds=3)),
            message('u1', 'user', '还行', at=NOW + timedelta(minutes=20))]
    turn = previous_assistant_turn(rows, 'u1')
    state = ContinuityState(verdict=TopicVerdict('unfinished', 'llm'), anchor_message_id='a1',
                            followup_sent_at=None, last_user_at=None, session_closed=False)
    cue = resolve_cue(state, previous_assistant_id=turn.id, gap_seconds=20 * 60)
    assert cue is not None and cue.unfinished and cue.return_line


def test_followup_after_a_reply_is_its_own_turn():
    rows = [message('u0', 'user', '今天面试'),
            message('a0', 'assistant', '怎么啦', metadata={'memory_retrievals': [{}]}),
            message('a1', 'assistant', '准备得怎么样？', at=NOW + timedelta(seconds=3)),
            message('f0', 'assistant', '在干嘛呢', at=NOW + timedelta(minutes=8),
                    metadata={'proactive': True, 'trigger_type': FOLLOWUP_TRIGGER_TYPE}),
            message('u1', 'user', '好', at=NOW + timedelta(minutes=9))]
    turn = previous_assistant_turn(rows, 'u1')
    assert turn.id == 'f0' and turn.content == '在干嘛呢'
    assert turn.metadata['trigger_type'] == FOLLOWUP_TRIGGER_TYPE
    assert turn.source_message_ids == ('f0',) and turn.previous_user_text is None
    # Feedback on the follow-up must not reach back to the earlier reply's trace.
    assert turn.retrieval_view().id == 'f0'


def test_proactive_card_and_text_stay_one_turn():
    rows = [message('u0', 'user', '嗯'),
            message('p0', 'assistant', '[活动卡片]', at=NOW + timedelta(hours=2), metadata={'proactive': True}),
            message('p1', 'assistant', '周末一起去吗？', at=NOW + timedelta(hours=2, seconds=2),
                    metadata={'proactive': True}),
            message('u1', 'user', '好', at=NOW + timedelta(hours=3))]
    turn = previous_assistant_turn(rows, 'u1')
    assert turn.source_message_ids == ('p0', 'p1') and turn.id == 'p1'
    assert turn.previous_user_text == '嗯'


@pytest.mark.parametrize('later', [
    message('p0', 'assistant', '今天天气不错', at=NOW + timedelta(minutes=2), metadata={'proactive': True}),
    message('a1', 'assistant', '我回来啦', at=NOW + timedelta(minutes=30)),
])
def test_old_schedule_offer_does_not_ground_a_later_turn(later):
    rows = [message('u0', 'user', '困了'), message('a0', 'assistant', '我可以晚点睡'), later,
            message('u1', 'user', '好', at=NOW + timedelta(hours=1))]
    turn = previous_assistant_turn(rows, 'u1')
    assert turn.content == later.content and turn.id == later.id
    assert not previous_turn_offered_schedule_change(turn.content)


@pytest.mark.asyncio
async def test_live_graph_routes_turn_identity_and_retrieval_anchor(monkeypatch):
    from unittest.mock import AsyncMock
    from tests.graph_harness_support import configure_chat
    from app.services.chat import orchestrator as chat
    io = configure_chat(monkeypatch)
    history = [message('u0', 'user', '今天面试'),
               message('a0', 'assistant', '怎么啦', metadata={'memory_retrievals': [{}]}),
               message('a1', 'assistant', '准备得怎么样？', at=NOW + timedelta(seconds=3))]
    # The real query returns newest first; load_turn reverses it.
    io.db.message.find_many.side_effect = lambda **_: history[::-1]
    recorded, corrections = [], []

    async def done():
        return None

    def record(**kwargs):
        recorded.append(kwargs)
        return done()

    async def correction(**kwargs):
        corrections.append(kwargs['previous_assistant'])
        if False:
            yield

    monkeypatch.setattr(chat, '_record_memory_retrieval_feedback', record)
    monkeypatch.setattr(chat, 'resolve_retrieval_feedback_correction', correction)
    continuation = AsyncMock(return_value=None)
    monkeypatch.setattr(chat, 'build_topic_continuation', continuation)
    events = [e async for e in chat.stream_chat_response('c-1', '还行吧，有点紧张', io.agent, 'u-1')]
    assert events[-1]['event'] == 'done'
    assert recorded[0]['assistant_message_id'] == 'a0'
    assert recorded[0]['assistant_reply'] == '怎么啦\n准备得怎么样？'
    assert [view.id for view in corrections] == ['a0']
    assert continuation.await_args.kwargs['previous_assistant'].id == 'a1'


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
