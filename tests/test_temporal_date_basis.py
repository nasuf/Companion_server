"""Recall must not confuse the date of a statement with the date of its subject."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.temporal_query import resolve_temporal_query
from app.services.chat.temporal_context import scope_event_memories
from app.services.memory.retrieval.context_selector import ClassifiedMemory

NOW = datetime(2026, 10, 20, 4, tzinfo=timezone.utc)


def query(text):
    return resolve_temporal_query(text, anchor=NOW, anchor_source='persisted_message')


@pytest.mark.parametrize('text', [
    '昨天我提过哪次旅行？', '昨天聊起的电影叫什么？', '昨天告诉你的收养日期是哪天？',
    '2026年10月19日的日记写了哪件旧事？', '今天导入的那次旅行是什么时候？',
    '昨天说过今天要去看展，最后去了吗？', '昨天转发的原帖说明什么时候开业？',
    '昨天问过的活动，是哪年的？', '昨天记录的事实际上哪天发生？',
])
def test_mention_and_record_references_do_not_filter_event_occurrence(text):
    q = query(text)
    assert q.date_basis == 'unbound_reference'
    assert not q.allows_event_range() and q.retrieval_range() is None and q.parsed_times() == []
    assert q.diagnostics()['date_basis']['event_range_eligible'] is False
    assert text not in str(q.diagnostics())
    memory = ClassifiedMemory('2018年8月2日收养了猫', 'strong', .8,
        main_category='生活', occur_time='2018-08-02T04:00:00Z', statement_time='2026-10-19T04:00:00Z')
    diagnostics = {}
    assert scope_event_memories([memory], text, now=NOW, diagnostics=diagnostics, temporal_query=q) == [memory]
    assert 'requested_event_date' not in diagnostics['temporal_context']
    assert diagnostics['temporal_context']['out_of_window_event_count'] == 0


def test_explicit_event_query_still_scopes_and_reference_preserves_the_parsed_calendar():
    q = query('昨天去了哪里？')
    assert q.retrieval_range() and q.parsed_times()
    assert q.diagnostics()['date_basis'] == {'version':1,'kind':'event_candidate','event_range_eligible':True}
    old = ClassifiedMemory('去海边', 'strong', .8, main_category='生活', occur_time='2026-01-02T04:00:00Z')
    current = ClassifiedMemory('去公园', 'strong', .8, main_category='生活', occur_time='2026-10-19T04:00:00Z')
    assert scope_event_memories([old,current], '昨天去了哪里？', now=NOW, diagnostics={}, temporal_query=q) == [current]
    mentioned = query('昨天提过哪次旅行？')
    assert mentioned.status == 'resolved' and mentioned.expressions == q.expressions
    assert mentioned.cache_token() != q.cache_token()
    assert query('问候你，昨天去了哪里？').allows_event_range()
    assert query('今天说话嗓子疼吗？').date_basis == 'event_candidate'


def test_query_limit_does_not_scan_or_claim_an_event_basis():
    q = query('x'*4001)
    assert q.date_basis == 'unavailable' and not q.allows_event_range()


@pytest.mark.asyncio
async def test_reference_recall_keeps_semantic_candidates_without_date_search(monkeypatch):
    from app.services.memory.retrieval import hybrid as h
    monkeypatch.setattr(h, 'cache_retrieval', AsyncMock(return_value=None))
    monkeypatch.setattr(h, 'cache_set_retrieval', AsyncMock())
    search = AsyncMock(return_value=[])
    vector = AsyncMock(return_value=[])
    monkeypatch.setattr(h, 'search_by_time_range', search)
    monkeypatch.setattr(h, 'search_similar_tiers', vector)
    monkeypatch.setattr(h, 'search_related_memories_for_query', AsyncMock(return_value=[]))
    await h.hybrid_retrieve('昨天提过哪次旅行？', 'u1', workspace_id='w1', temporal_query=query('昨天提过哪次旅行？'))
    search.assert_not_awaited()
    vector.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('executor', ['legacy','langgraph'])
async def test_both_executors_share_unbound_date_contract(monkeypatch, executor):
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from app.services.chat.intent_dispatcher import IntentType
    from tests.g02_harness_support import configure_pair
    io = configure_pair(monkeypatch)
    io.now = NOW
    monkeypatch.setattr(chat, '_generate_reply', AsyncMock(return_value=(['是那次旅行'], '', False, None)))
    events = [e async for e in chat.stream_chat_response('c-1','昨天告诉你的收养日期是什么？',io.agent,'u-1',
        forced_intent=IntentType.NONE, reply_context={'received_at':NOW.isoformat()})]
    args = chat.fetch_parallel_context.await_args.kwargs
    assert args['temporal_query'].date_basis == 'unbound_reference'
    assert args['parsed_times'] == []
    assert sum(e['event']=='done' for e in events) == 1


@pytest.mark.asyncio
async def test_filter_diagnostics_do_not_change_the_registry_prompt_contract(monkeypatch):
    from app.services.prompting import store
    from app.services.prompting.defaults import CHAT_TEMPORAL_CONTEXT_PROMPT
    from app.services.chat.temporal_context import build_turn_clock, prepare_reply_time
    monkeypatch.setattr(store, 'get_prompt_text', AsyncMock(return_value=CHAT_TEMPORAL_CONTEXT_PROMPT))
    diagnostics = {}
    _, _, prompt = await prepare_reply_time([], exclude_ids=set(), current_message='昨天聊起的旅行去哪了？',
        clock=build_turn_clock([], now=NOW, received_at=NOW), diagnostics=diagnostics)
    assert diagnostics['temporal_context']['query']['date_basis']['kind']=='unbound_reference'
    assert 'date_basis' not in prompt and 'event_range_eligible' not in prompt
    assert '"ranges"' in prompt and '2026-10-19' in prompt
