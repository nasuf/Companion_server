"""Independent boundary oracles for read windows; no model judgement involved."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock

import pytest

from app.services.temporal_query import resolve_temporal_query
from app.services.memory.retrieval.timeline import timeline_evidence

UTC = timezone.utc


@pytest.mark.parametrize('zone', ['Asia/Shanghai', 'America/New_York', 'Pacific/Auckland'])
@pytest.mark.parametrize('text,anchor,start,end', [
    ('昨天', '2026-10-21T12:00:00Z', '2026-10-20T00:00', '2026-10-21T00:00'),
    ('上个月', '2026-10-21T12:00:00Z', '2026-09-01T00:00', '2026-10-01T00:00'),
    ('去年', '2026-10-21T12:00:00Z', '2025-01-01T00:00', '2026-01-01T00:00'),
    ('昨天上午', '2026-10-21T12:00:00Z', '2026-10-20T09:00', '2026-10-20T12:00'),
    ('2024年2月29日', '2026-10-21T12:00:00Z', '2024-02-29T00:00', '2024-03-01T00:00'),
    ('昨天到今天', '2026-10-21T12:00:00Z', '2026-10-20T00:00', '2026-10-22T00:00'),
])
def test_half_open_calendar_windows(zone, text, anchor, start, end):
    # Use a local midday anchor, so different zones intentionally share a day.
    local_anchor = datetime.fromisoformat(anchor.replace('Z', '+00:00')).replace(tzinfo=ZoneInfo(zone))
    query = resolve_temporal_query(text, anchor=local_anchor, anchor_source='persisted_message', timezone_name=zone)
    actual = query.retrieval_range()
    expected = tuple(datetime.fromisoformat(t).replace(tzinfo=ZoneInfo(zone)).astimezone(UTC) for t in [start, end])
    assert actual == expected
    assert actual[0].tzinfo == actual[1].tzinfo == UTC
    before_end = expected[1] - timedelta(microseconds=1)
    assert actual[0] <= before_end < actual[1]
    assert not actual[0] <= expected[1] < actual[1]


@pytest.mark.parametrize('day,expected_hours', [('2026-03-08', 23), ('2026-11-01', 25)])
def test_calendar_day_is_not_always_24_elapsed_hours(day, expected_hours):
    query = resolve_temporal_query(day, anchor='2026-12-01T00:00:00Z', anchor_source='persisted_message', timezone_name='America/New_York')
    start, end = query.retrieval_range()
    assert (end - start).total_seconds() == expected_hours * 3600


def test_timeline_uses_query_zone_not_agent_default():
    rows = [{'content':'同一时刻的事件', 'occur_time':'2026-10-20T00:30:00Z', 'source':'user'}]
    text, diagnostics = timeline_evidence(rows, timezone_name='America/Los_Angeles')
    assert '2026-10-19' in text and '2026-10-20' not in text
    assert diagnostics['timezone'] == 'America/Los_Angeles'
    assert diagnostics['status'] == 'rendered'


@pytest.mark.asyncio
async def test_explicit_time_loader_and_hybrid_use_the_same_exclusive_end(monkeypatch):
    from app.services.chat import data_fetch_phase
    from app.services.memory.retrieval import hybrid, vector_search
    query = resolve_temporal_query('上个月面试几次', anchor='2026-10-21T01:00:00Z', anchor_source='persisted_message')
    search = AsyncMock(return_value=[])
    monkeypatch.setattr(vector_search, 'search_by_time_range', search)
    monkeypatch.setattr(hybrid, 'search_by_time_range', search)
    monkeypatch.setattr(hybrid, 'search_similar_tiers', AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid, 'search_related_memories_for_query', AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid, 'cache_retrieval', AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid, 'cache_set_retrieval', AsyncMock())
    await data_fetch_phase._load_time_memories('u', query.parsed_times(), 'w')
    await hybrid.hybrid_retrieve('上个月面试几次', 'u', workspace_id='w', temporal_query=query)
    assert search.await_count == 2
    assert all(c.args[1:] == query.retrieval_range() for c in search.await_args_list)
    assert search.await_args_list[0].args[2] == datetime(2026, 9, 30, 16, tzinfo=UTC)


@pytest.mark.parametrize('text,expected', [
    ('2小时前', '2026-10-21T00:59:59.123456+00:00'),
    ('15分钟前', '2026-10-21T01:45:59.123456+00:00'),
    ('昨天9点', '2026-10-20T02:00:00+00:00'),
])
def test_already_exclusive_clock_offsets_are_not_rounded_up(text, expected):
    q = resolve_temporal_query(text, anchor='2026-10-21T01:59:59.123456Z', anchor_source='persisted_message')
    assert q.retrieval_range()[1].isoformat() == expected


@pytest.mark.parametrize('text', ['昨天说过的旅行', '昨天和前天', '明天', '你好', '9999年12月31日'])
def test_unavailable_or_future_windows_are_explicit(text):
    q = resolve_temporal_query(text, anchor='2026-10-21T01:00:00Z', anchor_source='persisted_message')
    d = q.diagnostics()['retrieval_window']
    assert d['status'] == 'not_eligible'
    assert d['start'] is d['end_exclusive'] is None


def test_window_diagnostics_are_bounded_metadata_not_input_text():
    q = resolve_temporal_query('上个月私人秘密面试', anchor='2026-10-21T01:00:00Z', anchor_source='persisted_message')
    d = q.diagnostics()['retrieval_window']
    assert d == {'version':1, 'status':'eligible', 'bounds':'[start,end)',
        'start':'2026-08-31T16:00:00+00:00', 'end_exclusive':'2026-09-30T16:00:00+00:00'}
    assert '秘密' not in str(q.diagnostics())


def test_invalid_timeline_timezone_is_not_silently_replaced():
    rows = [{'content':'事件', 'occur_time':'2026-10-20T00:30:00Z', 'source':'user'}]
    text, d = timeline_evidence(rows, timezone_name='invalid/private-zone')
    assert text is None and d['status'] == 'unavailable' and d['reason'] == 'invalid_timezone'
    assert d['timezone'] is None and d['shown_count'] == 0 and d['dated_count'] == 1
    assert 'private-zone' not in str(d)


def test_empty_timeline_is_valid_zero_not_unavailable():
    text, d = timeline_evidence([], timezone_name='Europe/London')
    assert text is None and d['status'] == 'rendered'
    assert d['candidate_count'] == d['shown_count'] == 0 and not d['truncated']


def test_legacy_parser_dates_and_schedule_inputs_are_unchanged():
    from app.services.schedule_domain.time_parser import parse_time_expressions
    anchor = datetime(2026, 10, 21, 9, tzinfo=ZoneInfo('Asia/Shanghai'))
    expected = {
        '昨天': ('2026-10-20T00:00:00+08:00','2026-10-20T23:59:59.999999+08:00'),
        '上个月': ('2026-09-01T00:00:00+08:00','2026-09-30T23:59:59+08:00'),
        '上午': ('2026-10-21T09:00:00+08:00','2026-10-21T11:59:59+08:00'),
        '10点': ('2026-10-21T10:00:00+08:00','2026-10-21T11:00:00+08:00'),
    }
    for text, (start,end) in expected.items():
        parsed = parse_time_expressions(text, now=anchor)[0]
        assert parsed.start.isoformat() == start and parsed.end.isoformat() == end


@pytest.mark.asyncio
async def test_hybrid_timeline_respects_zone_even_on_cache_hits(monkeypatch):
    from app.services.memory.retrieval import hybrid as h
    row={'id':'m1','content':'用户曾去展馆', 'occur_time':'2026-10-20T00:30:00Z',
         'source':'user','level':2,'importance':0.7,'similarity':0.9}
    monkeypatch.setattr(h,'search_similar_tiers',AsyncMock(return_value=[row]))
    monkeypatch.setattr(h,'search_related_memories_for_query',AsyncMock(return_value=[]))
    monkeypatch.setattr(h,'search_by_time_range',AsyncMock(return_value=[]))
    cache={}
    async def get(key, *_args, **_kwargs): return cache.get(key)
    async def put(key, _uid, value, **_kwargs): cache[key]=value
    monkeypatch.setattr(h,'cache_retrieval',get)
    monkeypatch.setattr(h,'cache_set_retrieval',put)
    question='上次去展馆是什么时候'
    for zone,day in [('America/Los_Angeles','2026-10-19'),('Asia/Shanghai','2026-10-20'),('America/Los_Angeles','2026-10-19')]:
        q=resolve_temporal_query(question,anchor='2026-10-21T00:00:00Z',anchor_source='persisted_message',timezone_name=zone)
        result=await h.hybrid_retrieve(question,'u',workspace_id='w',temporal_query=q)
        assert day in result['timeline'] and result['timeline_diagnostics']['timezone']==zone
    assert h.search_similar_tiers.await_count == 2


@pytest.mark.parametrize('text,start,end,precision', [
    ('昨天9点30分', '2026-10-20T01:30:00Z', '2026-10-20T01:31:00Z', 'minute'),
    ('昨天9时30', '2026-10-20T01:30:00Z', '2026-10-20T01:31:00Z', 'minute'),
    ('昨天9点', '2026-10-20T01:00:00Z', '2026-10-20T02:00:00Z', 'hour'),
    ('昨天晚上12点30分', '2026-10-20T16:30:00Z', '2026-10-20T16:31:00Z', 'minute'),
    ('9点30分到10点30分', '2026-10-21T01:30:00Z', '2026-10-21T02:31:00Z', 'range'),
])
def test_explicit_minute_queries_do_not_widen_to_one_hour(text,start,end,precision):
    q=resolve_temporal_query(text,anchor='2026-10-21T04:00:00Z',anchor_source='persisted_message')
    assert q.retrieval_range() == tuple(datetime.fromisoformat(t.replace('Z','+00:00')) for t in [start,end])
    assert q.expressions[0].precision == precision
    assert q.parsed_times()[0].end_exclusive == q.retrieval_range()[1]


def test_minute_read_refinement_preserves_scheduling_clock():
    from app.services.schedule_domain.time_parser import parse_time_expressions
    t=parse_time_expressions('9点30分',now=datetime(2026,10,21,9,tzinfo=ZoneInfo('Asia/Shanghai')))[0]
    assert (t.start.hour,t.start.minute,t.end.hour,t.end.minute)==(9,30,10,30)


@pytest.mark.parametrize('day,hour', [('2026-03-08',2),('2026-11-01',1)])
def test_explicit_minutes_in_skipped_or_repeated_hours_remain_ambiguous(day,hour):
    q=resolve_temporal_query(f'{day}凌晨{hour}点30分',anchor='2026-12-01T00:00:00Z',anchor_source='persisted_message',timezone_name='America/New_York')
    assert q.reason=='ambiguous_clock' and q.retrieval_range() is None


@pytest.mark.parametrize('anchor,start,end', [
    ('2026-03-08T16:00:00Z', '2026-03-07T06:59:00Z', '2026-03-07T07:00:00Z'),
    ('2026-11-01T17:00:00Z', '2026-10-31T05:59:00Z', '2026-10-31T06:00:00Z'),
])
def test_minute_precision_uses_event_day_not_anchor_day_dst(anchor,start,end):
    q=resolve_temporal_query('昨天1点59分',anchor=anchor,anchor_source='persisted_message',timezone_name='America/New_York')
    assert q.retrieval_range() == tuple(datetime.fromisoformat(t.replace('Z','+00:00')) for t in [start,end])
