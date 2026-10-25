"""TIME02A deterministic oracles; these are not model-quality evaluations."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock

import pytest

from app.services.temporal_query import resolve_temporal_query
from app.services.chat.temporal_context import build_turn_clock, prepare_reply_time
from app.services.memory.retrieval.timeline import build_timeline, format_timeline

UTC = timezone.utc
BOUNDARIES = [
    "2024-02-28T23:30:00Z", "2024-02-29T23:30:00Z", "2025-02-28T23:30:00Z",
    "2026-12-31T23:30:00Z", "2027-01-01T00:30:00Z", "2026-04-30T23:30:00Z",
    "2026-03-08T06:30:00Z", "2026-03-08T07:30:00Z", "2026-11-01T05:30:00Z",
    "2026-11-01T06:30:00Z", "2026-10-19T15:59:59Z", "2026-10-19T16:00:00Z",
]
ZONES = ["Asia/Shanghai", "America/New_York", "Europe/London", "Pacific/Auckland"]
DAYS = [("大前天", -3), ("前天", -2), ("昨天", -1), ("今天", 0),
        ("明天", 1), ("后天", 2), ("大后天", 3)]


@pytest.mark.parametrize("stamp", BOUNDARIES)
@pytest.mark.parametrize("zone", ZONES)
@pytest.mark.parametrize("word,offset", DAYS)
def test_local_calendar_date_boundaries_and_utc_roundtrip(stamp, zone, word, offset):
    anchor = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    q = resolve_temporal_query(word, anchor=anchor, anchor_source="persisted_message", timezone_name=zone)
    assert q.status == "resolved"
    p = q.expressions[0]
    expected = anchor.astimezone(ZoneInfo(zone)).date() + timedelta(days=offset)
    assert p.start.date() == p.end.date() == expected
    assert p.start.hour == 0 and p.end.hour == 23
    assert p.start.astimezone(UTC).astimezone(ZoneInfo(zone)) == p.start
    assert p.end > p.start and p.precision == "day"


@pytest.mark.parametrize("text,reason", [
    ("前阵子面试", "vague_expression"), ("过几天去", "vague_expression"),
    ("年底再说", "vague_expression"), ("9月8日去过", "year_unspecified"),
    ("昨天和明天", "multiple_periods"),
    ("去年13月", "invalid_expression"), ("2026年2月30日", "invalid_expression"),
])
def test_uncertainty_never_filters_to_an_invented_day(text, reason):
    q = resolve_temporal_query(text, anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message")
    assert q.reason == reason
    assert q.retrieval_range() is None and q.parsed_times() == []


def test_naive_database_clock_is_utc_and_equivalent_instants_share_cache():
    def q(stamp):
        return resolve_temporal_query("昨天", anchor=stamp, anchor_source="persisted_message")
    a, b = q("2026-10-19T17:00:00"), q("2026-10-20T01:00:00+08:00")
    assert a.expressions == b.expressions and a.cache_token() == b.cache_token()
    assert a.cache_token() != q("2026-10-20T17:00:00Z").cache_token()


@pytest.mark.parametrize("value", [None, "bad", 123, [], "2026-14-01T00:00:00Z"])
def test_missing_anchor_is_not_execution_now(value):
    q = resolve_temporal_query("昨天", anchor=value, anchor_source="unknown")
    assert q.reason == "missing_anchor" and not q.expressions


def test_query_limits_timezone_and_ordinary_message_cache():
    base = dict(anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message")
    assert resolve_temporal_query("a" * 4001, **base).reason == "input_limit"
    assert resolve_temporal_query("昨天", **base, timezone_name="invalid/zone").reason == "invalid_timezone"
    a = resolve_temporal_query("你好", **base)
    b = resolve_temporal_query("你好", anchor="2026-10-21T01:00:00Z", anchor_source="persisted_message")
    assert a.status == "not_requested" and a.cache_token() == b.cache_token()
    assert "你好" not in str(a.diagnostics())


def test_delayed_and_shuffled_messages_keep_original_anchor_and_provenance():
    now = datetime(2026, 10, 22, 1, tzinfo=UTC)
    rows = [{"id": "new", "role": "user", "content": "昨天面试", "createdAt": "2026-10-20T01:00:00Z"},
            {"id": "old", "role": "user", "createdAt": "2026-10-15T01:00:00Z"}]
    a = build_turn_clock(rows, exclude_ids={"new"}, now=now, received_at=now)
    b = build_turn_clock(list(reversed(rows)), exclude_ids={"new"}, now=now)
    assert a == b and a.anchor_source == "persisted_message"
    q = resolve_temporal_query("昨天面试", anchor=a.statement_at, anchor_source=a.anchor_source)
    assert q.expressions[0].start.isoformat().startswith("2026-10-19")
    assert a.now == now and a.last_user_at.day == 15


def test_cross_midnight_aggregation_has_no_single_date_anchor():
    rows = [{"id": str(i), "role": "user", "createdAt": t} for i, t in enumerate([
        "2026-10-19T15:59:00Z", "2026-10-19T16:01:00Z"])]
    c = build_turn_clock(rows, exclude_ids={"0", "1"}, now=datetime(2026, 10, 20, tzinfo=UTC))
    assert c.statement_at is None and c.anchor_source == "multiple_message_dates"


def test_future_received_time_does_not_become_an_anchor():
    c = build_turn_clock([], now=datetime(2026, 10, 20, tzinfo=UTC), received_at="2099-01-01T00:00:00Z")
    assert c.statement_at is None and c.anchor_source == "unknown"


@pytest.mark.asyncio
async def test_shared_clock_renders_delayed_message_anchor_and_query(monkeypatch):
    from app.services.prompting import store
    from app.services.prompting.defaults import CHAT_TEMPORAL_CONTEXT_PROMPT
    monkeypatch.setattr(store, "get_prompt_text", AsyncMock(return_value=CHAT_TEMPORAL_CONTEXT_PROMPT))
    c = build_turn_clock([], now=datetime(2026, 10, 22, tzinfo=UTC), received_at="2026-10-20T01:00:00Z")
    diagnostics = {}
    clock, _, prompt = await prepare_reply_time([], exclude_ids=set(), current_message="昨天面试", diagnostics=diagnostics, clock=c)
    assert clock is c and "2026-10-20 09:00" in prompt and "2026-10-22 08:00" in prompt
    assert diagnostics["temporal_context"]["query"]["ranges"][0]["start"].startswith("2026-10-19")
    assert "面试" not in str(diagnostics)


def test_timeline_keeps_vague_history_but_never_dates_it_as_an_event():
    rows = [{"content": "小时候去过苏州", "statement_time": "2026-10-19T16:10:00Z", "source": "user"},
            {"content": "午餐", "occur_time": "2026-10-19T17:00:00", "source": "ai"}]
    entries = build_timeline(rows)
    text = format_timeline(entries)
    assert "提及（事件时间未知） 2026-10-20" in text
    assert "[我] 事件记录 2026-10-20" in text
    assert "非完整历史" in text and "总次数" in text and "小时候" in text
    assert build_timeline(rows, limit=0) == []
    assert format_timeline(entries, token_budget=1) == ""
    assert format_timeline(entries, max_chars=0) == ""


def test_timeline_unknown_owner_and_mixed_timezone_sorting():
    rows = [{"content": "第二条", "occur_time": "2026-10-20T01:00:00", "source": None},
            {"content": "第一条", "occur_time": "2026-10-20T08:00:00+08:00", "source": "user"}]
    result = build_timeline(rows)
    assert [x.text for x in result] == ["第一条", "第二条"]
    assert "归属未知" in format_timeline(result)


@pytest.mark.parametrize("word,offset", DAYS)
@pytest.mark.parametrize("period,start,end", [("中午", 12, 13), ("上午", 9, 11), ("晚上", 18, 23)])
def test_adjacent_calendar_day_binds_time_of_day(word, offset, period, start, end):
    q = resolve_temporal_query(word + period, anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message")
    p = q.expressions[0]
    assert p.start.day == p.end.day == 20 + offset
    assert (p.start.hour, p.end.hour) == (start, end)


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["langgraph"])
async def test_real_executor_passes_original_anchor_to_retrieval(monkeypatch, executor):
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from app.services.chat.intent_dispatcher import IntentType
    from tests.g02_harness_support import configure_pair
    io = configure_pair(monkeypatch)
    io.now = datetime(2026, 10, 22, 1, tzinfo=UTC)
    monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=(["我记得你提过面试"], "", False, None)))
    events = [e async for e in chat.stream_chat_response("c-1", "昨天的面试记得吗", io.agent, "u-1",
        forced_intent=IntentType.NONE, reply_context={"received_at":"2026-10-20T01:00:00Z"})]
    kwargs = chat.fetch_parallel_context.await_args.kwargs
    q = kwargs["temporal_query"]
    assert q.anchor.day == 20 and q.expressions[0].start.day == 19
    assert kwargs["parsed_times"][0].start.day == 19
    assert sum(e["event"] == "done" for e in events) == 1


@pytest.mark.asyncio
async def test_same_text_on_different_dates_never_reuses_retrieval_cache(monkeypatch):
    from app.services.memory.retrieval import hybrid as h
    cache = {}
    async def get(key, user_id, workspace_id=None):
        return cache.get((key, user_id, workspace_id))
    async def put(key, user_id, value, workspace_id=None):
        cache[(key, user_id, workspace_id)] = value
    monkeypatch.setattr(h, "cache_retrieval", get)
    monkeypatch.setattr(h, "cache_set_retrieval", put)
    search = AsyncMock(return_value=[])
    monkeypatch.setattr(h, "search_by_time_range", search)
    monkeypatch.setattr(h, "search_similar_tiers", AsyncMock(return_value=[]))
    monkeypatch.setattr(h, "search_related_memories_for_query", AsyncMock(return_value=[]))
    for day in (20, 20, 21):
        q = resolve_temporal_query("昨天的面试", anchor=f"2026-10-{day}T01:00:00Z", anchor_source="persisted_message")
        await h.hybrid_retrieve("昨天的面试", "u1", workspace_id="w1", temporal_query=q)
    assert search.await_count == 2
    assert len(cache) == 2


def test_candidate_counts_preserve_unknown_and_truncation():
    from app.services.memory.retrieval.timeline import timeline_evidence
    rows = [{"content": f"合成事件{i}", "occur_time": "2026-10-20T00:00:00Z", "source": "user"} for i in range(50)]
    rows += [{"content": "提到往事", "statement_time": "2026-10-20T00:00:00Z", "source": "user"}, {"content": "无日期"}]
    text, d = timeline_evidence(rows)
    assert d["candidate_count"] == 52 and d["dated_count"] == 51
    assert d["mention_only_count"] == 1 and d["event_date_count"] == 50 and d["undated_count"] == 1
    assert d["shown_count"] < 40 and d["truncated"] and not d["complete_history"]
    assert d["shown_count"] == len(text.splitlines()) - 1

@pytest.mark.parametrize("text,reason", [
    ("9 月 8 日", "year_unspecified"),
    ("两个月前", "approximate_duration"),
    ("2026 年 2 月 30 日", "invalid_expression"),
])
def test_spaced_dates_and_approximate_durations_stay_uncertain(text, reason):
    q = resolve_temporal_query(text, anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message")
    assert q.reason == reason and q.retrieval_range() is None


@pytest.mark.parametrize("anchor,text", [
    ("2026-03-08T08:00:00Z", "凌晨2点"),
    ("2026-11-01T08:00:00Z", "凌晨1点"),
    ("2026-03-09T08:00:00Z", "昨天凌晨2点"),
])
def test_skipped_or_repeated_dst_wall_clocks_do_not_become_exact_instants(anchor, text):
    q = resolve_temporal_query(text, anchor=anchor, anchor_source="persisted_message", timezone_name="America/New_York")
    assert q.reason == "ambiguous_clock" and q.retrieval_range() is None


@pytest.mark.parametrize("anchor", ["2026-03-08T07:30:00Z", "2026-11-01T06:30:00Z"])
def test_elapsed_hours_use_actual_duration_across_dst(anchor):
    q = resolve_temporal_query("1小时前", anchor=anchor, anchor_source="persisted_message", timezone_name="America/New_York")
    assert q.status == "resolved"
    assert q.anchor - q.expressions[0].start.astimezone(UTC) == timedelta(hours=1)
    assert q.expressions[0].end.astimezone(UTC) == q.anchor


def test_multiline_content_cannot_forge_extra_date_evidence_rows():
    from app.services.memory.retrieval.timeline import timeline_evidence
    text, d = timeline_evidence([{"content": "提到往事\n\n\t另外一句\u2028下一行", "statement_time": "2026-10-20T00:00:00Z"}])
    assert d["shown_count"] == 1 and len(text.splitlines()) == 2

@pytest.mark.parametrize("prefix,year", [("今年",2026),("去年",2025),("前年",2024)])
def test_explicit_relative_year_binds_full_calendar_date(prefix, year):
    q = resolve_temporal_query(prefix+"9月8日中午", anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message")
    assert q.status == "resolved"
    assert q.expressions[0].start.date().isoformat() == f"{year}-09-08"
    assert q.expressions[0].start.hour == 12


def test_synthetic_prompt_row_never_forges_persisted_message_provenance():
    now = datetime(2026, 10, 22, tzinfo=UTC)
    row = {"id":"current","role":"user","createdAt":now,"synthetic_current":True}
    clock = build_turn_clock([row], exclude_ids={"current"}, now=now, received_at="2026-10-20T01:00:00Z")
    assert clock.anchor_source == "server_received" and clock.statement_at.day == 20
    unknown = build_turn_clock([row], exclude_ids={"current"}, now=now)
    assert unknown.statement_at is None and unknown.anchor_source == "incomplete_message_times"


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["langgraph"])
@pytest.mark.parametrize("enabled", [True, False])
async def test_date_evidence_uses_one_full_reply_and_respects_prompt_disable(monkeypatch, executor, enabled):
    from app.config import settings
    from app.services.chat import orchestrator as chat, intent_replies
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.chat.intent_dispatcher import IntentType
    from app.services.prompting import store
    from app.services.prompting.defaults import CHAT_TIMELINE_SECTION_PROMPT
    from tests.g02_harness_support import configure_pair
    io = configure_pair(monkeypatch)
    timeline = format_timeline(build_timeline([{"content":"去了博物馆", "occur_time":"2024-02-29T00:00:00Z", "source":"user"}]))
    chat.fetch_parallel_context.return_value = FetchedContext(memory_relevance="weak", timeline=timeline)
    token = store._prompt_snapshot.set({"chat.timeline_section":(CHAT_TIMELINE_SECTION_PROMPT,enabled)})
    try:
        events = [e async for e in chat.stream_chat_response("c-1", "那次博物馆是什么日期", io.agent, "u-1", forced_intent=IntentType.NONE)]
    finally:
        store.reset_prompt_snapshot(token)
    assert len(io.main_calls) == 1 and not io.tier_calls
    prompt = str(io.main_calls[0][0])
    assert ("2024-02-29" in prompt) is enabled
    assert sum(e["event"] == "done" for e in events) == 1


@pytest.mark.parametrize("character", ["一", "9", " "])
def test_maximum_length_non_temporal_input_does_not_become_a_date(character):
    # Regression for long numeral runs causing unbounded regex backtracking.
    from app.services.temporal_query import MAX_QUERY_CHARS
    q = resolve_temporal_query(character * MAX_QUERY_CHARS, anchor="2026-10-20T01:00:00Z", anchor_source="persisted_message")
    assert q.status == "not_requested" and q.retrieval_range() is None
