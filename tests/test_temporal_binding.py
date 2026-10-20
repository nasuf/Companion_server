"""Generic syntax counterexamples: preserve recall unless one date scope is clear."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.temporal_query import resolve_temporal_query
from app.services.chat.temporal_context import scope_event_memories
from app.services.memory.retrieval.context_selector import ClassifiedMemory

NOW = datetime(2026, 10, 20, 4, tzinfo=timezone.utc)


def query(text):
    return resolve_temporal_query(text, anchor=NOW, anchor_source="persisted_message")


UNBOUND = [
    "今年去过南京，上个月搬了家", "本周参加会议，昨天去医院",
    "昨天加班，昨天晚上散步", "昨天早上见朋友，昨天晚上看电影",
    "今年去了广州，上个月去了成都，昨天去了杭州",
    "不是昨天去的，哪天去的？", "昨天没有去医院", "假如昨天去了就好了",
    "原定昨天去医院，后来取消了", "她说“昨天去了海边”，实际是哪天？",
    "2026年10月1日到了南京，2026年10月1日又去了苏州",
]


@pytest.mark.parametrize("text", UNBOUND)
def test_independent_dates_and_nonfactual_dates_never_exclude_event_candidates(text):
    q = query(text)
    assert not q.allows_event_range()
    assert q.retrieval_range() is None and q.parsed_times() == []
    memories = [ClassifiedMemory("事件", "strong", .8, main_category="生活",
                                occur_time=date) for date in
                ["2026-01-02T04:00:00Z", "2026-09-02T04:00:00Z", "2026-10-19T04:00:00Z"]]
    diagnostics = {}
    assert scope_event_memories(memories, text, now=NOW, diagnostics=diagnostics,
                                temporal_query=q) == memories
    assert diagnostics["temporal_context"]["out_of_window_event_count"] == 0


@pytest.mark.parametrize("text,expected", [
    ("昨晚去了哪里", ["昨晚", "昨晚"]),
    ("😀去年 8月2日去了哪里", ["去年 8月2日"]),
    ("昨天加班，昨天晚上散步", ["昨天", "昨天", "晚上"]),
    ("大前天回家，前天上班", ["大前天", "前天"]),
    ("昨天的下午3点去了哪里", ["昨天", "下午3点"]),
    ("2026年10月1日到2026年10月5日做了什么", ["2026年10月1日", "2026年10月5日"]),
])
def test_offsets_refer_to_original_codepoints_after_normalisation(text, expected):
    diagnostic = query(text).diagnostics()["binding"]
    assert diagnostic["offset_unit"] == "unicode_codepoint"
    assert [text[p["start"]:p["end"]] for p in diagnostic["spans"]] == expected
    assert text not in str(diagnostic)


@pytest.mark.parametrize("text,decision,day,hour", [
    ("昨天去了哪里", "single_expression", 19, 0),
    ("昨天的下午3点去了哪里", "adjacent_day_time", 19, 15),
    ("昨晚去了哪里", "adjacent_day_time", 19, 18),
    ("昨天到今天去了哪里", "explicit_range", 19, 0),
    ("昨天到昨天去了两次", "explicit_range", 19, 0),
    ("去年8月2日去了哪里", "single_expression", 2, 0),
])
def test_clear_dates_remain_usable(text, decision, day, hour):
    q = query(text)
    assert q.allows_event_range() and q.retrieval_range()
    assert q.binding["decision"] == decision
    assert q.expressions[0].start.day == day and q.expressions[0].start.hour == hour
    assert "binding" not in q.prompt_context()


def test_repeated_expressions_are_bounded_and_do_not_silently_use_first_match():
    q = query("昨天去了，" * 9)
    assert q.reason == "expression_limit" and q.binding["spans"] == []
    assert not q.allows_event_range()
    q = query("昨天去了，" * 8)
    assert len(q.binding["spans"]) == 8
    assert not q.allows_event_range()


def test_reverse_range_and_distinct_dates_do_not_share_cache_with_simple_date():
    assert query("今天到昨天去了哪里").reason == "reversed_range"
    assert query("昨天去了哪里").cache_token() != query("本周出差，昨天去了哪里").cache_token()


def test_scheduling_callers_keep_legacy_occurrence_policy():
    from app.services.schedule_domain.time_parser import parse_time_expressions
    text = "大前天早上出发，前天早上到了，前天晚上回来"
    old = parse_time_expressions(text, now=NOW)
    read = parse_time_expressions(text, now=NOW, all_occurrences=True)
    assert [p.original_text for p in old] == ["大前天", "早上", "晚上"]
    assert [p.original_text for p in read].count("前天") == 2
    assert [p.original_text for p in read].count("早上") == 2


def test_unlocatable_parser_output_fails_closed(monkeypatch):
    from app.services.schedule_domain import time_parser
    monkeypatch.setattr(time_parser, "parse_time_expressions",
        lambda *a, **kw: [time_parser.ParsedTime("not in source", "relative", NOW, NOW, 1, False)])
    q = query("昨天去了哪里")
    assert q.reason == "source_span_unavailable" and q.binding["spans"] == []
    assert not q.allows_event_range()


def test_no_expression_is_explicitly_not_analyzed_not_missing_collection():
    assert query("你好").diagnostics()["binding"]["decision"] == "not_analyzed"
    assert query("x" * 4001).diagnostics()["binding"]["spans"] == []


def test_frozen_reply_cases_are_distinct_synthetic_scenarios():
    import json
    from pathlib import Path
    rows = json.loads((Path(__file__).parents[1] / "evals/temporal_binding/cases.json").read_text())
    assert len(rows) == len({r["id"] for r in rows}) == len({r["message"] for r in rows}) == 12
    assert len([r for r in rows if r["id"].startswith("control_")]) == 5
    for row in rows:
        assert row["criterion"] and row["memories"] and row["history"] == []
        for memory in row["memories"]:
            assert memory["source"] == "user" and memory["id"].startswith("synthetic-")
            assert datetime.fromisoformat(memory["occur_time"].replace("Z", "+00:00")) <= NOW


@pytest.mark.asyncio
@pytest.mark.parametrize("text", UNBOUND[:4])
async def test_unbound_ranges_preserve_vector_recall_and_skip_date_search(monkeypatch, text):
    from app.services.memory.retrieval import hybrid as h
    monkeypatch.setattr(h, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(h, "cache_set_retrieval", AsyncMock())
    search, vector = AsyncMock(return_value=[]), AsyncMock(return_value=[])
    monkeypatch.setattr(h, "search_by_time_range", search)
    monkeypatch.setattr(h, "search_similar_tiers", vector)
    monkeypatch.setattr(h, "search_related_memories_for_query", AsyncMock(return_value=[]))
    await h.hybrid_retrieve(text, "u1", workspace_id="w1", temporal_query=query(text))
    search.assert_not_awaited()
    vector.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["legacy", "langgraph"])
async def test_main_executors_share_original_message_binding(monkeypatch, executor):
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from app.services.chat.intent_dispatcher import IntentType
    from tests.g02_harness_support import configure_pair
    io = configure_pair(monkeypatch)
    io.now = NOW
    monkeypatch.setattr(settings, "chat_executor", executor)
    monkeypatch.setattr(chat, "_generate_reply", AsyncMock(return_value=(["是两件不同的事"], "", False, None)))
    events = [e async for e in chat.stream_chat_response(
        "c-1", UNBOUND[0], io.agent, "u-1", forced_intent=IntentType.NONE,
        reply_context={"received_at": NOW.isoformat()})]
    args = chat.fetch_parallel_context.await_args.kwargs
    assert args["temporal_query"].binding["decision"] == "separate_expressions"
    assert args["parsed_times"] == []
    assert sum(e["event"] == "done" for e in events) == 1
