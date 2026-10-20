"""TIME02B: operation evidence obligations and multi-period parsing regressions."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.temporal_query import resolve_temporal_query


def query(text, anchor="2026-10-20T01:00:00Z", zone="Asia/Shanghai"):
    return resolve_temporal_query(text, anchor=anchor, anchor_source="persisted_message",
                                  timezone_name=zone)


@pytest.mark.parametrize("text,operation,requirements", [
    ("上周去过几次图书馆", "count", ["deduplicated_events", "complete_scope"]),
    ("总共骑车多少回？", "count", ["deduplicated_events", "complete_scope"]),
    ("从开始练琴到现在多久了", "duration", ["confirmed_endpoints", "duration_unit"]),
    ("手术后第几天了", "duration", ["confirmed_endpoints", "duration_unit"]),
    ("上一次去是什么时候", "order", ["ordered_events", "complete_scope"]),
    ("倒数第二次去哪里", "order", ["ordered_events", "complete_scope"]),
    ("两趟旅行哪个先？", "comparison", ["comparable_evidence"]),
    ("我现在还住在那里吗", "current", ["current_validity"]),
    ("那时候我在哪里工作", "as_of", ["historical_validity"]),
    ("明天有什么计划", "planned", ["plan_status"]),
    ("去过几次，上次是什么时候", "multiple", ["separate_queries"]),
    ("昨天说了什么", "range", ["event_dates"]),
    ("谢谢你", "unspecified", []),
    ("我第一次去图书馆", "unspecified", []),
    ("过去过好几次了", "count", ["deduplicated_events", "complete_scope"]),
])
def test_explicit_operations_do_not_claim_calculation(text, operation, requirements):
    q = query(text)
    d = q.diagnostics()["operation"]
    assert d == {"version": 1, "type": operation, "requirements": requirements,
                 "execution": "not_requested" if operation == "unspecified" else "not_executed"}
    assert text not in str(d)


@pytest.mark.parametrize("expression,start,end", [
    ("昨天到今天", "2026-10-19", "2026-10-20"),
    ("从上周至本周", "2026-10-12", "2026-10-25"),
    ("2024年2月28日～2024年3月1日", "2024-02-28", "2024-03-01"),
    ("去年到今年", "2025-01-01", "2026-12-31"),
])
def test_adjacent_range_endpoints_define_union_not_intersection(expression, start, end):
    q = query(expression)
    assert q.status == "resolved"
    a, b = q.retrieval_range()
    assert a.date().isoformat() == start and b.date().isoformat() == end


@pytest.mark.parametrize("expression", ["去年和上个月", "本周或今天", "今天以及今年", "上周、昨天"])
def test_distinct_overlapping_periods_never_collapse(expression):
    q = query(expression, anchor="2026-01-20T01:00:00Z")
    assert q.reason == "multiple_periods" and q.retrieval_range() is None


@pytest.mark.parametrize("text", ["明天到昨天", "2026年10月20日至2026年10月19日"])
def test_reversed_ranges_have_no_retrieval_window(text):
    q = query(text)
    assert q.reason == "reversed_range" and q.retrieval_range() is None


@pytest.mark.parametrize("word,day,hour", [
    ("昨晚", 19, 18), ("昨夜", 19, 18), ("今早", 20, 6), ("今晨", 20, 6),
    ("今晚", 20, 18), ("明早", 21, 6), ("明晚", 21, 18),
])
def test_colloquial_day_periods_keep_statement_day(word, day, hour):
    q = query(word)
    assert q.status == "resolved"
    assert q.expressions[0].start.day == day
    assert q.expressions[0].start.hour == hour


def test_social_conjunction_does_not_split_one_period():
    q = query("昨天中午和同事吃饭")
    assert q.status == "resolved"
    assert q.expressions[0].start.day == 19


@pytest.mark.parametrize("text", ["刚刚发生什么了", "刚才在忙什么", "前不久去过", "稍后再说"])
def test_imprecise_clocks_are_not_exact_dates(text):
    assert query(text).reason == "vague_expression"


def test_holiday_cache_does_not_supply_nearest_or_wrong_lunar_year(monkeypatch):
    from app.services.schedule_domain import holiday_cache
    monkeypatch.setattr(holiday_cache, "all_known_names", lambda: ["春节"])
    q = query("去年春节做了什么")
    assert q.reason == "holiday_calendar_required" and q.retrieval_range() is None


def test_query_obligations_survive_unknown_anchor_and_do_not_depend_on_now():
    q = query("以前去过几次", anchor=None)
    assert q.reason == "missing_anchor" and q.operation.type == "count"
    assert q.cache_token() == query("以前去过几次", anchor=None).cache_token()
    assert query("本周去了几次").cache_token() != query("本周去了哪里").cache_token()
    assert query("a" * 4001).reason == "input_limit"


@pytest.mark.parametrize("text", [
    "截至昨天我在哪里工作？", "从昨天到现在多久了？",
    "昨天的状态现在还有效吗？", "昨天和上周相比怎么样？",
    "昨天去过几次，上次是什么时候？",
])
def test_state_and_computation_queries_do_not_filter_away_prior_evidence(text):
    q = query(text)
    assert q.retrieval_range() is None and q.parsed_times() == []


@pytest.mark.asyncio
async def test_actual_reply_contract_renders_obligations_without_raw_text(monkeypatch):
    from app.services.chat.temporal_context import build_turn_clock, prepare_reply_time
    from app.services.prompting.defaults import CHAT_TEMPORAL_CONTEXT_PROMPT
    from app.services.prompting import store
    monkeypatch.setattr(store, "get_prompt_text", AsyncMock(return_value=CHAT_TEMPORAL_CONTEXT_PROMPT))
    c = build_turn_clock([], now=datetime(2026, 10, 20, 1, tzinfo=timezone.utc),
                         received_at="2026-10-20T01:00:00Z")
    diagnostics = {}
    _, _, rendered = await prepare_reply_time([], exclude_ids=set(),
        current_message="本周参加秘密活动几次", diagnostics=diagnostics, clock=c)
    assert "complete_scope" in rendered and "not_executed" in rendered
    assert "秘密活动" not in rendered
    assert diagnostics["temporal_context"]["query"]["operation"]["type"] == "count"
