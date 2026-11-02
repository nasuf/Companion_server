"""TIME07: 问现在 / 过去某段 / 将来计划, 各自按对的口径找答案。

场景: 九月说"我在A公司上班", 十月说"我跳槽到B公司了", 又说"下个月我要去C公司"。
"我现在在哪上班" 要答 B; "上个月呢 / 九月份我在哪上班" 要按当时的有效期答;
"下个月呢" 只能说成计划。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.services.memory.interaction.change_semantics import resolve_change_semantics
from app.services.memory.retrieval.context_selector import ClassifiedMemory, format_memory_time
from app.services.memory.retrieval.ranking import _occur_recency_factor
from app.services.time_domain.change_constraints import historical_change_filter
from app.services.time_domain.query import resolve_temporal_query

_TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 2, 10, 0, tzinfo=_TZ)


def _q(text: str, context: str | None = None):
    return resolve_temporal_query(text, anchor=NOW, anchor_source="persisted_message",
                                  context_text=context, bare_months=True)


@pytest.mark.parametrize("text, context, operation, start", [
    ("九月份我在哪上班", None, "as_of", "2026-09-01"),
    ("上个月我住哪", None, "as_of", "2026-10-01"),
    ("2025年9月我在哪个公司", None, "as_of", "2025-09-01"),
    ("那上个月呢", "我现在在哪上班？", "as_of", "2026-10-01"),
    ("九月份呢？", "我现在住哪？", "as_of", "2026-09-01"),
    ("我去年九月在哪上班", None, "as_of", "2025-09-01"),
    ("我昨天在哪上班", None, "as_of", "2026-11-01"),
    ("下个月呢", "我现在在哪上班", "planned", "2026-12-01"),
    ("十二月我打算去哪", None, "planned", "2026-12-01"),
    # 过去某天发生了什么仍按事件日期找
    ("你上个月去了哪", None, "range", "2026-10-01"),
    ("那上个月呢", "你今天中午吃了什么", "range", "2026-10-01"),
    # 上一句有谓语但这句不是省略追问: 不借
    ("上个月我买了一双鞋，好看吗", "我现在在哪上班", "range", "2026-10-01"),
])
def test_question_scope_routing(text, context, operation, start):
    q = _q(text, context)
    assert q.status == "resolved"
    assert q.operation.type == operation
    assert q.expressions[0].start.astimezone(_TZ).date().isoformat() == start


@pytest.mark.parametrize("text, context", [
    # 陈述句, 带"男朋友/对象"也不是在问状态
    ("今天跟男朋友吵架了好难过", None),
    ("上周我对象送了我一束花", None),
    # 问 AI 自己的, 照旧按人设回答
    ("你上个月在哪上班", None),
    # 不是省略追问: 不借上一句的谓语
    ("你今天累吗", "我现在在哪上班？"),
    ("上个月我买了一双鞋，好看吗", "我现在在哪上班？"),
    # 包含今天的时段不算"过去"
    ("今天我在哪上班", None),
])
def test_not_inferred_as_past_state(text, context):
    q = _q(text, context)
    assert q.operation.type != "as_of"


@pytest.mark.parametrize("text", ["五月天呢", "三月三有什么习俗", "农历八月十五", "6月18大促买了啥",
                                  "一年十二月都在忙", "这一月好累", "二十一月"])
def test_bare_month_false_positives(text):
    assert _q(text, "我在哪上班？").expressions == ()


def test_bare_months_are_read_path_only():
    plain = resolve_temporal_query("九月份我在哪上班", anchor=NOW, anchor_source="persisted_message")
    assert plain.expressions == ()
    from app.services.time_domain.parser import parse_time_expressions
    assert parse_time_expressions("提醒我12月续费会员", now=NOW) == []


def test_inferred_flag_reaches_diagnostics():
    assert _q("九月份我在哪上班").diagnostics()["operation"]["inferred"] is True
    assert "inferred" not in _q("2026年10月1日当时我住在哪里？").diagnostics()["operation"]


def test_current_and_unparsed_questions():
    assert _q("我现在在哪上班").operation.type == "current"
    # 不带年份的"月+日"仍按原设计视为年份不明
    assert _q("8月9日我在哪上班").reason == "year_unspecified"
    assert _q("三个月了").expressions == ()


def test_future_bare_month_without_plan_words_means_last_year():
    # "安排" "打算" 才算说今年的计划; "会员" "下午" 里的单字不算
    q = _q("十二月份我在哪上班")
    assert q.operation.type == "as_of"
    assert q.expressions[0].start.astimezone(_TZ).date().isoformat() == "2025-12-01"


# ── 计划不能当成最近发生的事 ───────────────────────────────────────

def test_future_plan_gets_no_recency_boost_and_is_labelled():
    future = (datetime.now(timezone.utc) + timedelta(days=20)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    assert _occur_recency_factor({"occur_time": future}) is None
    assert _occur_recency_factor({"occur_time": past}) > 0.9
    plan = ClassifiedMemory("下个月要去C公司上班", "strong", .8, main_category="生活", occur_time=future)
    assert "计划/未来" in format_memory_time(plan) and "尚未发生" in format_memory_time(plan)
    done = ClassifiedMemory("去面试了", "strong", .8, main_category="生活", occur_time=past)
    assert format_memory_time(done).startswith("[事件时间 ")


# ── 同一单值身份属性的旧说法 ─────────────────────────────────────

def test_older_singleton_identity_values_are_marked():
    from app.services.chat.prompt_builder import _older_singleton_ids

    old = ClassifiedMemory("我住在苏州", "strong", .8, id="m-old", main_category="身份", sub_category="现居地",
                           statement_time="2026-09-10T02:00:00Z")
    new = ClassifiedMemory("我搬到杭州了", "strong", .8, id="m-new", main_category="身份", sub_category="现居地",
                           statement_time="2026-10-20T02:00:00Z")
    same = ClassifiedMemory("我住在杭州", "strong", .8, id="m-same", main_category="身份", sub_category="现居地",
                            statement_time="2026-10-21T02:00:00Z")
    relative = ClassifiedMemory("我妹妹叫小美", "strong", .8, id="m-r", main_category="身份", sub_category="亲属",
                                statement_time="2026-09-01T02:00:00Z")
    agent = ClassifiedMemory("我住在平江路", "strong", .8, id="m-a", main_category="身份", sub_category="现居地",
                             statement_time="2026-01-01T02:00:00Z", source="ai")
    assert _older_singleton_ids([old, new, relative, agent]) == {"m-old"}
    assert _older_singleton_ids([old, new, same]) == {"m-old", "m-new"}
    assert _older_singleton_ids([new]) == set()
    # 不会变的属性 (出生地) 说法不同只是措辞; 没有说话时间的不参与比较
    born1 = ClassifiedMemory("我出生在苏州", "strong", .8, id="b1", main_category="身份", sub_category="出生地",
                             statement_time="2026-09-10T02:00:00Z")
    born2 = ClassifiedMemory("我是在苏州出生的", "strong", .8, id="b2", main_category="身份", sub_category="出生地",
                             statement_time="2026-10-10T02:00:00Z")
    merged = ClassifiedMemory("我住在上海", "strong", .8, id="mg", main_category="身份", sub_category="现居地",
                              created_at="2026-11-01T02:00:00Z")
    assert _older_singleton_ids([born1, born2]) == set()
    assert _older_singleton_ids([old, merged]) == set()


# ── 变化记录 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["我跳槽到B公司了", "我已经换到B公司上班了", "我上周辞职了", "我调到杭州分部了"])
def test_job_changes_are_transitions(text):
    row = {"id": "s-1", "role": "user", "content": text, "created_at": "2026-10-20T02:00:00Z"}
    assert resolve_change_semantics("变化", [row]).kind == "transition"


def _period(text: str):
    return _q(text).expressions[0]


def _unbounded_change(**extra):
    return {"id": "c-1", "old_memory_id": "old", "new_memory_id": "new", "kind": "transition",
            "source_available": True, "old_statement_at": datetime(2026, 9, 10, 2, tzinfo=timezone.utc),
            "semantics": {"statement_at": "2026-10-20T10:00:00+08:00", "boundary": {}}, **extra}


def test_new_fact_with_unknown_boundary_cannot_answer_before_old_statement():
    before_old = _period("2026年8月我在哪上班")
    after_old = _period("九月份我在哪上班")
    after_new = _period("2026年11月我在哪上班")
    change = _unbounded_change()
    assert historical_change_filter("new", before_old, [change])[:2] == (False, "new_fact_before_old_statement")
    # 九月里说过旧公司, 之后哪天换的不知道: 新事实不能排除, 交给回答说明不确定
    assert historical_change_filter("new", after_old, [change])[0] is True
    assert historical_change_filter("old", after_new, [change])[1] == "old_fact_after_confirmed_change_observation"
    legacy = _unbounded_change(old_statement_at=None)
    assert historical_change_filter("new", before_old, [legacy])[0] is True


# ── 交给模型的问法说明 ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_prompt_states_question_scope(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services.prompting import store
    from app.services.time_domain import context as ctx_mod

    monkeypatch.setattr(store, "get_prompt_text", AsyncMock(return_value="{query_context}"))
    clock = ctx_mod.TurnClock(now=NOW, last_user_at=None, last_dialogue_at=None, last_outbound_at=None,
                              inspected_messages=0, unknown_timestamps=0, statement_at=NOW, statement_end_at=NOW)
    for text, scope in (("下个月呢", "将来的计划"), ("九月份我在哪上班", "过去那段时间"),
                        ("我现在在哪上班", "现在的状态")):
        query = _q(text, "我现在在哪上班")
        rendered = await ctx_mod.render_temporal_context(clock, "允许", query)
        assert scope in rendered
    rendered = await ctx_mod.render_temporal_context(clock, "允许", _q("你上个月去了哪"))
    assert "question_scope" not in rendered
