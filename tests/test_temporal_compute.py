"""TIME08: 次数 / 先后 / 间隔按"对方跟你提到的日子"由程序算好; 次数永远是下限。"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.services.time_domain import compute
from app.services.time_domain.compute import EventRow, compute_plan, render_plan, topic_term
from app.services.time_domain.query import resolve_temporal_query

_TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 2, 10, 0, tzinfo=_TZ)


def _at(day: str, hour: int = 12) -> datetime:
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, hour, tzinfo=_TZ).astimezone(timezone.utc)


def _row(i: int, text: str, day: str | None, hour: int = 12, *, mentions: int = 0,
         sources: tuple[str, ...] = ()) -> EventRow:
    return EventRow(f"m{i}", text, _at(day, hour) if day else None, None, mentions,
                    tuple(_at(s) for s in sources))


def _q(text: str):
    return resolve_temporal_query(text, anchor=NOW, anchor_source="persisted_message", bare_months=True)


@pytest.mark.parametrize("question, term", [
    ("这个月我去了几次健身房", "健身"),
    ("我面试过几次", "面试"),
    ("我多久没去医院了", "医院"),
    ("这周我加了几次班", "加班"),
    ("我多久没吃火锅了", "吃火锅"),
    ("我喝了几次奶茶", "奶茶"),
    ("我练了几次吉他", "吉他"),
    ("我跟她吃饭看电影几次了", ""),  # 两件事
    ("你去过几次健身房", ""),          # 问的是 AI
    ("我妈去过几次医院", ""),          # 问的是家人
    ("这个月我跟男朋友吵了几次架", ""),
    ("我们多久没聊天了", ""),
])
def test_topic_term(question, term):
    assert topic_term(question) == term


def test_count_is_a_lower_bound_of_mentioned_days():
    rows = [
        _row(1, "用户去健身了", "2026-11-01", 9),
        _row(2, "用户说今天在健身房练了背", "2026-11-01", 20),  # 同一天说两遍算一次
        _row(3, "用户去健身房", "2026-10-28"),
        _row(4, "用户去健身房", "2026-09-30"),                  # 不在本月
    ]
    plan = compute_plan("count", "这个月我去了几次健身房", rows, now=NOW, zone=_TZ,
                        period=_q("这个月我去了几次健身房").expressions[0])
    assert plan.value == {"at_least_days": 1}
    assert plan.diagnostics()["lower_bound"] is True
    text = render_plan(plan)
    assert "至少 1 次" in text and "不能说成一共正好1次" in text and "范围：2026-11-01" in text
    assert "你跟我说过的" in text


def test_merged_mentions_use_source_times_and_flag_more():
    # 10-05 和 10-20 都说了"去健身房", 录入时合并成一条 (只记了 10-05)
    merged = _row(1, "用户去健身房了", "2026-10-05", mentions=1, sources=("2026-10-05",))
    plan = compute_plan("duration", "我多久没去健身了", [merged], now=NOW, zone=_TZ)
    assert plan.value["last_date"] == "2026-10-05"
    assert "merged_mentions" in plan.reasons and "可能更多、更近" in render_plan(plan)
    linked = _row(1, "用户去健身房了", "2026-10-05", mentions=1, sources=("2026-10-05", "2026-10-20"))
    plan = compute_plan("duration", "我多久没去健身了", [linked], now=NOW, zone=_TZ)
    assert plan.value == {"since_last_days": 13, "last_date": "2026-10-20", "gap_between_last_two_days": 15}
    assert "merged_mentions" not in plan.reasons
    assert "不要断定对方这段时间一次都没去" in render_plan(plan)


@pytest.mark.parametrize("text", ["用户昨天没去健身", "用户想去健身但下雨", "用户打算明天去健身",
                                  "用户取消了今天的健身课", "用户今天没健身", "用户今天不健身",
                                  "用户没能去健身", "用户健身去不了"])
def test_non_occurrences_are_not_counted(text):
    plan = compute_plan("count", "我去过几次健身房", [_row(1, text, "2026-10-30")], now=NOW, zone=_TZ)
    assert plan.status == "no_events" and plan.skipped == 1


@pytest.mark.parametrize("text", ["用户好久没健身了今天终于去了", "用户去健身了，不想动了", "用户按计划去健身了",
                                  "用户没想到今天健身这么累"])
def test_occurrences_near_negation_words_still_count(text):
    plan = compute_plan("count", "我去过几次健身房", [_row(1, text, "2026-10-30")], now=NOW, zone=_TZ)
    assert plan.value == {"at_least_days": 1}


def test_event_day_and_mention_day_are_not_double_counted():
    # 11-01 说"昨天去健身了": 发生日 10-31, 说的那天 11-01 不再另算一天
    said = _at("2026-11-01")
    row = EventRow("m1", "用户昨天去健身了", _at("2026-10-31"), said, 0, (said,))
    plan = compute_plan("count", "我去过几次健身房", [row], now=NOW, zone=_TZ)
    assert [d.isoformat() for d in plan.mention_days] == ["2026-10-31"]
    plan = compute_plan("count", "这个月我去了几次健身房", [row], now=NOW, zone=_TZ,
                        period=_q("这个月我去了几次健身房").expressions[0])
    assert plan.status == "no_events"


def test_future_dates_are_not_counted():
    plan = compute_plan("count", "我面试过几次", [_row(1, "用户下周面试", "2026-11-09"),
                                              _row(2, "用户面试了", "2026-10-20")], now=NOW, zone=_TZ)
    assert plan.value == {"at_least_days": 1}


@pytest.mark.parametrize("question, expected", [
    ("我最近一次面试是哪天", ("last", 1, "2026-10-30")),
    ("我上次面试是哪天", ("last", 1, "2026-10-30")),
    ("我第一次面试是什么时候", ("first", 1, "2026-10-02")),
    ("我第三次面试是什么时候", ("first", 3, "2026-10-30")),
    ("倒数第二次面试是什么时候", ("last", 2, "2026-10-15")),
    ("上上次面试是哪天", ("last", 2, "2026-10-15")),
])
def test_ordinals(question, expected):
    rows = [_row(i, "用户去面试", d) for i, d in enumerate(["2026-10-30", "2026-10-02", "2026-10-15"])]
    plan = compute_plan("order", question, rows, now=NOW, zone=_TZ)
    assert (plan.value["which"], plan.value["n"], plan.value["date"]) == expected
    too_few = compute_plan("order", "我第五次面试是什么时候", rows, now=NOW, zone=_TZ)
    assert too_few.status == "too_few" and "只有 3 次" in render_plan(too_few)
    assert compute_plan("order", "我第一次面试是什么时候", rows, now=NOW, zone=_TZ,
                        truncated=True).status == "topic_unclear"


def test_local_day_boundary_uses_configured_timezone():
    # 北京时间 11-01 23:30 和 11-02 00:30 是两天
    rows = [EventRow("a", "用户跑步", datetime(2026, 11, 1, 15, 30, tzinfo=timezone.utc), None),
            EventRow("b", "用户跑步", datetime(2026, 11, 1, 16, 30, tzinfo=timezone.utc), None)]
    assert compute_plan("count", "我跑了几次步", rows, now=NOW, zone=_TZ).value == {"at_least_days": 2}


def test_no_events_never_claims_absence():
    plan = compute_plan("count", "我去过几次医院", [], now=NOW, zone=_TZ)
    assert plan.status == "no_events" and "不能说没发生过" in render_plan(plan)


@pytest.mark.asyncio
async def test_plan_for_question_routes_and_degrades(monkeypatch):
    load = AsyncMock(return_value=([_row(1, "用户去健身了", "2026-11-01")], False))
    monkeypatch.setattr(compute, "load_event_rows", load)

    async def plan(text):
        return await compute.plan_for_question(text, _q(text), user_id="u", workspace_id="w", now=NOW)

    done = await plan("这个月我去了几次健身房")
    assert done.status == "computed" and load.await_args.kwargs["term"][:3] == ("健身", "撸铁", "举铁")
    assert await plan("今天好累") is None
    assert await plan("我减肥多久了") is None            # 持续时长, 不是距上次多久
    recent = await plan("我最近一次去健身房是哪天")      # "最近一次" 不是时间范围
    assert recent.status == "computed"
    await plan("我多久没吃火锅了")
    assert load.await_args.kwargs["term"][0] == "火锅"   # 动宾事件按宾语匹配
    assert (await plan("我跟她吃饭看电影几次了")).status == "topic_unclear"
    unclear = await plan("这个月和上个月我去了几次健身房")
    assert unclear.status == "topic_unclear" and unclear.reasons == ["range_unclear"]
    assert render_plan(unclear) == ""
    monkeypatch.setattr(compute, "load_event_rows", AsyncMock(side_effect=RuntimeError("db down")))
    down = await plan("我面试过几次")
    assert down.status == "unavailable" and down.diagnostics()["model_assist"] == "not_used"


@pytest.mark.asyncio
async def test_scan_query_is_bounded_excludes_reminders_and_escapes_like():
    calls = []

    class FakeDB:
        async def query_raw(self, sql, *args):
            calls.append((sql, args))
            return [{"id": f"m{i}", "content": "x", "occur_time": None, "statement_time": None,
                     "mention_count": 0, "source_times": []} for i in range(compute.SCAN_LIMIT + 1)]
    rows, truncated = await compute.load_event_rows(user_id="u", workspace_id="w", term="100%_跑", database=FakeDB())
    assert truncated and len(rows) == compute.SCAN_LIMIT
    sql, args = calls[0]
    assert "'提醒'" in sql and "memory_evidence_links" in sql
    assert args[2] == ["%100\\%\\_跑%"] and args[3] == compute.SCAN_LIMIT + 1
    assert "ILIKE ANY" in sql


@pytest.mark.parametrize("reply, verdict", [
    ("这是你跟我说过的至少三次哦", "consistent"),
    ("跟我说过的有3次", "consistent"),
    ("上个月你去了三次健身房", "overclaimed"),
    ("你去了5次呢", "overclaimed"),
    ("就去了两次吧", "mismatch"),
    ("你去得挺勤的", "consistent"),
    ("第一次去是十月，你跟我说过的有三次", "consistent"),
    ("再去一次吧", "consistent"),
    ("至少三次吧，这周去了5次", "overclaimed"),
])
def test_reply_count_check(reply, verdict):
    diag = {"status": "computed", "operation": "count", "value": {"at_least_days": 3}}
    assert compute.check_reply(diag, reply) == verdict
    assert compute.check_reply({"status": "no_events", "operation": "count"}, reply) is None


@pytest.mark.parametrize("reply, verdict", [
    ("你上次跟我提起健身是四天前", "consistent"),
    ("10月29号那次呀", "consistent"),
    ("十月二十九日去的", "consistent"),
    ("你上次是五天前跟我说的", "mismatch"),
    ("10月28号那次", "mismatch"),
    ("好像挺久没去了", "consistent"),
])
def test_reply_date_check(reply, verdict):
    order = {"status": "computed", "operation": "order",
             "value": {"which": "last", "n": 1, "date": "2026-10-29", "days_ago": 4}}
    duration = {"status": "computed", "operation": "duration",
                "value": {"since_last_days": 4, "last_date": "2026-10-29"}}
    assert compute.check_reply(order, reply) == verdict
    assert compute.check_reply(duration, reply) == verdict


def test_synonym_mentions_count_once_per_day():
    rows = [_row(1, "用户今天去撸铁了", "2026-10-20"), _row(2, "用户去健身房练腿", "2026-10-20"),
            _row(3, "用户练背好累", "2026-10-25"), _row(4, "用户今天没撸铁", "2026-10-27")]
    plan = compute_plan("count", "我去了几次健身房", rows, now=NOW, zone=_TZ)
    assert plan.value == {"at_least_days": 2} and plan.skipped == 1
    assert compute.search_terms("健身")[0] == "健身" and "撸铁" in compute.search_terms("健身")
    assert compute.search_terms("钢琴") == ("钢琴",)
    # 只从泛称往下展开: 问得具体就只数具体的
    assert compute.search_terms("体检") == ("体检",) and compute.search_terms("夜跑") == ("夜跑",)
    assert "打球" not in compute.search_terms("羽毛球") and "口腔" not in compute.search_terms("看牙")


@pytest.mark.parametrize("reply, verdict", [
    ("10月3日前后那阵子，10月29号去的", "consistent"),   # "3日前" 不是天数
    ("国庆10月1日之后，最近一次是10月29号", "consistent"),  # 顺带提到别的日子
    ("最近两次隔了5天，上次是四天前", "consistent"),        # 隔了几天是间隔
])
def test_reply_date_check_ignores_unrelated_dates(reply, verdict):
    order = {"status": "computed", "operation": "order",
             "value": {"which": "last", "n": 1, "date": "2026-10-29", "days_ago": 4}}
    assert compute.check_reply(order, reply) == verdict


def test_thoughts_are_not_plans():
    assert not compute._not_occurred("用户想起今天去健身了", "健身")
    assert compute._not_occurred("用户想去健身", "健身")


@pytest.mark.parametrize("text", [
    "用户和朋友约了下周去看电影", "用户预约了下周看牙医", "用户说车脏了，该洗车了",
    "用户抢到了演唱会门票", "用户说他同事周三去了咖啡店", "用户给儿子报了游泳班",
    "用户想去迪士尼过圣诞",
])
def test_plans_tickets_and_other_people_are_not_occurrences(text):
    term = next(t for t in ("电影", "牙医", "洗车", "演唱会", "咖啡", "游泳", "迪士尼") if t in text)
    assert compute._not_occurred(text, term)


@pytest.mark.parametrize("text, term", [
    ("用户和同事打羽毛球", "羽毛球"), ("用户带孩子去游泳", "游泳"), ("用户打算去健身结果练完了", "健身"),
    ("用户和男朋友去看电影了", "电影"), ("她陪用户去医院看病", "看病"), ("他请用户吃了火锅", "火锅"),
    ("用户带着孩子去游泳了", "游泳"), ("下班后同事叫上用户一起吃火锅", "火锅"),
    ("用户昨天应该是去健身了", "健身"), ("用户记得去健身了", "健身"),
])
def test_doing_it_with_others_or_after_planning_still_counts(text, term):
    assert not compute._not_occurred(text, term)


def test_vague_topics_are_not_guessed():
    assert compute.topic_term("我上次那个是什么时候来着") == ""
    assert compute.topic_term("这个月我做了几次") == ""
    assert compute.topic_term("我上个月看了几场电影") == "电影"
