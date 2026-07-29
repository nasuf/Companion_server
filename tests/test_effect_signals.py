"""回复效果指标.

这类指标算错了没有任何症状 —— 它只是给出一个看着合理的数, 然后被拿去做决策。所以
钉住的重点不是"能跑", 而是那些错了也不报错的地方: 日界、碎片消息、样本量门槛、
未定型的回访率。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.services.effect import signals, store
from app.services.effect.signals import (
    CONTINUATION_WINDOW,
    MIN_SLICE_TURNS,
    SESSION_GAP,
    EffectMetrics,
    SliceMetric,
    _day_bounds,
)
from app.services.effect.store import merge_slices, summarise, wilson_interval


class TestDayBounds:
    def test_local_day_maps_to_naive_utc(self):
        """库里是 naive UTC。带时区的边界会静默偏 8 小时 —— 作息落库那个 bug 的同款."""
        start, end = _day_bounds(date(2026, 7, 29))
        assert start == datetime(2026, 7, 28, 16, 0)
        assert end == datetime(2026, 7, 29, 16, 0)
        assert start.tzinfo is None and end.tzinfo is None

    def test_window_is_exactly_one_day(self):
        start, end = _day_bounds(date(2026, 7, 29))
        assert end - start == timedelta(days=1)

    def test_consecutive_days_do_not_overlap_or_gap(self):
        _, end_a = _day_bounds(date(2026, 7, 28))
        start_b, _ = _day_bounds(date(2026, 7, 29))
        assert end_a == start_b


class TestThresholds:
    def test_continuation_window_matches_the_measured_curve(self):
        """5 分钟不是随手定的: 实测 2 分钟 77% / 5 分钟 82% / 3 小时 89%,

        曲线在 2 分钟后走平。窗口再开大只会逼近饱和, 失去分辨力。
        """
        assert CONTINUATION_WINDOW == timedelta(minutes=5)

    def test_session_gap_matches_topic_reset(self):
        """会话边界要跟话题重置/重逢感知同线, 否则三处对"一段对话"的定义不一致."""
        from app.services.topic import TOPIC_RESET_GAP_SECONDS

        assert SESSION_GAP.total_seconds() == TOPIC_RESET_GAP_SECONDS


class TestRates:
    def test_continuation_rate_is_none_without_turns(self):
        """没有回合时不该报 0% —— 那会被读成"一个都没接住"."""
        assert EffectMetrics(date="2026-07-29").continuation_rate is None

    def test_proactive_rate_is_none_without_sends(self):
        assert EffectMetrics(date="2026-07-29").proactive_response_rate is None

    def test_return_rate_is_none_until_the_next_day_closes(self):
        m = EffectMetrics(date="2026-07-29", active_users=10, returned_next_day=None)
        assert m.next_day_return_rate is None

    def test_rates_computed_when_data_present(self):
        m = EffectMetrics(
            date="2026-07-28", turns=100, continued=79,
            proactive_sent=20, proactive_answered=4,
            active_users=10, returned_next_day=4,
        )
        assert m.continuation_rate == 0.79
        assert m.proactive_response_rate == 0.2
        assert m.next_day_return_rate == 0.4


class TestSliceSampleGuard:
    def test_small_slice_reports_no_rate(self):
        """3 个回合里 2 个延续是 67%, 跟 300 里 200 个的 67% 不是一回事."""
        s = SliceMetric(dimension="reply_path", value="short_circuit", turns=3, continued=2)
        assert s.continuation_rate is None
        assert s.as_dict()["sufficient_sample"] is False

    def test_large_enough_slice_reports_a_rate(self):
        s = SliceMetric(
            dimension="reply_path", value="main_llm",
            turns=MIN_SLICE_TURNS, continued=MIN_SLICE_TURNS // 2,
        )
        assert s.continuation_rate == 0.5


class TestMergeSlices:
    def test_accumulates_across_days(self):
        """单日切完每格只有个位数, 必须累计才有可比性."""
        days = [
            {"slices": [
                {"dimension": "memory_relevance", "value": "weak", "turns": 15, "continued": 12},
            ]},
            {"slices": [
                {"dimension": "memory_relevance", "value": "weak", "turns": 15, "continued": 11},
            ]},
        ]
        merged = merge_slices(days)
        assert len(merged) == 1
        assert merged[0]["turns"] == 30 and merged[0]["continued"] == 23

    def test_merged_slice_carries_a_confidence_interval(self):
        """只给比率不给区间, 等于邀请人把噪声读成差异."""
        days = [{"slices": [
            {"dimension": "d", "value": "v", "turns": 100, "continued": 79},
        ]}]
        m = merge_slices(days)[0]
        assert m["ci_low"] is not None and m["ci_high"] is not None
        assert m["ci_low"] < m["continuation_rate"] < m["ci_high"]

    def test_insufficient_sample_gets_no_interval_either(self):
        days = [{"slices": [{"dimension": "d", "value": "v", "turns": 5, "continued": 4}]}]
        m = merge_slices(days)[0]
        assert m["continuation_rate"] is None and m["ci_low"] is None

    def test_empty_input_is_safe(self):
        assert merge_slices([]) == []
        assert merge_slices([{"slices": None}]) == []


class TestWilson:
    def test_interval_brackets_the_estimate(self):
        lo, hi = wilson_interval(79, 100)
        assert lo < 0.79 < hi

    def test_small_sample_gives_a_wide_interval(self):
        narrow = wilson_interval(790, 1000)
        wide = wilson_interval(8, 10)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0]) * 3

    def test_never_escapes_zero_to_one(self):
        """正态近似在贴边时会给出 102% 这种数, Wilson 不会."""
        for s, t in ((0, 5), (5, 5), (1, 3), (99, 100)):
            lo, hi = wilson_interval(s, t)
            assert 0.0 <= lo <= hi <= 1.0

    def test_zero_total_degrades_to_full_range(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestSummarise:
    def test_rates_are_volume_weighted_not_day_averaged(self):
        """按天平均会让只有 3 个回合的冷清日跟 300 个回合的日子等权."""
        days = [
            {"turns": 3, "continued": 3},        # 冷清的一天, 100%
            {"turns": 300, "continued": 150},    # 正常的一天, 50%
        ]
        s = summarise(days)
        assert s["turns"] == 303
        # 量权 = 153/303 ≈ 0.505; 天平均会得到 0.75
        assert 0.50 <= s["continuation_rate"] <= 0.51

    def test_retention_only_counts_settled_days(self):
        """未定型的日子 returned_next_day 是 None, 计入会把回访率拉低."""
        days = [
            {"active_users": 10, "returned_next_day": 4},
            {"active_users": 10, "returned_next_day": None},   # 今天, 次日还没到
        ]
        s = summarise(days)
        assert s["next_day_return_rate"] == 0.4
        assert s["settled_days"] == 1

    def test_empty_range_returns_nones_not_zeros(self):
        s = summarise([])
        assert s["continuation_rate"] is None
        assert s["proactive_response_rate"] is None
        assert s["median_gap_s"] is None


class TestSettlement:
    def test_today_is_never_settled(self):
        now = datetime(2026, 7, 29, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        assert not store._is_settled(date(2026, 7, 29), now)

    def test_yesterday_is_not_settled_either(self):
        """昨天的次日回访率要等今天过完才定型, 提前缓存会固化一个偏低的值."""
        now = datetime(2026, 7, 29, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        assert not store._is_settled(date(2026, 7, 28), now)

    def test_two_days_ago_is_settled(self):
        now = datetime(2026, 7, 29, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        assert store._is_settled(date(2026, 7, 27), now)


class TestQueryShape:
    def test_turn_cte_excludes_fragments_replying_before_the_reply(self):
        """用户连发多条 (碎片聚合) 时, 后一条不能算作"被回复接住"."""
        sql = signals._TURNS_CTE
        assert "next_ask_at > t.replied_at" in sql, (
            "缺少这个条件的话, 碎片消息会把延续率虚高 —— 用户只是话没说完, "
            "不是被回复打动了"
        )

    def test_turn_cte_drops_turns_without_a_reply(self):
        assert "WHERE t.replied_at IS NOT NULL" in signals._TURNS_CTE

    def test_lead_looks_past_the_day_boundary(self):
        """每天最后一个回合不能因为"下一条在次日"就被判成没接住.

        LEAD 只在当日窗口内取值时, 那个回合的 next_ask 必为 NULL —— 用户过两分钟
        就回了、只是跨过了午夜, 也会被算成流失。偏差方向是系统性向下的, 而且不会
        报任何错。
        """
        sql = signals._TURNS_CTE
        assert "$2::timestamp + $4::interval" in sql, "LEAD 的取数窗口没有越过日界"
        assert "in_window" in sql, "越界多取的行必须再过滤掉, 否则次日回合会被计入"

    def test_lookahead_covers_reply_time_plus_window(self):
        """往后看的时长要能容下「回复耗时 + 延续窗口」, 否则边界修了个寂寞."""
        assert signals._BOUNDARY_LOOKAHEAD > CONTINUATION_WINDOW

    def test_slice_query_drops_null_values(self):
        """键存在但值为 null 时会切出一个叫 "None" 的格子, 它不携带分组信息."""
        import inspect

        src = inspect.getsource(signals._fetch_slices)
        assert "IS NOT NULL" in src, "空值切片会在界面上变成一个名叫 None 的分组"

    @pytest.mark.parametrize("dim", signals.SLICE_DIMENSIONS)
    def test_slice_dimensions_are_low_cardinality(self, dim):
        """高基数字段切出来每格样本太少, 比率剧烈抖动反而误导."""
        assert dim in {
            "reply_path", "memory_relevance", "needs_web_search", "reply_emotion_source",
        }
