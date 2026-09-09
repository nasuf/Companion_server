"""CI smoke for the temporal-awareness eval mechanics (no DB / Redis / models).

The real generation+judge run needs live services (that's the eval itself,
run manually). Here we pin the pure机制: gap layout, time-context format,
deterministic red-line detection, verdict parsing, case-bank integrity —
the parts that would silently rot and make the eval measure the wrong thing.
"""

from __future__ import annotations

from datetime import timezone

import pytest

from evals.temporal_awareness import judge as J
from evals.temporal_awareness.cases import CASES, GROUPS, TemporalCase
from evals.temporal_awareness.run_eval import (
    _det_violations,
    _history_rows,
    _now_local,
    _time_context,
)


class TestGapLayout:
    def test_current_message_is_at_now_history_ends_gap_before(self):
        # 关键机制: 历史最后一条落在 now-gap, 当前消息在 now —— 重逢感知 gap 与
        # [MM-DD HH:MM] 前缀都靠这个布局才对。
        case = TemporalCase("t", "reunion", "在吗", gap_seconds=2 * 86400,
                            now_hour=15, history=(("user", "a"), ("assistant", "b")))
        now = _now_local(15)
        rows = _history_rows(case, now)
        assert rows[-1]["content"] == "在吗"
        assert rows[-1]["createdAt"] == now.astimezone(timezone.utc)
        gap = (rows[-1]["createdAt"] - rows[-2]["createdAt"]).total_seconds()
        # 最后一条历史 (assistant "b") 到当前消息, 间隔≈gap_seconds
        assert abs(gap - case.gap_seconds) < 1

    def test_no_history_still_has_current_message(self):
        case = TemporalCase("t", "time_of_day", "睡不着", gap_seconds=3600, history=())
        rows = _history_rows(case, _now_local(2))
        assert len(rows) == 1 and rows[0]["content"] == "睡不着"


class TestTimeContext:
    def test_format_matches_production_shape(self):
        # 复刻 build_time_context 的格式 (小时精度 + 星期), 否则模型看到的时刻串
        # 跟线上不一样, 评测就测了个线上不存在的输入。
        tc = _time_context(_now_local(2))
        assert "当前时间：2026年09月09日 2时" in tc
        assert "星期" in tc

    def test_hour_is_controlled(self):
        assert "23时" in _time_context(_now_local(23))
        assert "0时" in _time_context(_now_local(0))


class TestDeterministicRedLines:
    def test_flags_forbidden_words(self):
        case = TemporalCase("t", "time_of_day", "睡不着", gap_seconds=3600,
                            now_hour=2, must_not_contain=("早上好", "上午"))
        assert _det_violations(case, "早上好呀，怎么还没睡") == ["早上好"]
        assert _det_violations(case, "这么晚还没睡呀") == []

    def test_no_forbidden_words_means_no_violation(self):
        case = TemporalCase("t", "reunion", "在吗", gap_seconds=300,
                            must_not_contain=("好久不见",))
        assert _det_violations(case, "嗯我在的，刚看到") == []


class TestVerdictParsing:
    def test_parses_clean_json(self):
        v = J.parse_verdict('{"gap_ok": true, "hallucination": false, '
                            '"tod_ok": true, "stale_topic": false, "reason": "ok"}')
        assert v == {"gap_ok": True, "hallucination": False, "tod_ok": True,
                     "stale_topic": False, "reason": "ok"}

    def test_tolerates_surrounding_text(self):
        v = J.parse_verdict('好的，判断如下：{"gap_ok": false, "hallucination": true, '
                            '"tod_ok": true, "stale_topic": false, "reason": "x"} 完毕')
        assert v is not None and v["gap_ok"] is False and v["hallucination"] is True

    def test_returns_none_on_missing_field(self):
        assert J.parse_verdict('{"gap_ok": true}') is None

    def test_returns_none_on_garbage(self):
        assert J.parse_verdict("完全不是 JSON") is None

    @pytest.mark.parametrize("sec,expect", [
        (300, "分钟"), (5 * 3600, "小时"), (2 * 86400, "天"),
    ])
    def test_gap_text_granularity(self, sec, expect):
        assert expect in J.gap_text(sec)


def test_case_bank_integrity():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "case id 重复"
    assert {c.group for c in CASES} == set(GROUPS), "有分组没有用例或用例串组"
    for c in CASES:
        assert c.gap_seconds > 0
        assert 0 <= c.now_hour <= 23
        # 带确定性红线的用例, 红线词非空
        for w in c.must_not_contain:
            assert w and isinstance(w, str)
