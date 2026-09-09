"""CI smoke for the durative-state eval mechanics (no DB / Redis / models).

The real generation+judge run needs live services (that's the eval itself,
run manually). Here we pin the pure机制 that would silently rot: history layout
(state statement at now-stated_hours_ago, current message at now), time-context
format, deterministic red-line detection, verdict parsing with the ground-truth
window word, and case-bank integrity.
"""

from __future__ import annotations

from datetime import timezone

from evals.durative_state import judge as J
from evals.durative_state.cases import CASES, KINDS, DurativeCase
from evals.durative_state.run_eval import (
    _det_violations,
    _history_rows,
    _now_local,
    _time_context,
)


class TestHistoryLayout:
    def test_state_stated_before_now_message_at_now(self):
        case = DurativeCase(
            "t", "active", "我明天去出差三天", "路上小心",
            duration_hint="三天", stated_hours_ago=24, now_hour=15,
            message="累死了")
        now = _now_local(15)
        rows = _history_rows(case, now)
        assert [r["role"] for r in rows] == ["user", "assistant", "user"]
        assert rows[0]["content"] == "我明天去出差三天"
        assert rows[-1]["content"] == "累死了"
        assert rows[-1]["createdAt"] == now.astimezone(timezone.utc)
        # 状态陈述落在 now - stated_hours_ago (±ack 的 3 分钟内)
        delta_h = (rows[-1]["createdAt"] - rows[0]["createdAt"]).total_seconds() / 3600
        assert abs(delta_h - case.stated_hours_ago) < 0.1

    def test_expired_state_is_further_back(self):
        case = DurativeCase(
            "t", "expired", "我去出差三天", "玩得开心",
            duration_hint="三天, 8天前", stated_hours_ago=8 * 24,
            message="想看剧")
        rows = _history_rows(case, _now_local(15))
        delta_h = (rows[-1]["createdAt"] - rows[0]["createdAt"]).total_seconds() / 3600
        assert abs(delta_h - 8 * 24) < 0.1


class TestTimeContext:
    def test_format_matches_production_shape(self):
        tc = _time_context(_now_local(2))
        assert "当前时间：2026年09月09日 2时" in tc
        assert "星期" in tc


class TestDeterministicRedLines:
    def test_flags_forbidden_direction_words(self):
        case = DurativeCase(
            "t", "expired", "去出差三天", "开心",
            duration_hint="x", stated_hours_ago=8 * 24, message="想看剧",
            must_not_contain=("还在出差", "出差顺利吗"))
        assert _det_violations(case, "出差顺利吗？") == ["出差顺利吗"]
        assert _det_violations(case, "看剧好呀，想看啥类型") == []


class TestVerdictParsing:
    def test_parses_clean_json(self):
        v = J.parse_verdict('{"state_ok": true, "mentioned": false, "reason": "没提"}')
        assert v == {"state_ok": True, "mentioned": False, "reason": "没提"}

    def test_tolerates_surrounding_text(self):
        v = J.parse_verdict('判断:{"state_ok": false, "mentioned": true, "reason": "x"} 完')
        assert v is not None and v["state_ok"] is False and v["mentioned"] is True

    def test_returns_none_on_missing_field(self):
        assert J.parse_verdict('{"state_ok": true}') is None

    def test_returns_none_on_garbage(self):
        assert J.parse_verdict("不是 JSON") is None


class TestJudgePrompt:
    def test_active_and_expired_use_different_window_word(self):
        common = dict(state_line="去出差三天", duration_hint="三天",
                      message="累", reply="嗯")
        active = J.build_prompt(kind="active", **common)
        expired = J.build_prompt(kind="expired", **common)
        # ground truth 窗口词必须随 kind 切换, 否则 judge 拿错标准判
        assert "进行中" in active
        assert "已结束" in expired


def test_case_bank_integrity():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "case id 重复"
    assert {c.kind for c in CASES} == set(KINDS), "有 kind 没用例或用例串了 kind"
    for c in CASES:
        assert c.kind in KINDS
        assert c.stated_hours_ago > 0
        assert 0 <= c.now_hour <= 23
        assert c.state_line and c.message
        for w in c.must_not_contain:
            assert w and isinstance(w, str)
    # 两侧都要有足够用例, 否则分 kind 的 state_ok 没有统计意义
    assert sum(c.kind == "active" for c in CASES) >= 3
    assert sum(c.kind == "expired" for c in CASES) >= 3
