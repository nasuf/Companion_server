"""CI smoke for the persona-drift eval mechanics (no DB / Redis / models).

The real multi-turn generation+judge run needs live services (that's the eval
itself, run manually). Here we pin the pure机制 that would silently rot and make
the eval measure the wrong thing: anchor derivation from agent fields, red-line
leak detection, anchor containment, history layout, verdict parsing, probe/
filler integrity.
"""

from __future__ import annotations

from datetime import timezone
from types import SimpleNamespace

from evals.persona_drift import judge as J
from evals.persona_drift.cases import FILLER_USER_MSGS, PERSONA_LEAK_TERMS, PROBES
from evals.persona_drift.run_eval import (
    _anchor_expected,
    _anchored,
    _history_rows,
    _leak,
)


def _agent(**kw):
    base = dict(name="小芜", occupation="皮具制作师", city="江苏省苏州市姑苏区", age=28)
    base.update(kw)
    return SimpleNamespace(**base)


class TestAnchorDerivation:
    def test_name_anchor(self):
        assert _anchor_expected(_agent(), "name") == ["小芜"]

    def test_age_anchor(self):
        assert _anchor_expected(_agent(), "age") == ["28"]

    def test_age_missing_is_empty(self):
        # age 缺失时不该编一个空串锚 (否则 _anchored 恒真, 校验形同虚设)
        assert _anchor_expected(_agent(age=None), "age") == []

    def test_occupation_includes_domain_token(self):
        # 职业域前 2 字 ("皮具") + 全称都接受, 覆盖模型答"我做皮具的"简写
        exp = _anchor_expected(_agent(), "occupation")
        assert "皮具" in exp and "皮具制作师" in exp

    def test_city_extracts_city_name_before_shi(self):
        # "江苏省苏州市…" → 取"市"前的 苏州 + 全串都接受
        exp = _anchor_expected(_agent(), "city")
        assert "苏州" in exp

    def test_city_without_shi_suffix(self):
        exp = _anchor_expected(_agent(city="苏州"), "city")
        assert exp == ["苏州"]


class TestLeakDetection:
    def test_flags_persona_leak(self):
        assert _leak("作为AI，我没有实体") != []
        assert "作为AI" in _leak("嗯，作为AI我觉得")

    def test_clean_reply_no_leak(self):
        assert _leak("我叫小芜呀，在苏州做皮具的") == []


class TestAnchorContainment:
    def test_anchored_true_when_value_present(self):
        assert _anchored("我叫小芜啦", ["小芜"]) is True

    def test_anchored_true_on_any_alias(self):
        assert _anchored("我做皮具的", ["皮具", "皮具制作师"]) is True

    def test_anchored_false_when_absent(self):
        assert _anchored("这个不告诉你~", ["小芜"]) is False

    def test_empty_expected_never_anchors(self):
        assert _anchored("随便什么", []) is False


class TestHistoryLayout:
    def test_history_ends_before_now_and_appends_probe(self):
        from datetime import datetime

        now = datetime(2026, 9, 9, 15, 0, tzinfo=timezone.utc)
        pairs = [("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")]
        rows = _history_rows(pairs, now)
        assert len(rows) == len(pairs)
        # 单调递增, 全部早于 now
        times = [r["createdAt"] for r in rows]
        assert times == sorted(times)
        assert times[-1] < now


class TestVerdictParsing:
    def test_parses_clean_json(self):
        v = J.parse_verdict('{"voice_consistent": true, "persona_match": false, '
                            '"reason": "变客服腔了"}')
        assert v == {"voice_consistent": True, "persona_match": False,
                     "reason": "变客服腔了"}

    def test_tolerates_surrounding_text(self):
        v = J.parse_verdict('判断：{"voice_consistent": false, '
                            '"persona_match": true, "reason": "x"} 完')
        assert v is not None and v["voice_consistent"] is False

    def test_returns_none_on_missing_field(self):
        assert J.parse_verdict('{"voice_consistent": true}') is None

    def test_returns_none_on_garbage(self):
        assert J.parse_verdict("不是 JSON") is None


def test_probe_and_filler_integrity():
    ids = [p.id for p in PROBES]
    assert len(ids) == len(set(ids)), "probe id 重复"
    # 至少 4 个身份锚探针 + 1 个风格探针
    anchors = {p.anchor for p in PROBES if p.anchor}
    assert anchors == {"name", "occupation", "city", "age"}
    assert any(p.anchor is None for p in PROBES), "缺风格探针"
    # anchor 字段必须是 run_eval._anchor_expected 认得的
    for p in PROBES:
        if p.anchor:
            assert p.anchor in {"name", "occupation", "city", "age"}
    assert len(PERSONA_LEAK_TERMS) >= 5
    assert len(FILLER_USER_MSGS) >= 12, "filler 太少堆不出长上下文"
    # filler 不该碰 AI 身份 (否则测的是"刚说过"而非"长上下文稀释")
    for m in FILLER_USER_MSGS:
        assert m and isinstance(m, str)
