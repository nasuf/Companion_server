"""CI smoke for the proactive_naturalness eval mechanics (no DB / Redis / models).

The real generation+judge run needs live services (that's the eval itself,
run manually). Here we pin: case bank shape, judge prompt formation, verdict
parsing, source-kind coverage — the mechanics that would silently rot and
make the eval measure the wrong thing.
"""

from __future__ import annotations

from evals.proactive_naturalness import judge as J
from evals.proactive_naturalness.cases import CASES, SOURCE_KINDS


class TestCaseBankShape:
    def test_no_duplicate_case_ids(self):
        ids = [c.id for c in CASES]
        assert len(ids) == len(set(ids)), "duplicate case ids"

    def test_all_source_kinds_have_cases(self):
        kinds = {c.source_kind for c in CASES}
        # 三档主源 + none 兜底 都要有用例
        assert kinds >= {"user_interest_match", "ai_persona_match", "socially_hot"}

    def test_each_source_kind_has_enough_coverage(self):
        # 每档至少 2 道, 少于这个数 A/B 结论就是 under-coverage 的假象
        from collections import Counter
        counts = Counter(c.source_kind for c in CASES)
        for k in ("user_interest_match", "ai_persona_match", "socially_hot"):
            assert counts[k] >= 2, f"{k} 只有 {counts[k]} 道, 太少不出信号"

    def test_v0_baseline_junk_case_exists(self):
        # 至少一道 case 用垃圾内容 (八卦/死讯), 验证"AI 应拒绝这类内容"
        junk = [c for c in CASES if "junk" in c.id or "celeb" in c.id.lower()]
        assert len(junk) >= 1, "缺 V0 baseline 垃圾内容 case, 无法测拒绝率"

    def test_source_kinds_declared_are_valid(self):
        for c in CASES:
            assert c.source_kind in SOURCE_KINDS

    def test_should_emit_card_implies_has_candidates(self):
        # should_emit_card=True 时必须有 trending_candidates, 否则挂啥卡?
        for c in CASES:
            if c.should_emit_card:
                assert c.trending_candidates, f"{c.id} should emit card 但无 candidates"


class TestJudgePrompt:
    def _base(self):
        return dict(
            agent_persona="小菁, ENFP, 皮具制作师",
            user_portrait="25岁上海白领",
            trigger_type="silence_wakeup",
            source_kind="user_interest_match",
            trending_candidates=[{"title": "test", "snippet": "abc", "platform": "微博"}],
            message="你好",
            card=None,
        )

    def test_source_kind_appears_in_prompt(self):
        for kind in ("user_interest_match", "ai_persona_match", "socially_hot", "none"):
            args = self._base()
            args["source_kind"] = kind
            p = J.build_prompt(**args)
            assert kind in p, f"prompt 里没提到 source_kind={kind}"

    def test_no_card_desc_is_marked(self):
        p = J.build_prompt(**self._base())
        assert "无卡" in p

    def test_card_desc_includes_platform_and_title(self):
        args = self._base()
        args["card"] = {"platform": "B站", "title": "手机拍夜景技巧", "url": "x"}
        p = J.build_prompt(**args)
        assert "B站" in p and "手机拍夜景" in p


class TestVerdictParsing:
    def _all_bools(self, **overrides):
        base = {"naturalness": True, "source_fit": True, "persona_match": True,
                "mentions_card": True, "advertorial_feel": False,
                "used_junk_content": False, "reason": "ok"}
        base.update(overrides)
        import json as _j
        return _j.dumps(base, ensure_ascii=False)

    def test_parses_clean_json(self):
        v = J.parse_verdict(self._all_bools())
        assert v is not None
        assert v["naturalness"] is True and v["advertorial_feel"] is False

    def test_flags_ad_feel(self):
        v = J.parse_verdict(self._all_bools(advertorial_feel=True))
        assert v is not None and v["advertorial_feel"] is True

    def test_returns_none_on_missing_field(self):
        # 缺 used_junk_content 应该 None (parse 严格, 不容忍)
        import json as _j
        raw = _j.dumps({"naturalness": True, "source_fit": True, "persona_match": True,
                        "mentions_card": True, "advertorial_feel": False,
                        "reason": "x"})
        assert J.parse_verdict(raw) is None

    def test_returns_none_on_garbage(self):
        assert J.parse_verdict("不是 JSON") is None

    def test_tolerates_surrounding_text(self):
        raw = "判断: " + self._all_bools() + " 完"
        v = J.parse_verdict(raw)
        assert v is not None
