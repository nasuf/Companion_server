"""Unit tests for L1 coverage helpers in taxonomy (gap plan, conditional
analysis, per-sub target / importance)."""

import pytest

from app.services.memory.taxonomy import (
    L1_CONDITIONAL_SUBS,
    L1_COVERAGE_EXEMPT,
    L1_SINGLETON_SUBS,
    L1_TARGET_EMOTION,
    L1_TARGET_MULTI,
    PROMOTION_RULES,
    TAXONOMY_MATRIX,
    analyze_conditional_subs,
    l1_min_importance,
    l1_target_count,
)


class TestL1TargetCount:
    @pytest.mark.parametrize("main, sub", sorted(L1_SINGLETON_SUBS))
    def test_singletons_target_one(self, main, sub):
        assert l1_target_count(main, sub) == 1

    @pytest.mark.parametrize("main, sub", sorted(L1_COVERAGE_EXEMPT))
    def test_exempt_target_zero(self, main, sub):
        assert l1_target_count(main, sub) == 0

    def test_emotion_target_is_midpoint(self):
        expected = (L1_TARGET_EMOTION[0] + L1_TARGET_EMOTION[1]) // 2
        assert l1_target_count("情绪", "高兴") == expected
        assert l1_target_count("情绪", "悲伤") == expected

    def test_default_multi_target(self):
        expected = (L1_TARGET_MULTI[0] + L1_TARGET_MULTI[1]) // 2
        assert l1_target_count("身份", "外貌特征") == expected
        assert l1_target_count("生活", "旅行") == expected
        assert l1_target_count("偏好", "饮食喜好") == expected


class TestL1MinImportance:
    def test_uses_promotion_rule(self):
        # 思维/价值观 has min_importance 0.9 in PROMOTION_RULES
        assert l1_min_importance("思维", "价值观") == pytest.approx(0.9)
        assert l1_min_importance("身份", "姓名") == pytest.approx(0.8)

    def test_default_for_unregistered(self):
        assert l1_min_importance("情绪", "高兴") == 0.75


class TestAnalyzeConditionalSubs:
    def test_empty_profile_returns_empty(self):
        assert analyze_conditional_subs({}) == set()

    def test_pet_from_family_mention(self):
        p = {"identity": {"family": "和父母、一只叫小白的拉布拉多一起住"}}
        subs = analyze_conditional_subs(p)
        assert ("身份", "宠物") in subs
        assert ("生活", "宠物") in subs

    def test_pet_from_liked_animals(self):
        p = {"likes": {"animals": ["柴犬"]}}
        assert ("身份", "宠物") in analyze_conditional_subs(p)

    def test_feared_animals_do_not_trigger_pet(self):
        """fears.animals alone must not pull in 宠物 子类 (false-positive fix)."""
        p = {"likes": {}, "fears": {"animals": ["猫", "狗"]}}
        subs = analyze_conditional_subs(p)
        assert ("身份", "宠物") not in subs
        assert ("生活", "宠物") not in subs

    def test_liked_animals_conflicting_with_fears_skips_pet(self):
        p = {"likes": {"animals": ["猫"]}, "fears": {"animals": ["猫"]}}
        assert ("身份", "宠物") not in analyze_conditional_subs(p)

    def test_faith_from_motto(self):
        p = {"values": {"motto": "信仰是我坚持的力量", "believes": ["因果"]}}
        assert ("思维", "信仰/寄托") in analyze_conditional_subs(p)

    def test_no_faith_trigger_from_neutral_motto(self):
        p = {"values": {"motto": "努力就有回报", "believes": ["勤能补拙"]}}
        assert ("思维", "信仰/寄托") not in analyze_conditional_subs(p)

    def test_dislikes_habits_triggers_interpersonal(self):
        p = {"dislikes": {"habits": ["迟到", "食言"]}}
        assert ("偏好", "人际厌恶") in analyze_conditional_subs(p)

    def test_values_opposes_triggers_interpersonal(self):
        p = {"values": {"opposes": ["虚伪", "敷衍"]}}
        assert ("偏好", "人际厌恶") in analyze_conditional_subs(p)

    def test_unknown_fields_safe(self):
        """Malformed profile shouldn't throw."""
        p = {"identity": "not-a-dict", "likes": None, "values": 42}
        assert analyze_conditional_subs(p) == set()


class TestTaxonomyInvariants:
    def test_singletons_are_all_under_identity(self):
        assert all(main == "身份" for main, _ in L1_SINGLETON_SUBS)

    def test_conditional_subs_live_in_ai_l1(self):
        ai_l1 = TAXONOMY_MATRIX["ai"][1]
        for main, sub in L1_CONDITIONAL_SUBS:
            assert sub in ai_l1.get(main, ()), \
                f"{main}/{sub} declared conditional but not in AI L1 taxonomy"

    def test_exempt_subs_live_in_ai_l1(self):
        ai_l1 = TAXONOMY_MATRIX["ai"][1]
        for main, sub in L1_COVERAGE_EXEMPT:
            assert sub in ai_l1.get(main, ()), \
                f"{main}/{sub} declared exempt but not in AI L1 taxonomy"

    def test_promotion_rules_align_with_taxonomy(self):
        ai_l1 = TAXONOMY_MATRIX["ai"][1]
        for main, sub in PROMOTION_RULES.keys():
            # All promotion rule keys should reference a real (main, sub).
            assert sub in ai_l1.get(main, ()), \
                f"PROMOTION_RULES[{main}/{sub}] not a real AI L1 pair"
