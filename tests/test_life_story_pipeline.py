"""Unit tests for life_story L1 generation pipeline (pure Python paths).

Skipped: `generate_l1_coverage` end-to-end — requires real LLM + Redis.
Covered here: deterministic helpers — profile conversion, gap computation,
prompt assembly, dedupe bucketing.
"""

import pytest

from app.services.life_story import (
    _as_list,
    _build_constraints,
    _clean_text,
    _compute_l1_gaps,
    _derive_timeline,
    _digest_existing,
    _embed_and_dedupe,
    _slice_profile,
    _spec_for_gap,
    convert_profile_to_memories,
)
from app.services.memory.taxonomy import (
    L1_SINGLETON_SUBS,
    analyze_conditional_subs,
    as_dict,
    l1_min_importance,
)


@pytest.fixture
def rich_profile():
    """Profile covering many fields, used as the baseline test fixture."""
    return {
        "identity": {
            "gender": "male",
            "age": 27,
            "birthday": "1998-06-15",
            "location": "苏州",
            "birthplace": "上海",
            "growing_up_location": "上海",
            "family": "父母健在,独生",
            "ethnicity": "汉族",
            "blood_type": "O型",
        },
        "appearance": {
            "height": "178cm", "weight": "偏瘦", "features": "戴眼镜",
            "style": "简约", "voice": "低沉",
        },
        "education_knowledge": {
            "degree": "本科",
            "strengths": ["声学", "摄影"],
            "self_taught": ["剪辑", "木工"],
        },
        "likes": {
            "foods": ["火锅", "日料"],
            "colors": ["蓝色", "灰"],
            "animals": ["柴犬"],
            "music": ["爵士"],
        },
        "dislikes": {
            "foods": ["香菜"],
            "sounds": ["尖锐声"],
            "habits": ["迟到"],
        },
        "fears": {
            "animals": ["蛇"],
            "atmospheres": ["吵闹"],
        },
        "values": {
            "motto": "认真生活",
            "believes": ["诚实", "专注"],
            "opposes": ["敷衍"],
            "goal": "做一名优秀的声音工作者",
        },
        "abilities": {
            "good_at": ["采音"],
            "never_do": ["伤害动物"],
            "limits": ["不擅长社交"],
        },
    }


class TestHelperFunctions:
    def test_clean_text_handles_none_and_empty(self):
        assert _clean_text(None) is None
        assert _clean_text("") is None
        assert _clean_text("  ") is None
        assert _clean_text("  x ") == "x"

    def test_as_list_normalizes(self):
        assert _as_list(None) == []
        assert _as_list([]) == []
        assert _as_list(["a", "", "b", None]) == ["a", "b"]
        assert _as_list("single") == ["single"]
        assert _as_list(42) == ["42"]

    def test_as_dict_fallback(self):
        assert as_dict({}, "missing") == {}
        assert as_dict({"x": None}, "x") == {}
        assert as_dict({"x": "string"}, "x") == {}
        assert as_dict({"x": {"k": "v"}}, "x") == {"k": "v"}


class TestConvertProfileToMemories:
    def test_empty_profile_yields_no_memories(self):
        assert convert_profile_to_memories({}, None) == []

    def test_rich_profile_produces_expected_memories(self, rich_profile):
        mems = convert_profile_to_memories(rich_profile, None)
        # Every memory has required keys.
        for m in mems:
            assert set(m.keys()) >= {"summary", "main_category", "sub_category",
                                     "type", "importance"}
        # identity singletons present.
        subs = {(m["main_category"], m["sub_category"]) for m in mems}
        for singleton_sub in ("性别", "年龄", "生日", "现居地",
                              "出生地", "成长地", "民族", "血型"):
            assert ("身份", singleton_sub) in subs

    def test_list_fields_expand_to_multiple_memories(self, rich_profile):
        mems = convert_profile_to_memories(rich_profile, None)
        foods = [m for m in mems
                if m["main_category"] == "偏好" and m["sub_category"] == "饮食喜好"
                and "喜欢吃" in m["summary"]]
        # ["火锅","日料"] → 2 separate memories (one per tag).
        assert len(foods) >= 2

    def test_gender_translated_to_chinese(self):
        p = {"identity": {"gender": "female"}}
        mems = convert_profile_to_memories(p, None)
        gender_mem = next(m for m in mems if m["sub_category"] == "性别")
        assert "女" in gender_mem["summary"]

    def test_career_template_applied(self):
        career = {"title": "程序员", "duties": "写代码",
                  "outputs": ["产品", "工具"], "clients": ["团队"],
                  "social_value": "创造价值"}
        mems = convert_profile_to_memories({}, career)
        career_mems = [m for m in mems
                      if m["sub_category"] == "职业/与经济" or m["sub_category"] == "工作"]
        assert len(career_mems) >= 5  # title + duties + 2 outputs + 1 client + social

    def test_malformed_sections_are_skipped(self):
        """Non-dict sections shouldn't throw."""
        p = {"identity": "not-a-dict", "likes": None, "values": 42}
        assert convert_profile_to_memories(p, None) == []


class TestComputeL1Gaps:
    def test_exempt_subs_never_in_gaps(self):
        gaps = _compute_l1_gaps([], conditional_include=set())
        for main, subs in gaps.items():
            for sub in subs:
                assert (main, sub) not in {("生活", "交互"), ("身份", "其他"),
                                           ("偏好", "其他"), ("生活", "其他"),
                                           ("情绪", "其他"), ("思维", "其他")}

    def test_emotions_are_in_gaps(self):
        """After removing emotion from EXEMPT, they must be in gap plan."""
        gaps = _compute_l1_gaps([], conditional_include=set())
        assert "情绪" in gaps
        # 16 emotion subs minus "其他" (exempt) = 15
        assert len(gaps["情绪"]) == 15

    def test_conditional_subs_skipped_by_default(self):
        gaps = _compute_l1_gaps([], conditional_include=set())
        assert ("宠物" not in gaps.get("身份", {}))
        assert ("宠物" not in gaps.get("生活", {}))
        assert ("信仰/寄托" not in gaps.get("思维", {}))

    def test_conditional_subs_included_when_passed(self):
        gaps = _compute_l1_gaps(
            [], conditional_include={("身份", "宠物"), ("生活", "宠物")}
        )
        assert "宠物" in gaps["身份"]
        assert "宠物" in gaps["生活"]

    def test_existing_memories_reduce_gap(self):
        mems = [
            {"main_category": "身份", "sub_category": "性别"},
            {"main_category": "身份", "sub_category": "性别"},  # duplicates allowed in count
        ]
        gaps = _compute_l1_gaps(mems, conditional_include=set())
        # 性别 is singleton (target=1), already met → not in gaps.
        assert "性别" not in gaps.get("身份", {})

    def test_partial_fill_reports_remainder(self):
        mems = [
            {"main_category": "偏好", "sub_category": "饮食喜好"},
            {"main_category": "偏好", "sub_category": "饮食喜好"},
        ]
        gaps = _compute_l1_gaps(mems, conditional_include=set())
        # target_multi midpoint=4, have=2 → 2 missing.
        assert gaps["偏好"]["饮食喜好"] == 2


class TestPromptAssembly:
    def test_build_constraints_includes_core_facts(self, rich_profile):
        c = _build_constraints("Lumia", "male", {"EI": 70, "NS": 60,
                                                 "TF": 40, "JP": 55,
                                                 "type": "ENFJ"},
                              rich_profile, None)
        assert "姓名: Lumia" in c
        assert "性别: 男" in c
        assert "年龄: 27" in c
        assert "现居地: 苏州" in c

    def test_build_constraints_no_mbti_gracefully(self, rich_profile):
        c = _build_constraints("X", "female", None, rich_profile, None)
        assert "姓名: X" in c
        assert "MBTI" not in c

    def test_derive_timeline_reflects_ages(self, rich_profile):
        t = _derive_timeline(rich_profile)
        assert "0-6" in t and "27" in t

    def test_spec_for_gap_uses_singleton_for_identity(self):
        spec = _spec_for_gap("身份", {"姓名": 1, "外貌特征": 3})
        assert "姓名: 1-1 条" in spec
        assert "外貌特征: 3-5 条" in spec

    def test_spec_for_gap_uses_emotion_range(self):
        spec = _spec_for_gap("情绪", {"高兴": 2})
        # L1_TARGET_EMOTION is (2, 3)
        assert "2-3 条" in spec

    def test_digest_existing_caps_per_sub(self):
        # many memories per sub → per_sub=2 trimmed.
        mems = [{"main_category": "偏好", "sub_category": "饮食喜好",
                "summary": f"memory {i}"} for i in range(10)]
        digest = _digest_existing(mems, per_sub=2)
        # Should reference "memory 0" and "memory 1" at most.
        assert "memory 0" in digest
        assert "memory 1" in digest
        assert "memory 5" not in digest

    def test_slice_profile_respects_main(self):
        profile = {"identity": {"gender": "male"}, "likes": {"foods": ["x"]},
                  "values": {"motto": "y"}}
        sliced_id = _slice_profile("身份", profile, None)
        sliced_pref = _slice_profile("偏好", profile, None)
        assert "identity" in sliced_id
        assert "likes" in sliced_pref
        # 偏好 cares about likes, not about raw values-dict (values is for thought/emotion)
        assert "values" not in sliced_pref


class TestEmbedAndDedupe:
    @pytest.mark.asyncio
    async def test_short_list_passthrough(self):
        mems = [{"summary": "只一条", "main_category": "身份",
                "sub_category": "姓名", "importance": 1.0}]
        result = await _embed_and_dedupe(mems)
        assert len(result) == 1
        assert "_embedding" in result[0]

    @pytest.mark.asyncio
    async def test_empty_list(self):
        assert await _embed_and_dedupe([]) == []


class TestIntegration:
    def test_gap_plan_covers_all_non_exempt_ai_l1(self):
        """A totally-empty profile gap plan should target the full AI L1
        coverage set (minus exempt and minus conditional without signals)."""
        from app.services.memory.taxonomy import TAXONOMY_MATRIX, L1_COVERAGE_EXEMPT, L1_CONDITIONAL_SUBS
        expected_subs: set[tuple[str, str]] = set()
        for main, subs in TAXONOMY_MATRIX["ai"][1].items():
            for sub in subs:
                key = (main, sub)
                if key in L1_COVERAGE_EXEMPT or key in L1_CONDITIONAL_SUBS:
                    continue
                expected_subs.add(key)
        gaps = _compute_l1_gaps([], conditional_include=set())
        gap_subs = {(m, s) for m, subs in gaps.items() for s in subs}
        assert expected_subs == gap_subs

    def test_rich_profile_reduces_non_empty_gap(self, rich_profile):
        mems = convert_profile_to_memories(rich_profile, None)
        cond = analyze_conditional_subs(rich_profile)
        gaps = _compute_l1_gaps(mems, conditional_include=cond)
        # 情绪 16 - 其他 = 15 still needed (profile doesn't seed emotion)
        assert "情绪" in gaps
        # 偏好/饮食喜好 partially filled
        assert gaps.get("偏好", {}).get("饮食喜好", 0) < 4

    def test_singleton_importance_meets_promotion_policy(self, rich_profile):
        """Every singleton memory's importance must be ≥ its l1_min_importance."""
        mems = convert_profile_to_memories(rich_profile, None)
        for m in mems:
            key = (m["main_category"], m["sub_category"])
            if key in L1_SINGLETON_SUBS:
                min_imp = l1_min_importance(*key)
                assert m["importance"] >= min_imp, \
                    f"{m['summary']} importance {m['importance']} < required {min_imp}"
