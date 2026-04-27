"""Unit tests for life_story L1 generation pipeline (pure Python paths).

Skipped: `generate_l1_coverage` end-to-end — requires real LLM + Redis.
Covered here: deterministic helpers — profile conversion, gap computation,
prompt assembly, dedupe bucketing.
"""

import pytest

from app.services.life_story import (
    _as_list,
    _clean_text,
    _compute_l1_gaps,
    _embed_and_dedupe,
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
                  "clients": ["团队"], "social_value": "创造价值"}
        mems = convert_profile_to_memories({}, career)
        career_mems = [m for m in mems
                      if m["sub_category"] == "职业/与经济" or m["sub_category"] == "工作"]
        assert len(career_mems) >= 4  # title + duties + 1 client + social

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


# NOTE: Plan B 后旧 phase-3 LLM gap-fill 路径全删 (_build_constraints /
# _derive_timeline / _spec_for_gap / _digest_existing / _slice_profile),
# 对应 prompt assembly 测试同步移除. 单步 character.generation 的 prompt
# contract 在 tests/test_convert_profile_contract.py 中保护。


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


class TestContradictionDetection:
    """LLM-based pairwise contradiction scan (spec §1.4)."""

    @pytest.mark.asyncio
    async def test_drops_lower_importance_side(self, monkeypatch):
        from app.services import life_story

        memories = [
            {"summary": "我养了一只名为糯米的博美犬",
             "main_category": "身份", "sub_category": "宠物", "importance": 0.85},
            {"summary": "我没有养宠物，家里常备急救包帮助小动物",
             "main_category": "生活", "sub_category": "宠物", "importance": 0.85},
            {"summary": "我叫Hia",
             "main_category": "身份", "sub_category": "姓名", "importance": 0.95},
        ]

        async def fake_invoke_json(_model, _prompt):
            # Simulate LLM finding the pet contradiction.
            return {"contradictions": [{"a": 0, "b": 1, "reason": "养/没养冲突"}]}

        monkeypatch.setattr(life_story, "invoke_json", fake_invoke_json)
        monkeypatch.setattr(life_story, "get_utility_model", lambda: object())
        async def fake_get_prompt_text(_key): return "{memory_list}"
        monkeypatch.setattr(life_story, "get_prompt_text", fake_get_prompt_text)

        result = await life_story._detect_and_resolve_contradictions(memories)
        # Tied importance → drop b (the second one, "没有养宠物").
        assert len(result) == 2
        assert all("没有养宠物" not in m["summary"] for m in result)

    @pytest.mark.asyncio
    async def test_no_contradiction_returns_unchanged(self, monkeypatch):
        from app.services import life_story
        memories = [
            {"summary": "我叫Hia", "main_category": "身份",
             "sub_category": "姓名", "importance": 0.95},
            {"summary": "我喜欢咖啡", "main_category": "偏好",
             "sub_category": "饮食喜好", "importance": 0.86},
        ]

        async def fake_invoke_json(_model, _prompt):
            return {"contradictions": []}

        monkeypatch.setattr(life_story, "invoke_json", fake_invoke_json)
        monkeypatch.setattr(life_story, "get_utility_model", lambda: object())
        async def fake_get_prompt_text(_key): return "{memory_list}"
        monkeypatch.setattr(life_story, "get_prompt_text", fake_get_prompt_text)

        result = await life_story._detect_and_resolve_contradictions(memories)
        assert result == memories

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self, monkeypatch):
        """LLM 调用炸了 → 跳过, 不阻塞 agent 创建."""
        from app.services import life_story
        memories = [
            {"summary": "a", "main_category": "身份", "sub_category": "姓名", "importance": 0.9},
            {"summary": "b", "main_category": "身份", "sub_category": "姓名", "importance": 0.9},
        ]

        async def fake_invoke_json(_model, _prompt):
            raise RuntimeError("LLM down")

        monkeypatch.setattr(life_story, "invoke_json", fake_invoke_json)
        monkeypatch.setattr(life_story, "get_utility_model", lambda: object())
        async def fake_get_prompt_text(_key): return "{memory_list}"
        monkeypatch.setattr(life_story, "get_prompt_text", fake_get_prompt_text)

        result = await life_story._detect_and_resolve_contradictions(memories)
        assert result == memories  # 原样回退

    @pytest.mark.asyncio
    async def test_skips_when_too_few_memories(self, monkeypatch):
        from app.services import life_story
        called = False
        async def fake_invoke_json(_model, _prompt):
            nonlocal called
            called = True
            return {"contradictions": []}
        monkeypatch.setattr(life_story, "invoke_json", fake_invoke_json)

        assert await life_story._detect_and_resolve_contradictions([]) == []
        single = [{"summary": "x", "main_category": "身份",
                   "sub_category": "姓名", "importance": 0.9}]
        assert await life_story._detect_and_resolve_contradictions(single) == single
        assert called is False  # 短路, 不调 LLM

    @pytest.mark.asyncio
    async def test_drops_higher_importance_pair_loser_correctly(self, monkeypatch):
        """importance 不等时, drop 低分那条 (而非简单按 a/b 顺序)."""
        from app.services import life_story
        memories = [
            {"summary": "我没养宠物", "main_category": "生活",
             "sub_category": "宠物", "importance": 0.86},
            {"summary": "我有一只博美犬", "main_category": "身份",
             "sub_category": "宠物", "importance": 0.92},
        ]
        async def fake_invoke_json(_m, _p):
            return {"contradictions": [{"a": 0, "b": 1, "reason": "宠物冲突"}]}
        monkeypatch.setattr(life_story, "invoke_json", fake_invoke_json)
        monkeypatch.setattr(life_story, "get_utility_model", lambda: object())
        async def fake_get_prompt_text(_key): return "{memory_list}"
        monkeypatch.setattr(life_story, "get_prompt_text", fake_get_prompt_text)

        result = await life_story._detect_and_resolve_contradictions(memories)
        assert len(result) == 1
        assert result[0]["summary"] == "我有一只博美犬"

    @pytest.mark.asyncio
    async def test_ignores_invalid_pairs(self, monkeypatch):
        """LLM 输出包含无效 idx / 自反对 / 重叠 drop, 应安全跳过."""
        from app.services import life_story
        memories = [
            {"summary": "a", "main_category": "身份", "sub_category": "姓名", "importance": 0.9},
            {"summary": "b", "main_category": "身份", "sub_category": "姓名", "importance": 0.85},
        ]
        async def fake_invoke_json(_m, _p):
            return {"contradictions": [
                {"a": 0, "b": 99, "reason": "out of range"},
                {"a": 1, "b": 1, "reason": "self"},
                "not a dict",
                {"a": 0, "b": 1, "reason": "valid"},
                {"a": 0, "b": 1, "reason": "duplicate after drop"},
            ]}
        monkeypatch.setattr(life_story, "invoke_json", fake_invoke_json)
        monkeypatch.setattr(life_story, "get_utility_model", lambda: object())
        async def fake_get_prompt_text(_key): return "{memory_list}"
        monkeypatch.setattr(life_story, "get_prompt_text", fake_get_prompt_text)

        result = await life_story._detect_and_resolve_contradictions(memories)
        # Only the valid pair (0, 1) → drop b (importance 0.85).
        assert len(result) == 1
        assert result[0]["summary"] == "a"
