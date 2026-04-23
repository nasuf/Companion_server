import pytest

from app.services.memory.taxonomy import (
    resolve_taxonomy,
    summarize_batch_taxonomy,
)


class _MemoryStub:
    def __init__(self, main_category: str | None, sub_category: str | None):
        self.mainCategory = main_category
        self.subCategory = sub_category


def test_resolve_taxonomy_refuses_unknown_main_category():
    """spec：未识别的非空 main_category 拒绝默认填到生活/其他，
    保留原值并标 allowed=False，由调用方决定丢弃/归档/降级。"""
    result = resolve_taxonomy(main_category="不存在", sub_category="乱填", legacy_type=None)
    assert result.main_category == "不存在"
    assert result.sub_category == "乱填"
    assert result.allowed is False


def test_summarize_batch_taxonomy_picks_majority():
    items = [
        _MemoryStub("身份", "姓名"),
        _MemoryStub("身份", "姓名"),
        _MemoryStub("生活", "工作"),
    ]
    assert summarize_batch_taxonomy(items) == ("身份", "姓名")


# ── Alias exact match ──


class TestAliasExactMatch:
    def test_cat_alias_to_pet(self):
        r = resolve_taxonomy(main_category="身份", sub_category="猫")
        assert r.sub_category == "宠物"

    def test_dog_alias_to_pet(self):
        r = resolve_taxonomy(main_category="身份", sub_category="狗")
        assert r.sub_category == "宠物"

    def test_cat_emoji_alias(self):
        r = resolve_taxonomy(main_category="身份", sub_category="猫咪")
        assert r.sub_category == "宠物"

    def test_job_alias_to_career(self):
        r = resolve_taxonomy(main_category="身份", sub_category="上班")
        assert r.sub_category == "职业/与经济"

    def test_family_alias(self):
        r = resolve_taxonomy(main_category="身份", sub_category="爸爸")
        assert r.sub_category == "亲属关系"

    def test_social_alias(self):
        r = resolve_taxonomy(main_category="身份", sub_category="闺蜜")
        assert r.sub_category == "社会关系"

    def test_health_alias(self):
        r = resolve_taxonomy(main_category="生活", sub_category="感冒")
        assert r.sub_category == "健康"


# ── Contains matching ──


class TestContainsMatch:
    def test_yang_mao_contains_cat(self):
        """'养猫' exact alias hit."""
        r = resolve_taxonomy(main_category="身份", sub_category="养猫")
        assert r.sub_category == "宠物"

    def test_mao_de_mingzi(self):
        """'猫的名字' exact alias hit."""
        r = resolve_taxonomy(main_category="身份", sub_category="猫的名字")
        assert r.sub_category == "宠物"

    def test_compound_pet_phrase(self):
        """'养了一只猫' contains '猫' alias → 宠物."""
        r = resolve_taxonomy(main_category="身份", sub_category="养了一只猫")
        assert r.sub_category == "宠物"

    def test_compound_dog_phrase(self):
        """'家里的狗狗' contains '狗狗' alias → 宠物."""
        r = resolve_taxonomy(main_category="生活", sub_category="家里的狗狗")
        assert r.sub_category == "宠物"

    def test_compound_job_phrase(self):
        """'关于工作' contains canonical '工作' in 生活 taxonomy."""
        r = resolve_taxonomy(main_category="生活", sub_category="关于工作")
        assert r.sub_category == "工作"

    def test_compound_travel_phrase(self):
        """'出国旅游' contains '旅游' alias → 旅行."""
        r = resolve_taxonomy(main_category="生活", sub_category="出国旅游")
        assert r.sub_category == "旅行"

    def test_no_single_char_contains(self):
        """Single-char alias keys (len<2) should not trigger contains matching."""
        # "猫" is a single-char key but also an exact alias,
        # so we test something that would only match via single-char contains
        r = resolve_taxonomy(main_category="身份", sub_category="未知分类")
        assert r.sub_category == "其他"


# ── Direct match takes priority ──


class TestDirectMatchPriority:
    def test_work_direct_in_life(self):
        """'工作' is a direct sub_category in 生活, should match directly."""
        r = resolve_taxonomy(main_category="生活", sub_category="工作")
        assert r.sub_category == "工作"

    def test_pet_direct_in_identity(self):
        """'宠物' is a direct sub_category in 身份."""
        r = resolve_taxonomy(main_category="身份", sub_category="宠物")
        assert r.sub_category == "宠物"

    def test_pet_direct_in_life(self):
        """'宠物' is also a direct sub_category in 生活."""
        r = resolve_taxonomy(main_category="生活", sub_category="宠物")
        assert r.sub_category == "宠物"


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_sub(self):
        r = resolve_taxonomy(main_category="身份", sub_category="")
        assert r.sub_category == "其他"

    def test_none_sub(self):
        r = resolve_taxonomy(main_category="身份", sub_category=None)
        assert r.sub_category == "其他"

    def test_none_main_with_legacy(self):
        r = resolve_taxonomy(main_category=None, sub_category="宠物", legacy_type="identity")
        assert r.main_category == "身份"
        assert r.sub_category == "宠物"

    def test_unknown_main_marked_disallowed(self):
        """spec：未识别的非空 main 不再 silently 落到"生活"，而是保留原值 + allowed=False。"""
        r = resolve_taxonomy(main_category="不存在的", sub_category="宠物")
        assert r.main_category == "不存在的"
        assert r.allowed is False

    def test_whitespace_handling(self):
        r = resolve_taxonomy(main_category=" 身份 ", sub_category=" 猫咪 ")
        assert r.main_category == "身份"
        assert r.sub_category == "宠物"


# ── D10: spec §1.4 「其他特殊事件」子类 ──


class TestOtherSpecialEvents:
    """spec §1.4 生活记忆明列"其他特殊事件"作为独立子类 (非通用"其他")."""

    def test_allowed_in_user_life_matrix(self):
        from app.services.memory.taxonomy import TAXONOMY_MATRIX
        assert "其他特殊事件" in TAXONOMY_MATRIX["user"][2]["生活"]

    def test_allowed_in_ai_life_matrix(self):
        from app.services.memory.taxonomy import TAXONOMY_MATRIX
        assert "其他特殊事件" in TAXONOMY_MATRIX["ai"][2]["生活"]

    def test_direct_match_is_preserved(self):
        """直接传 "其他特殊事件" 应原样通过, 不被降级到 "其他"."""
        r = resolve_taxonomy(main_category="生活", sub_category="其他特殊事件")
        assert r.sub_category == "其他特殊事件"
        assert r.allowed is True

    def test_alias_special_event_maps_to_canonical(self):
        r = resolve_taxonomy(main_category="生活", sub_category="特殊事件")
        assert r.sub_category == "其他特殊事件"

    def test_alias_unforgettable_experience(self):
        r = resolve_taxonomy(main_category="生活", sub_category="难忘经历")
        assert r.sub_category == "其他特殊事件"

    def test_general_other_still_works(self):
        """通用兜底 "其他" 独立存在, 不被 "其他特殊事件" 吞并."""
        r = resolve_taxonomy(main_category="生活", sub_category="其他")
        assert r.sub_category == "其他"
