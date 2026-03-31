"""Tests for memory category normalization — keyword hints and integration."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.normalization import (
    _keyword_hint,
    normalize_memory_category,
)


# ── _keyword_hint unit tests ──


class TestKeywordHint:
    def test_cat_in_summary(self):
        assert _keyword_hint("用户养了一只猫叫小花") == ("身份", "宠物")

    def test_dog_in_summary(self):
        assert _keyword_hint("用户家里有一只狗狗") == ("身份", "宠物")

    def test_cat_emoji_in_summary(self):
        assert _keyword_hint("用户的猫咪叫咪咪") == ("身份", "宠物")

    def test_walk_dog_in_summary(self):
        assert _keyword_hint("用户今天遛狗去了公园") == ("生活", "宠物")

    def test_family_in_summary(self):
        assert _keyword_hint("用户的妈妈做了好吃的") == ("身份", "亲属关系")

    def test_grandparent_in_summary(self):
        assert _keyword_hint("用户和爷爷住在一起") == ("身份", "亲属关系")

    def test_health_in_summary(self):
        assert _keyword_hint("用户最近感冒了") == ("生活", "健康")

    def test_housing_in_summary(self):
        assert _keyword_hint("用户准备搬家了") == ("生活", "居住")

    def test_girlfriend_in_summary(self):
        assert _keyword_hint("用户和女朋友吵架了") == ("身份", "社会关系")

    def test_no_match(self):
        assert _keyword_hint("用户今天很开心") is None

    def test_empty_summary(self):
        assert _keyword_hint("") is None

    def test_none_summary(self):
        assert _keyword_hint(None) is None

    def test_longer_keyword_wins(self):
        """'猫咪' (2 chars) should match before '猫' (1 char)."""
        result = _keyword_hint("用户的猫咪生病了")
        assert result == ("身份", "宠物")


# ── normalize_memory_category integration tests ──


class TestNormalizeCategoryWithSummary:
    @pytest.mark.asyncio
    async def test_other_rescued_by_keyword_hint(self):
        """When LLM returns '其他' but summary has cat keyword, keyword hint rescues."""
        result = await normalize_memory_category(
            main_category="生活",
            sub_category="养猫相关",  # non-standard → contains match should catch "养猫"
            summary="用户养了一只猫叫小花",
        )
        assert result.sub_category == "宠物"

    @pytest.mark.asyncio
    async def test_already_specific_not_overridden(self):
        """When LLM returns a valid sub_category, keyword hint should not interfere."""
        result = await normalize_memory_category(
            main_category="身份",
            sub_category="宠物",
            summary="用户养了一只猫叫小花",
        )
        assert result.sub_category == "宠物"
        assert result.main_category == "身份"

    @pytest.mark.asyncio
    async def test_no_summary_falls_through(self):
        """Without summary, non-standard sub falls to semantic or 其他."""
        # Mock embedding to avoid actual API call
        with patch(
            "app.services.memory.normalization.generate_embedding",
            new_callable=AsyncMock,
            return_value=[0.0] * 768,
        ):
            result = await normalize_memory_category(
                main_category="身份",
                sub_category="未知分类xyz",
                summary=None,
            )
            # With zero embeddings the semantic fallback won't match
            assert result.sub_category == "其他"

    @pytest.mark.asyncio
    async def test_keyword_hint_cross_category(self):
        """Keyword hint can correct main_category when sub doesn't exist in LLM's main."""
        result = await normalize_memory_category(
            main_category="情绪",  # LLM wrongly said 情绪
            sub_category="不明",
            summary="用户家的猫咪叫咪咪",
        )
        # keyword hint should push to 身份/宠物
        assert result.sub_category == "宠物"
        assert result.main_category == "身份"

    @pytest.mark.asyncio
    async def test_contains_match_catches_before_keyword_hint(self):
        """Contains matching in resolve_taxonomy catches '养猫' via alias before keyword hint is needed."""
        result = await normalize_memory_category(
            main_category="身份",
            sub_category="养猫",  # contains "养猫" alias → "宠物" directly
            summary="用户养了一只猫",
        )
        assert result.sub_category == "宠物"
