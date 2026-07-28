"""Phase 3 polarity 检测 + dedup/retrieval 极性校验.

实证基础: bge-m3 反义对 cosine 0.84-0.89, 几乎跟同义对 (0.92) 难分.
- DEDUP threshold 0.85: "我住北京" vs "我不住北京" 0.89 > 0.85 → 误判重复 → 数据丢失
- RETRIEVAL threshold 0.5: 反义对都召回 → prompt 同时含正反事实 → LLM 选错
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# polarity 模块基础: has_negation / is_polarity_match
# ═══════════════════════════════════════════════════════════════════


def test_has_negation_chinese():
    from app.services.memory.polarity import has_negation

    assert has_negation("我不喜欢咖啡")
    assert has_negation("我没去过北京")
    # 真否定
    assert has_negation("用户不住在上海")
    assert has_negation("AI 没有兄弟姐妹")
    # 无否定
    assert not has_negation("我喜欢咖啡")
    assert not has_negation("用户在北京工作")
    assert not has_negation("今天天气好")
    # neutralized: "没事" = I'm fine, 不算否定 (即便含"没")
    assert not has_negation("我没事")


def test_has_negation_neutralizes_common_phrases():
    """'不错'/'差不多' 等非否定语境短语不算 negation."""
    from app.services.memory.polarity import has_negation

    # 单纯使用这些短语不算否定
    assert not has_negation("不错")
    assert not has_negation("不少")
    assert not has_negation("差不多")
    assert not has_negation("不仅好吃还便宜")
    # 这种语境实际是 positive

    # 但若同句还有真否定, 仍命中
    assert has_negation("我感觉不错, 但他没来")  # 后半句"没来" 是真 negation


def test_has_negation_english():
    from app.services.memory.polarity import has_negation

    assert has_negation("I don't like coffee")
    assert has_negation("I do not like coffee")
    assert has_negation("user has never been to Tokyo")
    assert has_negation("without sugar please")

    assert not has_negation("I like coffee")
    assert not has_negation("user lives in Beijing")


def test_is_polarity_match():
    from app.services.memory.polarity import is_polarity_match

    # 同极性 (都 positive)
    assert is_polarity_match("我喜欢咖啡", "我爱咖啡")
    # 同极性 (都 negation)
    assert is_polarity_match("我不喜欢咖啡", "我从来没喜欢过咖啡")
    # 反义 (一 positive 一 negation)
    assert not is_polarity_match("我喜欢咖啡", "我不喜欢咖啡")
    assert not is_polarity_match("我住北京", "我不住北京")


def test_semantic_conflict_detects_non_negation_opposites():
    from app.services.memory.polarity import semantic_conflict_reasons

    assert "偏好立场" in semantic_conflict_reasons("用户喜欢咖啡", "用户讨厌咖啡")
    assert "伴侣身份" in semantic_conflict_reasons("用户前男友联系她", "用户前女友联系她")
    assert "伴侣状态" in semantic_conflict_reasons("用户男朋友来杭州", "用户前男友来杭州")
    assert semantic_conflict_reasons("用户前任联系她", "用户前男友联系她") == []
    assert "就医状态" in semantic_conflict_reasons("用户妈妈住院", "用户妈妈出院")
    assert "就医阶段" in semantic_conflict_reasons("用户妈妈手术", "用户妈妈出院")
    assert semantic_conflict_reasons("用户不喜欢咖啡", "用户讨厌咖啡") == []


def test_query_semantic_conflict_is_directional():
    from app.services.memory.polarity import query_semantic_conflict_reasons

    assert query_semantic_conflict_reasons("我对咖啡的看法", "用户讨厌咖啡") == []
    assert "偏好立场" in query_semantic_conflict_reasons("我喜欢咖啡吗", "用户讨厌咖啡")
    assert "伴侣身份" in query_semantic_conflict_reasons("前男友那件事", "用户前女友联系她")
    assert query_semantic_conflict_reasons("前任那件事", "用户前男友联系她") == []
    assert query_semantic_conflict_reasons("我不喜欢咖啡吗", "用户讨厌咖啡") == []


def test_query_semantic_conflict_only_downweights_aligned_factual_negation():
    from app.services.memory.polarity import query_semantic_conflict_reasons

    assert query_semantic_conflict_reasons(
        "一个不是特别复杂的coding",
        "用户是一名程序员",
    ) == []
    assert query_semantic_conflict_reasons(
        "一个不是特别复杂的coding",
        "用户被老板要求两天内完成一个项目，觉得很难",
    ) == []
    assert "否定极性" in query_semantic_conflict_reasons(
        "我不是程序员",
        "用户是一名程序员",
    )
    assert "否定极性" in query_semantic_conflict_reasons(
        "我不住北京",
        "用户住在北京",
    )


# ═══════════════════════════════════════════════════════════════════
# 3.1: dedup polarity check
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_dedup_skips_polarity_mismatch():
    """Phase 3.1: cosine > 0.85 但极性不一致 → 不算重复, 允许两条都存."""
    from app.services.memory.storage.persistence import find_duplicate_id

    # 模拟 search_by_embedding 返回一条高 sim 的反义 memory
    fake_results = [
        {"id": "old-mem-1", "summary": "用户住在北京",
         "content": "用户住在北京", "similarity": 0.89},
    ]

    with patch(
        "app.services.memory.storage.persistence.search_by_embedding",
        new_callable=AsyncMock, return_value=fake_results,
    ):
        # 新内容是反义版本: "用户不住在北京"
        result = await find_duplicate_id(
            user_id="u1",
            content="用户不住在北京",
            embedding=[0.1] * 1024,
        )

    # 反义 → 不算重复, 返 None (允许两条都存)
    assert result is None, (
        "反义对 (cosine 0.89 > DEDUP_THRESHOLD) 必须不被判重复, "
        "防数据丢失"
    )


@pytest.mark.asyncio
async def test_dedup_still_fires_for_paraphrase():
    """同极性 paraphrase (cosine > 0.85) 仍走正常 dedup."""
    from app.services.memory.storage.persistence import find_duplicate_id

    fake_results = [
        {"id": "old-mem-1", "summary": "用户喜欢咖啡",
         "content": "用户喜欢咖啡", "similarity": 0.92},
    ]

    with patch(
        "app.services.memory.storage.persistence.search_by_embedding",
        new_callable=AsyncMock, return_value=fake_results,
    ):
        # 新内容是同义改写, 都 positive
        result = await find_duplicate_id(
            user_id="u1", content="用户爱咖啡", embedding=[0.1] * 1024,
        )

    # 同极性 → 视为重复
    assert result == "old-mem-1"


@pytest.mark.asyncio
async def test_dedup_skips_semantic_conflict_without_negation():
    """喜欢/讨厌等无显式否定的反义事实也不能被高 cosine 吃掉。"""
    from app.services.memory.storage.persistence import find_duplicate_id

    fake_results = [
        {"id": "old-mem-1", "summary": "用户喜欢咖啡",
         "content": "用户喜欢咖啡", "similarity": 0.92},
    ]

    with patch(
        "app.services.memory.storage.persistence.search_by_embedding",
        new_callable=AsyncMock, return_value=fake_results,
    ):
        result = await find_duplicate_id(
            user_id="u1", content="用户讨厌咖啡", embedding=[0.1] * 1024,
        )

    assert result is None


@pytest.mark.asyncio
async def test_dedup_below_threshold_unchanged():
    """sim < DEDUP_THRESHOLD 不进入极性判断 (走正常路径返 None)."""
    from app.services.memory.storage.persistence import find_duplicate_id

    fake_results = [
        {"id": "old-mem-1", "summary": "用户喜欢咖啡",
         "content": "用户喜欢咖啡", "similarity": 0.6},  # 低于 0.85
    ]

    with patch(
        "app.services.memory.storage.persistence.search_by_embedding",
        new_callable=AsyncMock, return_value=fake_results,
    ):
        result = await find_duplicate_id(
            user_id="u1", content="任何内容", embedding=[0.1] * 1024,
        )

    assert result is None


# ═══════════════════════════════════════════════════════════════════
# 3.2: retrieval polarity 降权
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_retrieval_downweights_pos_when_user_negates():
    """用户 query 含否定 → positive candidate 显著降权."""
    from app.services.memory.retrieval import hybrid

    # 模拟两个 candidate: 一 positive, 一 negation, 同 sim
    pos_cand = {
        "id": "m-pos", "summary": "用户喜欢咖啡", "content": "用户喜欢咖啡",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }
    neg_cand = {
        "id": "m-neg", "summary": "用户不喜欢咖啡", "content": "用户不喜欢咖啡",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }

    captured_scores = {}
    original_select = hybrid.select_context

    def _spy_select(candidates, budget, **kwargs):
        for c in candidates:
            captured_scores[c["id"]] = c.get("rank_score", 0)
        return original_select(candidates, budget, **kwargs)

    with (
        patch.object(hybrid, "search_similar_tiers", new_callable=AsyncMock,
                     return_value=[pos_cand, neg_cand]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval", new_callable=AsyncMock),
        patch.object(hybrid, "select_context", side_effect=_spy_select),
    ):
        # 用户 query 含否定: "我不喜欢什么咖啡?"
        await hybrid.hybrid_retrieve(
            message="我不喜欢什么咖啡?",
            user_id="u1", workspace_id="w1",
        )

    # negation candidate 应该排前 (不被降权), positive 被显著降权。
    # rerank v2 会额外给字面/主题命中加分, 所以这里验证行为而不锁死精确倍率。
    assert captured_scores["m-neg"] > captured_scores["m-pos"], (
        f"用户否定 query 时, 否定 candidate 应排前 negation; got "
        f"neg={captured_scores['m-neg']:.3f}, pos={captured_scores['m-pos']:.3f}"
    )
    assert captured_scores["m-pos"] < captured_scores["m-neg"] * 0.35


@pytest.mark.asyncio
async def test_retrieval_no_downweight_when_user_positive():
    """用户 query 无否定 → 不降权 negation candidate (用户可能想看全部偏好)."""
    from app.services.memory.retrieval import hybrid

    pos_cand = {
        "id": "m-pos", "summary": "用户喜欢咖啡", "content": "用户喜欢咖啡",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }
    neg_cand = {
        "id": "m-neg", "summary": "用户不喜欢咖啡", "content": "用户不喜欢咖啡",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }

    captured_scores = {}
    original_select = hybrid.select_context

    def _spy_select(candidates, budget, **kwargs):
        for c in candidates:
            captured_scores[c["id"]] = c.get("rank_score", 0)
        return original_select(candidates, budget, **kwargs)

    with (
        patch.object(hybrid, "search_similar_tiers", new_callable=AsyncMock,
                     return_value=[pos_cand, neg_cand]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval", new_callable=AsyncMock),
        patch.object(hybrid, "select_context", side_effect=_spy_select),
    ):
        # 用户 query 无否定: "我对咖啡的看法"
        await hybrid.hybrid_retrieve(
            message="我对咖啡的看法",
            user_id="u1", workspace_id="w1",
        )

    # 两 candidate score 相同 (都 importance × freshness × similarity, 不降权)
    assert abs(captured_scores["m-pos"] - captured_scores["m-neg"]) < 0.001, (
        f"positive query 不该降权 negation candidate; got "
        f"pos={captured_scores['m-pos']:.3f}, neg={captured_scores['m-neg']:.3f}"
    )


@pytest.mark.asyncio
async def test_retrieval_downweights_dislike_when_query_likes():
    """用户明确问喜欢时, 讨厌/过敏类反向偏好应被降权。"""
    from app.services.memory.retrieval import hybrid

    pos_cand = {
        "id": "m-pos", "summary": "用户喜欢咖啡", "content": "用户喜欢咖啡",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }
    neg_cand = {
        "id": "m-neg", "summary": "用户讨厌咖啡", "content": "用户讨厌咖啡",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }

    captured_scores = {}
    captured_reasons = {}
    original_select = hybrid.select_context

    def _spy_select(candidates, budget, **kwargs):
        for c in candidates:
            captured_scores[c["id"]] = c.get("rank_score", 0)
            captured_reasons[c["id"]] = c.get("rank_reasons", [])
        return original_select(candidates, budget, **kwargs)

    with (
        patch.object(hybrid, "search_similar_tiers", new_callable=AsyncMock,
                     return_value=[pos_cand, neg_cand]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "search_related_memories_for_query",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval", new_callable=AsyncMock),
        patch.object(hybrid, "select_context", side_effect=_spy_select),
    ):
        await hybrid.hybrid_retrieve(
            message="我喜欢咖啡吗?",
            user_id="u1", workspace_id="w1",
        )

    assert captured_scores["m-pos"] > captured_scores["m-neg"]
    assert any("语义对立降权" in r for r in captured_reasons["m-neg"])


@pytest.mark.asyncio
async def test_retrieval_downweights_wrong_partner_role():
    """前男友/前女友这类 embedding 高相似但角色相反的记忆要降权。"""
    from app.services.memory.retrieval import hybrid

    male_ex = {
        "id": "m-male", "summary": "用户前男友曾联系她", "content": "用户前男友曾联系她",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }
    female_ex = {
        "id": "m-female", "summary": "用户前女友曾联系她", "content": "用户前女友曾联系她",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }

    captured_scores = {}
    original_select = hybrid.select_context

    def _spy_select(candidates, budget, **kwargs):
        for c in candidates:
            captured_scores[c["id"]] = c.get("rank_score", 0)
        return original_select(candidates, budget, **kwargs)

    with (
        patch.object(hybrid, "search_similar_tiers", new_callable=AsyncMock,
                     return_value=[female_ex, male_ex]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "search_related_memories_for_query",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval", new_callable=AsyncMock),
        patch.object(hybrid, "select_context", side_effect=_spy_select),
    ):
        await hybrid.hybrid_retrieve(
            message="前男友后来怎么样了?",
            user_id="u1", workspace_id="w1",
        )

    assert captured_scores["m-male"] > captured_scores["m-female"]


@pytest.mark.asyncio
async def test_retrieval_downweights_medical_status_mismatch():
    """问出院时, 住院状态记忆不应压过出院状态记忆。"""
    from app.services.memory.retrieval import hybrid

    admitted = {
        "id": "m-admitted", "summary": "用户妈妈最近住院", "content": "用户妈妈最近住院",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }
    discharged = {
        "id": "m-discharged", "summary": "用户妈妈已经出院", "content": "用户妈妈已经出院",
        "importance": 0.7, "similarity": 0.8, "created_at": None,
    }

    captured_scores = {}
    original_select = hybrid.select_context

    def _spy_select(candidates, budget, **kwargs):
        for c in candidates:
            captured_scores[c["id"]] = c.get("rank_score", 0)
        return original_select(candidates, budget, **kwargs)

    with (
        patch.object(hybrid, "search_similar_tiers", new_callable=AsyncMock,
                     return_value=[admitted, discharged]),
        patch.object(hybrid, "search_by_time_range",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "search_related_memories_for_query",
                     new_callable=AsyncMock, return_value=[]),
        patch.object(hybrid, "cache_retrieval",
                     new_callable=AsyncMock, return_value=None),
        patch.object(hybrid, "cache_set_retrieval", new_callable=AsyncMock),
        patch.object(hybrid, "select_context", side_effect=_spy_select),
    ):
        await hybrid.hybrid_retrieve(
            message="妈妈出院了吗?",
            user_id="u1", workspace_id="w1",
        )

    assert captured_scores["m-discharged"] > captured_scores["m-admitted"]
