"""L3 有界采样的边界守卫 (Phase 2).

这一步改的是检索语义 —— 全流程里风险最高的改动。冷层记忆重新有机会进 prompt,
所以"有界"必须真的有界: 数量、质量、以及能否一键退回旧行为, 三条都要钉死。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.retrieval import hybrid


def _mem(mid: str, sim: float, level: int = 3, source: str = "user") -> dict:
    return {
        "id": mid, "content": f"记忆 {mid}", "similarity": sim, "level": level,
        "importance": 0.4, "source": source, "main_category": "生活",
        "sub_category": "日常", "created_at": None,
    }


class TestBudgetIsRespected:
    def test_budget_is_small_relative_to_the_injection_quota(self):
        """注入上限是 user/ai 各 10 条; 冷层占位必须远小于它, 否则会挤掉热记忆。"""
        from app.services.memory.retrieval.context_selector import MAX_MEMORIES_INJECTED

        assert 0 < hybrid.WARM_SAMPLE_BUDGET <= MAX_MEMORIES_INJECTED // 3

    def test_warm_threshold_is_stricter_than_the_hot_threshold(self):
        """冷记忆要够像才值得翻出来, 否则等于把噪声放回候选集。"""
        assert hybrid.WARM_SAMPLE_THRESHOLD > hybrid._SIMILARITY_THRESHOLD

    def test_warm_threshold_matches_the_l3_awakening_floor(self):
        """两条冷层通路共用一个门 —— 否则同一条记忆在一条路上够格、另一条不够。"""
        from app.services.memory.retrieval.legacy import _L3_SIMILARITY_FLOOR

        assert hybrid.WARM_SAMPLE_THRESHOLD == _L3_SIMILARITY_FLOOR


def _run_retrieval(fake_tiers, message="聊点什么", **overrides):
    """跑一次检索, 用假的分层检索桩。返回 (结果, 捕获到的 tiers)。"""
    captured: dict = {}

    async def _fake(query, user_id, tiers, workspace_id=None):
        captured["tiers"] = tiers
        captured["query"] = query
        return fake_tiers(tiers)  # 扁平列表, 冷热行靠自身 level 区分

    patches = [
        patch.object(hybrid, "search_similar_tiers", _fake),
        patch.object(hybrid, "search_by_time_range", AsyncMock(return_value=[])),
        patch.object(
            hybrid, "search_related_memories_for_query", AsyncMock(return_value=[]),
        ),
        patch.object(hybrid, "cache_retrieval", AsyncMock(return_value=None)),
        patch.object(hybrid, "cache_set_retrieval", AsyncMock()),
    ]
    for name, value in overrides.items():
        patches.append(patch.object(hybrid, name, value))

    async def _go():
        for ptc in patches:
            ptc.start()
        try:
            result = await hybrid.hybrid_retrieve(
                message, "user-1", workspace_id="ws-1",
            )
        finally:
            for ptc in patches:
                ptc.stop()
        return result, captured

    return _go()


class TestRollback:
    @pytest.mark.asyncio
    async def test_budget_zero_issues_no_l3_query(self):
        """κ=0 是这一步的回滚开关, 必须真的连查询都不发。"""
        _, captured = await _run_retrieval(
            lambda tiers: [], WARM_SAMPLE_BUDGET=0,
        )
        assert [levels for levels, _ in captured["tiers"]] == [[1, 2]], (
            "κ=0 时仍然请求了 L3"
        )

    @pytest.mark.asyncio
    async def test_budget_positive_requests_the_cold_tier(self):
        _, captured = await _run_retrieval(lambda tiers: [])
        assert ([3], hybrid.WARM_SAMPLE_BUDGET) in captured["tiers"]


class TestWarmCandidatesAreFiltered:
    @pytest.mark.asyncio
    async def test_low_similarity_cold_memories_are_dropped(self):
        """低于冷层门的候选不该进候选集 —— 那是把降级的东西原样放回来。"""
        captured: dict = {}

        def _capture(candidates, *a, **kw):
            captured["ids"] = [c.get("id") for c in candidates]
            return []

        await _run_retrieval(
            lambda tiers: [
                _mem("cold-good", hybrid.WARM_SAMPLE_THRESHOLD + 0.05),
                _mem("cold-weak", hybrid.WARM_SAMPLE_THRESHOLD - 0.05),
            ],
            select_context=_capture,
        )

        assert "cold-good" in captured["ids"]
        assert "cold-weak" not in captured["ids"]

    @pytest.mark.asyncio
    async def test_cold_query_is_capped_at_the_budget(self):
        """数量上限靠 top_k 传下去, 不能只在合并阶段截断 —— 那样白拉一堆行。"""
        _, captured = await _run_retrieval(lambda tiers: [])
        cold = [top_k for levels, top_k in captured["tiers"] if levels == [3]]
        assert cold == [hybrid.WARM_SAMPLE_BUDGET]

    @pytest.mark.asyncio
    async def test_search_failure_does_not_break_retrieval(self):
        """两层共用一个 future, 它挂了不该让整轮回复失败。"""
        def _boom(tiers):
            raise RuntimeError("pgvector hiccup")

        result, _ = await _run_retrieval(_boom)
        assert result is not None


class TestWarmSampleFeedsTheValueLoop:
    @pytest.mark.asyncio
    async def test_cold_candidates_appear_in_candidate_ids(self):
        """采样到的冷记忆要进 candidate_ids, 否则它拿不到弱使用信号, 永远爬不回来
        —— 那样"消除降级悬崖"就只做了一半。"""
        result, _ = await _run_retrieval(
            lambda tiers: [_mem("cold-1", 0.9)],
        )
        assert "cold-1" in (result.get("candidate_ids") or [])


class TestEmbeddingIsComputedOnce:
    @pytest.mark.asyncio
    async def test_two_tiers_share_a_single_embedding(self):
        """热层和冷层用的是同一段 query。分别嵌两遍会给 Ollama 白压一倍负载 ——
        而且两路并行时 Redis 缓存挡不住同时 miss, 所以缓存救不了。"""
        from app.services.memory.retrieval import vector_search

        calls: list[str] = []

        async def _fake_embed(text):
            calls.append(text)
            return [0.1] * 8

        with patch.object(vector_search, "generate_embedding", _fake_embed), \
             patch.object(
                 vector_search, "search_by_embedding", AsyncMock(return_value=[]),
             ):
            await vector_search.search_similar_tiers(
                "同一句话", "user-1", [([1, 2], 50), ([3], 3)],
            )

        assert calls == ["同一句话"], f"嵌入被算了 {len(calls)} 次"
