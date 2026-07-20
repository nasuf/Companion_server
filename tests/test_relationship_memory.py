"""Phase 2 关系记忆槽: shared-history detection, protected slot, ranking boost,
proactive source wiring, and AI-extraction milestone guidance."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.retrieval.query_patterns import asks_shared_history


class TestSharedHistoryDetection:
    @pytest.mark.parametrize("q", [
        "还记得我们第一次聊天吗",
        "咱们认识多久了",
        "上次跟你聊的那件事怎么样了",
        "我们之前的约定还算数吗",
    ])
    def test_positive(self, q):
        assert asks_shared_history(q) is True

    @pytest.mark.parametrize("q", [
        "我们点外卖吧",          # shared subject but no history hint
        "你去过哪些城市",         # no shared subject
        "第一次去北京是什么体验",   # history hint but no shared subject
        "今天天气怎么样",
        # 2026-07-20: 剔除 "一起"/"多久" 弱线索后, 当下时提议/日常问句不再误判.
        "我们一起点外卖吧",
        "咱们一起去看电影吧",
        "咱们多久能到",
    ])
    def test_negative(self, q):
        assert asks_shared_history(q) is False

    @pytest.mark.parametrize("q", [
        "我俩第一次见面是什么时候",   # 2026-07-20: 补 "我俩" 主语
        "还记得我们最初怎么认识的吗",
    ])
    def test_positive_extra(self, q):
        assert asks_shared_history(q) is True


class TestRelationshipSlot:
    def test_shared_history_query_reserves_interaction_memories(self):
        from app.services.memory.retrieval.context_selector import select_context

        # Low rank_score interaction memory buried under high-score persona rows.
        candidates = [
            {
                "id": f"p{i}", "summary": f"我喜欢人设事实{i}", "importance": 0.9,
                "rank_score": 0.9, "source": "ai",
                "main_category": "偏好", "sub_category": "审美爱好",
                "created_at": "2026-01-01T00:00:00",
            }
            for i in range(10)
        ] + [
            {
                "id": "rel-1",
                "summary": "我和用户第一次深聊了他的家乡，聊到了凌晨",
                "importance": 0.88, "rank_score": 0.2, "source": "ai",
                "main_category": "生活", "sub_category": "交互",
                "created_at": "2026-01-01T00:00:00",
            },
        ]
        result = select_context(candidates, 800, query="还记得我们第一次聊天吗")
        picked = {m.id for m in result}
        assert "rel-1" in picked
        rel = next(m for m in result if m.id == "rel-1")
        assert "保护槽:关系记忆" in (rel.rank_reasons or [])

    def test_non_shared_history_query_no_forced_slot(self):
        from app.services.memory.retrieval.context_selector import select_context

        candidates = [
            {
                "id": "rel-1", "summary": "我和用户第一次深聊",
                "importance": 0.88, "rank_score": 0.05, "source": "ai",
                "main_category": "生活", "sub_category": "交互",
                "created_at": "2026-01-01T00:00:00",
            },
        ]
        result = select_context(candidates, 800, query="今天天气怎么样")
        rel = next((m for m in result if m.id == "rel-1"), None)
        # It may still be picked by final fill, but never via the protected slot.
        if rel is not None:
            assert "保护槽:关系记忆" not in (rel.rank_reasons or [])

    def test_low_importance_interaction_not_protected(self):
        from app.services.memory.retrieval.context_selector import _is_relationship_memory

        assert _is_relationship_memory({
            "source": "ai", "sub_category": "交互", "importance": 0.3,
        }) is False
        assert _is_relationship_memory({
            "source": "ai", "sub_category": "交互", "importance": 0.88,
        }) is True
        assert _is_relationship_memory({
            "source": "user", "sub_category": "交互", "importance": 0.88,
        }) is False


class TestRelationshipRankingBoost:
    def test_interaction_memory_boosted_on_shared_history_query(self):
        from app.services.memory.retrieval.ranking import rank_memory_candidate

        mem = {
            "id": "rel-1",
            "summary": "我和用户第一次深聊了他的家乡",
            "content": "我和用户第一次深聊了他的家乡",
            "importance": 0.85, "similarity": 0.6, "source": "ai",
            "main_category": "生活", "sub_category": "交互",
            "created_at": "2026-01-01T00:00:00",
        }
        boosted, reasons = rank_memory_candidate(mem, "还记得我们第一次聊天吗")
        plain, _ = rank_memory_candidate(mem, "今天吃什么好")
        assert boosted > plain
        assert "关系记忆相关" in reasons


class TestProactiveWiring:
    def test_memory_source_dist_includes_relationship_and_sums_to_one(self):
        from app.services.proactive.policy import MEMORY_SOURCE_DIST

        for stage in ("warming", "intimate"):
            dist = MEMORY_SOURCE_DIST[stage]
            assert "relationship" in dist and dist["relationship"] > 0
            assert sum(dist.values()) == pytest.approx(1.0)
        # Cold-start stages have no shared history yet.
        for stage in ("p1_cold", "p2_cold", "cold_start"):
            assert "relationship" not in MEMORY_SOURCE_DIST[stage]

    def test_sender_maps_relationship_to_memory_ai_prompt(self):
        from app.services.proactive.sender import _MEMORY_SOURCES, _PROMPT_KEY_BY_SOURCE

        assert "relationship" in _MEMORY_SOURCES
        assert _PROMPT_KEY_BY_SOURCE[("memory_proactive", "relationship")] == "proactive.memory_ai"

    @pytest.mark.asyncio
    async def test_context_loader_filters_interaction_subcategory(self):
        from app.services.proactive import context as ctx_mod

        find_many = AsyncMock(return_value=[])
        with patch.object(ctx_mod.memory_repo, "find_many", find_many):
            texts, ids = await ctx_mod._load_proactive_memories(
                user_id="u1", workspace_id="ws1", source="relationship",
            )

        assert texts == [] and ids == []
        where = find_many.await_args.kwargs["where"]
        assert where["subCategory"] == "交互"
        assert "level" not in where
        assert find_many.await_args.kwargs["source"] == "ai"


def test_extraction_ai_prompt_teaches_milestones_and_bans_persona():
    from app.services.prompting.defaults import MEMORY_EXTRACTION_AI_PROMPT as p

    assert "生活/交互" in p
    assert "关系里程碑" in p
    assert "不要输出\"偏好\"和\"身份\"类别" in p
    # Old persona examples must be gone (they contradict the pipeline block).
    assert "我喜欢喝红茶" not in p
    assert "我是个程序员" not in p


class TestRelationshipRecallGate:
    """交互 (共同经历) 记忆相似度天然偏低; 用户明确问"我们之间"时, hybrid 对这
    一小类放宽相似度门, 让 ranking boost / 保护槽有机会发挥 (2026-07-20 review)."""

    def _interaction_row(self, sim: float):
        return {
            "id": "hx1",
            "content": "我们第一次聊天你说你叫小伴",
            "summary": "我们第一次聊天的情形",
            "level": 2,
            "importance": 0.7,
            "similarity": sim,
            "source": "ai",
            "main_category": "生活",
            "sub_category": "交互",
        }

    def _patch_common(self, monkeypatch, hybrid_mod):
        monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
        monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
        monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
        monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            hybrid_mod, "search_related_memories_for_query", AsyncMock(return_value=[]),
        )

    @pytest.mark.asyncio
    async def test_low_sim_interaction_survives_shared_history_query(self, monkeypatch):
        from app.services.memory.retrieval import hybrid as hybrid_mod
        self._patch_common(monkeypatch, hybrid_mod)
        # sim 0.42 is below the normal 0.50 gate but above the relationship gate.
        monkeypatch.setattr(
            hybrid_mod, "search_similar",
            AsyncMock(return_value=[self._interaction_row(0.42)]),
        )
        result = await hybrid_mod.hybrid_retrieve(
            "还记得我们第一次聊天吗", "u1", workspace_id="ws1",
        )
        assert result["memories"] and result["memories"][0].id == "hx1"

    @pytest.mark.asyncio
    async def test_low_sim_interaction_dropped_for_ordinary_query(self, monkeypatch):
        from app.services.memory.retrieval import hybrid as hybrid_mod
        self._patch_common(monkeypatch, hybrid_mod)
        monkeypatch.setattr(
            hybrid_mod, "search_similar",
            AsyncMock(return_value=[self._interaction_row(0.42)]),
        )
        # Not a shared-history query → normal 0.50 gate applies → dropped.
        result = await hybrid_mod.hybrid_retrieve(
            "今天晚饭吃什么好", "u1", workspace_id="ws1",
        )
        assert not result["memories"]
