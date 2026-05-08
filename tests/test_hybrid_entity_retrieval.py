from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_entity_recall_bypasses_vector_miss(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    monkeypatch.setattr(hybrid_mod, "search_similar", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        hybrid_mod,
        "search_related_memories_for_query",
        AsyncMock(return_value=[{
            "id": "mom-hospital",
            "summary": "用户妈妈最近住院，用户很担心她",
            "content": "用户妈妈最近住院，用户很担心她",
            "level": 2,
            "importance": 0.82,
            "source": "user",
            "matched_entity": "妈妈",
        }]),
    )

    result = await hybrid_mod.hybrid_retrieve(
        "妈妈现在怎么样了", "u1", workspace_id="ws1",
    )

    memories = result["memories"]
    assert memories
    assert memories[0].id == "mom-hospital"
    assert "实体命中" in (memories[0].rank_reasons or [])


def test_short_entity_followup_is_not_trivial():
    from app.services.memory.retrieval.hybrid import _is_trivial_message

    assert _is_trivial_message("嗯嗯") is True
    assert _is_trivial_message("妈妈呢") is False


@pytest.mark.asyncio
async def test_hybrid_cache_hit_rehydrates_structured_memories(monkeypatch):
    from app.services.memory.retrieval.context_selector import ClassifiedMemory
    from app.services.memory.retrieval import hybrid as hybrid_mod

    monkeypatch.setattr(
        hybrid_mod,
        "cache_retrieval",
        AsyncMock(return_value={
            "memories": [{
                "id": "m1",
                "text": "用户妈妈最近住院",
                "relevance": "strong",
                "score": 0.82,
                "importance": 0.9,
                "similarity": 0.78,
                "rank_reasons": ["实体命中"],
                "source": "user",
            }],
            "memory_strings": ["用户妈妈最近住院"],
            "graph_context": None,
        }),
    )
    search_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(hybrid_mod, "search_similar", search_mock)

    result = await hybrid_mod.hybrid_retrieve(
        "妈妈呢", "u1", workspace_id="ws1",
    )

    assert search_mock.await_count == 0
    assert isinstance(result["memories"][0], ClassifiedMemory)
    assert result["memories"][0].id == "m1"
    assert result["memory_strings"] == ["用户妈妈最近住院"]


@pytest.mark.asyncio
async def test_hybrid_cache_write_serializes_classified_memories(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    captured: dict = {}

    async def _capture_cache_set(key, user_id, result, workspace_id=None):
        captured["result"] = result

    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", _capture_cache_set)
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "search_related_memories_for_query", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        hybrid_mod,
        "search_similar",
        AsyncMock(return_value=[{
            "id": "m1",
            "summary": "用户妈妈最近住院",
            "content": "用户妈妈最近住院",
            "level": 2,
            "importance": 0.9,
            "similarity": 0.8,
            "source": "user",
        }]),
    )

    await hybrid_mod.hybrid_retrieve("妈妈", "u1", workspace_id="ws1")

    cached_memory = captured["result"]["memories"][0]
    assert isinstance(cached_memory, dict)
    assert cached_memory["id"] == "m1"
    assert cached_memory["text"] == "用户妈妈最近住院"


@pytest.mark.asyncio
async def test_hybrid_retrieve_passes_effective_query_to_selector(monkeypatch):
    from app.services.memory.retrieval.context_selector import ClassifiedMemory
    from app.services.memory.retrieval import hybrid as hybrid_mod

    captured: dict = {}

    def _select_context(candidates, token_budget, max_items=10, query=None):
        captured["query"] = query
        return [
            ClassifiedMemory(
                id="m1",
                text="用户的直属领导叫陈姐",
                relevance="strong",
                score=0.8,
                source="user",
            )
        ]

    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "search_related_memories_for_query", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "select_context", _select_context)
    monkeypatch.setattr(
        hybrid_mod,
        "search_similar",
        AsyncMock(return_value=[{
            "id": "m1",
            "summary": "用户的直属领导叫陈姐",
            "content": "用户的直属领导叫陈姐",
            "level": 2,
            "importance": 0.8,
            "similarity": 0.7,
            "source": "user",
        }]),
    )

    await hybrid_mod.hybrid_retrieve(
        "她叫什么",
        "u1",
        workspace_id="ws1",
        enhanced_query="用户的直属领导叫什么",
    )

    assert captured["query"] == "用户的直属领导叫什么"


@pytest.mark.asyncio
async def test_entity_recall_keeps_l1_l2_scope(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    captured: dict = {}

    async def _capture_entity_recall(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    monkeypatch.setattr(hybrid_mod, "search_similar", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid_mod, "search_related_memories_for_query", _capture_entity_recall)

    await hybrid_mod.hybrid_retrieve("妈妈呢", "u1", workspace_id="ws1")

    assert captured["levels"] == [1, 2]
    assert captured["workspace_id"] == "ws1"


@pytest.mark.asyncio
async def test_entity_recall_marks_existing_vector_candidate(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", AsyncMock())
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        hybrid_mod,
        "search_similar",
        AsyncMock(return_value=[{
            "id": "mom-hospital",
            "summary": "用户妈妈最近住院",
            "content": "用户妈妈最近住院",
            "level": 2,
            "importance": 0.8,
            "similarity": 0.52,
            "source": "user",
        }]),
    )
    monkeypatch.setattr(
        hybrid_mod,
        "search_related_memories_for_query",
        AsyncMock(return_value=[{
            "id": "mom-hospital",
            "summary": "用户妈妈最近住院",
            "content": "用户妈妈最近住院",
            "level": 2,
            "importance": 0.8,
            "source": "user",
            "matched_entity": "妈妈",
        }]),
    )

    result = await hybrid_mod.hybrid_retrieve(
        "妈妈呢", "u1", workspace_id="ws1",
    )

    memories = result["memories"]
    assert memories
    assert len(memories) == 1
    assert "实体命中" in (memories[0].rank_reasons or [])
    assert memories[0].similarity == hybrid_mod._ENTITY_RECALL_SIMILARITY
