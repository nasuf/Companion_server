"""Retrieval cache must not be poisoned by a failed vector arm.

Production incident: a transient embedding 503 made search_similar raise; the
resulting empty retrieval was cached for 5 minutes, so every follow-up of the
same query hit the poisoned empty cache — one embedding hiccup amplified into
sustained "amnesia" while relevance still classified the topic as strong.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def _patch_common(monkeypatch, hybrid_mod):
    monkeypatch.setattr(hybrid_mod, "cache_retrieval", AsyncMock(return_value=None))
    monkeypatch.setattr(hybrid_mod, "has_explicit_time", lambda _: False)
    monkeypatch.setattr(hybrid_mod, "search_by_time_range", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        hybrid_mod, "search_related_memories_for_query", AsyncMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_vector_failure_skips_cache_write(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    cache_set = AsyncMock()
    _patch_common(monkeypatch, hybrid_mod)
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", cache_set)
    monkeypatch.setattr(
        hybrid_mod, "search_similar",
        AsyncMock(side_effect=RuntimeError("embedding 503")),
    )

    result = await hybrid_mod.hybrid_retrieve("你喜欢什么颜色", "u1", workspace_id="ws1")

    assert result["memories"] is None  # degraded result still returned to caller
    cache_set.assert_not_awaited()  # ← but never cached


@pytest.mark.asyncio
async def test_legit_empty_result_still_cached(monkeypatch):
    """A successful search with zero hits is a real answer and stays cacheable."""
    from app.services.memory.retrieval import hybrid as hybrid_mod

    cache_set = AsyncMock()
    _patch_common(monkeypatch, hybrid_mod)
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", cache_set)
    monkeypatch.setattr(hybrid_mod, "search_similar", AsyncMock(return_value=[]))

    result = await hybrid_mod.hybrid_retrieve("随便聊聊天气", "u1", workspace_id="ws1")

    assert result["memories"] is None
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_successful_retrieval_cached(monkeypatch):
    from app.services.memory.retrieval import hybrid as hybrid_mod

    cache_set = AsyncMock()
    _patch_common(monkeypatch, hybrid_mod)
    monkeypatch.setattr(hybrid_mod, "cache_set_retrieval", cache_set)
    monkeypatch.setattr(hybrid_mod, "search_similar", AsyncMock(return_value=[
        {
            "id": "m1",
            "content": "我喜欢雾霾蓝",
            "summary": "我喜欢雾霾蓝",
            "level": 1,
            "importance": 0.86,
            "similarity": 0.75,
            "source": "ai",
        },
    ]))

    result = await hybrid_mod.hybrid_retrieve("你喜欢什么颜色", "u1", workspace_id="ws1")

    assert result["memories"] and result["memories"][0].id == "m1"
    cache_set.assert_awaited_once()
