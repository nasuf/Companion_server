"""Owner-boundary isolation in dedup / deletion vector search.

search_by_embedding must be scopeable to a single owner so user-side dedup
never matches an AI self-memory (which would drop or mis-route data), and
deletion only ever touches the user's own memories.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.retrieval import vector_search as vs


@pytest.mark.asyncio
async def test_search_default_queries_both_tables():
    captured = {}

    async def _fake_query_raw(sql, *args):
        captured["sql"] = sql
        return []

    with (
        patch.object(vs.db, "query_raw", _fake_query_raw),
        patch.object(vs, "resolve_workspace_id", AsyncMock(return_value="w1")),
    ):
        await vs.search_by_embedding([0.1] * 4, "u1")

    assert "memories_user" in captured["sql"]
    assert "memories_ai" in captured["sql"]
    assert "UNION ALL" in captured["sql"]


@pytest.mark.asyncio
async def test_search_user_scope_excludes_ai_table():
    captured = {}

    async def _fake_query_raw(sql, *args):
        captured["sql"] = sql
        return []

    with (
        patch.object(vs.db, "query_raw", _fake_query_raw),
        patch.object(vs, "resolve_workspace_id", AsyncMock(return_value="w1")),
    ):
        await vs.search_by_embedding([0.1] * 4, "u1", sources=["user"])

    assert "memories_user" in captured["sql"]
    assert "memories_ai" not in captured["sql"]
    assert "UNION ALL" not in captured["sql"]


@pytest.mark.asyncio
async def test_search_ai_scope_excludes_user_table():
    captured = {}

    async def _fake_query_raw(sql, *args):
        captured["sql"] = sql
        return []

    with (
        patch.object(vs.db, "query_raw", _fake_query_raw),
        patch.object(vs, "resolve_workspace_id", AsyncMock(return_value="w1")),
    ):
        await vs.search_by_embedding([0.1] * 4, "u1", sources=["ai"])

    assert "memories_ai" in captured["sql"]
    assert "memories_user" not in captured["sql"]


@pytest.mark.asyncio
async def test_search_empty_sources_falls_back_to_both():
    captured = {}

    async def _fake_query_raw(sql, *args):
        captured["sql"] = sql
        return []

    with (
        patch.object(vs.db, "query_raw", _fake_query_raw),
        patch.object(vs, "resolve_workspace_id", AsyncMock(return_value="w1")),
    ):
        await vs.search_by_embedding([0.1] * 4, "u1", sources=[])

    assert "memories_user" in captured["sql"]
    assert "memories_ai" in captured["sql"]


@pytest.mark.asyncio
async def test_find_duplicate_id_scopes_to_source():
    """find_duplicate_id 必须把 source 透传给 search_by_embedding."""
    from app.services.memory.storage import persistence as pstore

    seen = {}

    async def _fake_search(embedding, user_id, top_k=5, workspace_id=None, sources=None):
        seen["sources"] = sources
        return []

    with patch.object(pstore, "search_by_embedding", _fake_search):
        await pstore.find_duplicate_id("u1", "喝水", [0.1] * 4, source="user")

    assert seen["sources"] == ["user"]
