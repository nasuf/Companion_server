"""Deletion candidate search must be scoped to the user's own memories in the
current workspace — never AI self-memories, never another companion's memories.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.interaction import deletion as deletion_mod


@pytest.mark.asyncio
async def test_find_matching_scopes_source_and_workspace():
    search_calls = {}
    repo_calls = {}

    async def _fake_search(embedding, user_id, top_k=5, workspace_id=None, sources=None):
        search_calls["workspace_id"] = workspace_id
        search_calls["sources"] = sources
        return []

    async def _fake_find_many(*, source, where, take):
        repo_calls["source"] = source
        repo_calls["where"] = where
        return []

    with (
        patch.object(deletion_mod, "generate_embedding", AsyncMock(return_value=[0.1] * 4)),
        patch.object(deletion_mod, "search_by_embedding", _fake_search),
        patch.object(deletion_mod.memory_repo, "find_many", _fake_find_many),
    ):
        await deletion_mod.find_matching_memories("u1", "喝咖啡", workspace_id="w-current")

    assert search_calls["sources"] == ["user"]
    assert search_calls["workspace_id"] == "w-current"
    # literal 匹配也限定 user 表 + 当前 workspace
    assert repo_calls["source"] == "user"
    assert repo_calls["where"]["workspaceId"] == "w-current"


@pytest.mark.asyncio
async def test_delete_by_description_scopes_source_and_workspace():
    search_calls = {}

    async def _fake_search(embedding, user_id, top_k=5, workspace_id=None, sources=None):
        search_calls["workspace_id"] = workspace_id
        search_calls["sources"] = sources
        return []

    with (
        patch.object(deletion_mod, "generate_embedding", AsyncMock(return_value=[0.1] * 4)),
        patch.object(deletion_mod, "search_by_embedding", _fake_search),
    ):
        n = await deletion_mod.delete_memories_by_description(
            "u1", "喝咖啡", workspace_id="w-current",
        )

    assert n == 0
    assert search_calls["sources"] == ["user"]
    assert search_calls["workspace_id"] == "w-current"
