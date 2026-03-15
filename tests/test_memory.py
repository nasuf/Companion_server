"""记忆服务单元测试。

测试覆盖：
- 记忆检索组合策略
- 记忆去重
- Prompt格式化
- 记忆存储去重阈值
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.memory.retrieval import (
    _memory_to_dict,
    format_memories_for_prompt,
    retrieve_memories,
)
from app.services.memory.storage import DEDUP_THRESHOLD, is_duplicate


# --- _memory_to_dict ---

class TestMemoryToDict:
    def test_converts_prisma_object(self):
        m = MagicMock()
        m.id = "mem1"
        m.content = "test content"
        m.summary = "test summary"
        m.level = 2
        m.importance = 0.8
        m.type = "事实"
        m.createdAt = "2025-01-01T00:00:00Z"

        result = _memory_to_dict(m, similarity=0.95)
        assert result["id"] == "mem1"
        assert result["summary"] == "test summary"
        assert result["similarity"] == 0.95
        assert result["level"] == 2

    def test_default_similarity(self):
        m = MagicMock()
        m.id = "mem1"
        m.content = "c"
        m.summary = "s"
        m.level = 3
        m.importance = 0.5
        m.type = None
        m.createdAt = "2025-01-01"

        result = _memory_to_dict(m)
        assert result["similarity"] == 0.0


# --- format_memories_for_prompt ---

class TestFormatMemories:
    def test_uses_summary_first(self):
        memories = [
            {"summary": "summary text", "content": "full content"},
        ]
        result = format_memories_for_prompt(memories)
        assert result == ["summary text"]

    def test_falls_back_to_content(self):
        memories = [
            {"summary": None, "content": "full content"},
        ]
        result = format_memories_for_prompt(memories)
        assert result == ["full content"]

    def test_skips_empty(self):
        memories = [
            {"summary": None, "content": None},
            {"summary": "", "content": ""},
            {"summary": "valid", "content": "c"},
        ]
        result = format_memories_for_prompt(memories)
        assert result == ["valid"]

    def test_empty_list(self):
        assert format_memories_for_prompt([]) == []


# --- retrieve_memories dedup ---

@pytest.mark.asyncio
class TestRetrieveMemories:
    async def test_deduplicates_across_sources(self, mock_db):
        """Same memory from semantic + recent should appear only once."""
        mem = MagicMock()
        mem.id = "shared_id"
        mem.content = "shared content"
        mem.summary = "shared"
        mem.level = 2
        mem.importance = 0.9
        mem.type = "事实"
        mem.createdAt = "2025-01-01"

        semantic_results = [{"id": "shared_id", "content": "shared content", "summary": "shared", "similarity": 0.95}]
        mock_db.memory.find_many = AsyncMock(return_value=[mem])

        with (
            patch("app.services.memory.retrieval.search_similar", return_value=semantic_results),
            patch("app.services.memory.retrieval.db", mock_db),
        ):
            results = await retrieve_memories("test query", "user1", semantic_k=5, recent_k=3, important_k=2)
            ids = [r["id"] for r in results]
            assert ids.count("shared_id") == 1

    async def test_empty_query_skips_semantic(self, mock_db):
        mock_db.memory.find_many = AsyncMock(return_value=[])
        with (
            patch("app.services.memory.retrieval.search_similar", new_callable=AsyncMock) as mock_search,
            patch("app.services.memory.retrieval.db", mock_db),
        ):
            await retrieve_memories("", "user1")
            mock_search.assert_not_called()


# --- is_duplicate ---

@pytest.mark.asyncio
class TestIsDuplicate:
    async def test_above_threshold_is_duplicate(self):
        results = [{"similarity": 0.95}]
        with patch("app.services.memory.storage.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is True

    async def test_below_threshold_not_duplicate(self):
        results = [{"similarity": 0.85}]
        with patch("app.services.memory.storage.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is False

    async def test_no_results_not_duplicate(self):
        with patch("app.services.memory.storage.search_by_embedding", return_value=[]):
            assert await is_duplicate("user1", "test", [0.1]) is False

    async def test_string_similarity_parsed(self):
        """Similarity can come as string from raw query."""
        results = [{"similarity": "0.92"}]
        with patch("app.services.memory.storage.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is True


def test_dedup_threshold_value():
    assert DEDUP_THRESHOLD == 0.9
