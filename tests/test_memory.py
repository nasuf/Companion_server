"""记忆服务单元测试。

测试覆盖：
- 记忆检索组合策略
- 记忆去重
- Prompt格式化
- 记忆存储去重阈值
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.memory.storage.repo import MemoryRecord
from app.services.memory.retrieval.legacy import (
    _memory_to_dict,
    format_memories_for_prompt,
    retrieve_memories,
)
from app.services.memory.storage.persistence import DEDUP_THRESHOLD, is_duplicate, store_memory


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
        mem = MemoryRecord(
            id="shared_id",
            userId="user1",
            type="identity",
            source="user",
            level=2,
            content="shared content",
            summary="shared",
            importance=0.9,
            mentionCount=0,
            isArchived=False,
            occurTime=None,
            createdAt="2025-01-01",
            updatedAt="2025-01-01",
        )

        semantic_results = [{"id": "shared_id", "content": "shared content", "summary": "shared", "similarity": 0.95}]

        with (
            patch("app.services.memory.retrieval.legacy.search_similar", return_value=semantic_results),
            patch("app.services.memory.retrieval.legacy.memory_repo.find_many", new_callable=AsyncMock, side_effect=[[mem], [mem]]),
            patch("app.services.memory.retrieval.legacy.increment_mention_count", new_callable=AsyncMock),
        ):
            results = await retrieve_memories("test query", "user1", semantic_k=5, recent_k=3, important_k=2)
            ids = [r["id"] for r in results]
            assert ids.count("shared_id") == 1

    async def test_empty_query_skips_semantic(self, mock_db):
        with (
            patch("app.services.memory.retrieval.legacy.search_similar", new_callable=AsyncMock) as mock_search,
            patch("app.services.memory.retrieval.legacy.memory_repo.find_many", new_callable=AsyncMock, side_effect=[[], []]),
        ):
            await retrieve_memories("", "user1")
            mock_search.assert_not_called()


# --- is_duplicate ---

@pytest.mark.asyncio
class TestIsDuplicate:
    async def test_above_threshold_is_duplicate(self):
        results = [{"similarity": 0.95}]
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is True

    async def test_below_threshold_not_duplicate(self):
        results = [{"similarity": 0.80}]
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is False

    async def test_no_results_not_duplicate(self):
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=[]):
            assert await is_duplicate("user1", "test", [0.1]) is False

    async def test_string_similarity_parsed(self):
        """Similarity can come as string from raw query."""
        results = [{"similarity": "0.92"}]
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is True


def test_dedup_threshold_value():
    assert DEDUP_THRESHOLD == 0.85


# --- L1 SINGLETON 闸门 (spec §1.5.1) ---


@contextmanager
def _patch_storage_chain(*, existing_l1: list | None = None, create_id: str = "new-id"):
    """patch store_memory 的依赖. existing_l1 控制 SINGLETON find_many 返回值."""
    P = "app.services.memory.storage.persistence"
    existing_records = existing_l1 if existing_l1 is not None else []
    with (
        patch(f"{P}.memory_repo.find_many", new_callable=AsyncMock, return_value=existing_records),
        patch(f"{P}.resolve_workspace_id", new_callable=AsyncMock, return_value="ws1"),
        patch(f"{P}.generate_embedding", new_callable=AsyncMock, return_value=[0.1]) as mock_embed,
        patch(f"{P}.is_duplicate", new_callable=AsyncMock, return_value=False),
        patch(f"{P}.memory_repo.create", new_callable=AsyncMock, return_value=MagicMock(id=create_id)) as mock_create,
        patch(f"{P}.store_embedding", new_callable=AsyncMock),
        patch(f"{P}.log_memory_changelog", new_callable=AsyncMock),
    ):
        yield {"embed": mock_embed, "create": mock_create}


@pytest.mark.asyncio
class TestL1SingletonGate:
    """写入 L1 时, 若 (main, sub) 在 SINGLETON 集合且已有 L1, 拒收新条目.

    场景: extraction LLM 把 '我今年28岁，生日是3月15号' 评分≥85 → level=1,
    跟已有 L1 '我今年28岁' 单看 cosine=0.81 (低于 0.85 阈值) 没被 dedup 拦.
    SINGLETON 闸门作为 dedup 漏掉时的硬兜底.
    """

    async def test_blocks_when_singleton_l1_exists(self):
        """主路径: SINGLETON 子类 + 已有 L1 → 短路拒收, 不调 embed/create."""
        existing = MagicMock(id="existing-id")
        with _patch_storage_chain(existing_l1=[existing]) as mocks:
            result = await store_memory(
                user_id="u1", content="我今年28岁", level=1, importance=0.9,
                main_category="身份", sub_category="年龄", source="ai",
            )
        assert result is None  # 拒收
        mocks["embed"].assert_not_called()  # 短路在 embed 之前, 省嵌入开销 — perf 契约
        mocks["create"].assert_not_called()  # 没写入

    async def test_allows_when_l1_does_not_exist(self):
        """SINGLETON 子类还没 L1 → 正常入库."""
        with _patch_storage_chain(existing_l1=[]) as mocks:
            result = await store_memory(
                user_id="u1", content="我今年28岁", level=1, importance=0.9,
                main_category="身份", sub_category="年龄", source="ai",
            )
        assert result == "new-id"
        mocks["create"].assert_called_once()

    async def test_no_block_for_l2_l3(self):
        """L1 SINGLETON 闸门只在 level=1 触发, L2/L3 写入正常通过.

        注: (ai, L2, 身份) 被 spec §1.5.1 完全禁止 (TAXONOMY_MATRIX 空集),
        所以测试 L2 必须 source=user. user/L2/身份 fallback 是合法的.
        """
        existing = MagicMock(id="other-l1-id")
        with _patch_storage_chain(existing_l1=[existing]) as mocks:
            result = await store_memory(
                user_id="u1", content="用户今年28岁", level=2, importance=0.7,
                main_category="身份", sub_category="年龄", source="user",
            )
        assert result == "new-id"
        mocks["create"].assert_called_once()

    async def test_no_block_for_non_singleton_sub(self):
        """非 SINGLETON 子类 (偏好/饮食喜好) 不走闸门, 多条共存合规."""
        existing = MagicMock(id="other-l1-id")
        with _patch_storage_chain(existing_l1=[existing]) as mocks:
            result = await store_memory(
                user_id="u1", content="我喜欢吃辣", level=1, importance=0.9,
                main_category="偏好", sub_category="饮食喜好", source="user",
            )
        assert result == "new-id"
        mocks["create"].assert_called_once()
