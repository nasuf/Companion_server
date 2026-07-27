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
from app.services.memory.storage.reconciliation import (
    ReconciliationDecision,
    resolve_memory_write,
)


# --- _memory_to_dict ---

class TestMemoryToDict:
    def test_converts_prisma_object(self):
        m = MagicMock()
        m.id = "mem1"
        m.content = "test content"
        m.level = 2
        m.importance = 0.8
        m.type = "事实"
        m.createdAt = "2025-01-01T00:00:00Z"

        result = _memory_to_dict(m, similarity=0.95)
        assert result["id"] == "mem1"
        assert result["content"] == "test content"
        assert result["similarity"] == 0.95
        assert result["level"] == 2

    def test_default_similarity(self):
        m = MagicMock()
        m.id = "mem1"
        m.content = "c"
        m.level = 3
        m.importance = 0.5
        m.type = None
        m.createdAt = "2025-01-01"

        result = _memory_to_dict(m)
        assert result["similarity"] == 0.0


# --- format_memories_for_prompt ---

class TestFormatMemories:
    def test_ignores_legacy_summary_key(self):
        """`content` is the single source of truth; a stray legacy `summary`
        key must never win over it."""
        memories = [
            {"summary": "summary text", "content": "full content"},
        ]
        result = format_memories_for_prompt(memories)
        assert result == ["full content"]

    def test_uses_content(self):
        memories = [
            {"content": "full content"},
        ]
        result = format_memories_for_prompt(memories)
        assert result == ["full content"]

    def test_skips_empty(self):
        memories = [
            {"content": None},
            {"content": ""},
            {"content": "valid"},
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
            importance=0.9,
            mentionCount=0,
            isArchived=False,
            occurTime=None,
            createdAt="2025-01-01",
            updatedAt="2025-01-01",
        )

        semantic_results = [{"id": "shared_id", "content": "shared content", "similarity": 0.95}]

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
        results = [{"id": "mem-existing", "similarity": 0.95}]
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is True

    async def test_below_threshold_not_duplicate(self):
        results = [{"id": "mem-other", "similarity": 0.80}]
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is False

    async def test_no_results_not_duplicate(self):
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=[]):
            assert await is_duplicate("user1", "test", [0.1]) is False

    async def test_string_similarity_parsed(self):
        """Similarity can come as string from raw query."""
        results = [{"id": "mem-existing", "similarity": "0.92"}]
        with patch("app.services.memory.storage.persistence.search_by_embedding", return_value=results):
            assert await is_duplicate("user1", "test", [0.1]) is True


def test_dedup_threshold_value():
    assert DEDUP_THRESHOLD == 0.85


# --- write reconciliation ---


def _record(
    *,
    id: str,
    content: str,
    source: str = "ai",
    main: str = "身份",
    sub: str = "宠物",
    level: int = 1,
    importance: float = 0.85,
    provenance: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=id,
        userId="u1",
        type="identity",
        source=source,
        level=level,
        content=content,
        importance=importance,
        mentionCount=0,
        isArchived=False,
        occurTime=None,
        createdAt="2026-01-01",
        updatedAt="2026-01-01",
        mainCategory=main,
        subCategory=sub,
        workspaceId="ws1",
        provenance=provenance,
    )


@pytest.mark.asyncio
class TestMemoryReconciliation:
    async def test_drops_new_memory_when_existing_covers_it(self):
        existing = _record(
            id="old-pet",
            content="养了一只叫“芝麻”的黑猫，是从灵隐寺附近的草丛里捡来的流浪猫，当时它只有巴掌大。",
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="ai",
                workspace_id="ws1",
                content="我养了一只叫“芝麻”的黑猫",
                embedding=[0.1],
                main_category="身份",
                sub_category="宠物",
                entities=["芝麻", "黑猫"],
                topics=["宠物", "猫"],
            )

        assert decision.action == "drop_duplicate"
        assert decision.existing_id == "old-pet"

    async def test_updates_existing_when_new_memory_is_richer_within_main_category(self):
        existing = _record(
            id="old-pref",
            content="用户喜欢咖啡",
            source="user",
            main="偏好",
            sub="饮食喜好",
            level=2,
            importance=0.7,
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="user",
                workspace_id="ws1",
                content="用户喜欢研究咖啡豆，尤其关注浅烘埃塞豆",
                embedding=[0.1],
                main_category="偏好",
                sub_category="饮食喜好",
                entities=["咖啡豆"],
                topics=["咖啡", "饮食喜好"],
            )

        assert decision.action == "update_existing"
        assert decision.existing_id == "old-pref"

    async def test_does_not_merge_across_main_categories(self):
        existing = _record(
            id="old-job",
            content="用户是一名咖啡师",
            source="user",
            main="身份",
            sub="职业/与经济",
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="user",
                workspace_id="ws1",
                content="用户喜欢咖啡",
                embedding=[0.1],
                main_category="偏好",
                sub_category="饮食喜好",
                entities=["咖啡"],
                topics=["饮食喜好"],
            )

        assert decision.action == "insert_new"

    async def test_does_not_update_across_main_categories_even_when_text_contains_old(self):
        existing = _record(
            id="old-job",
            content="用户是一名咖啡师",
            source="user",
            main="身份",
            sub="职业/与经济",
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="user",
                workspace_id="ws1",
                content="用户是一名咖啡师，也很喜欢咖啡",
                embedding=[0.1],
                main_category="偏好",
                sub_category="饮食喜好",
                entities=["咖啡"],
                topics=["饮食喜好"],
            )

        assert decision.action == "insert_new"

    async def test_singleton_l1_never_updated_by_containment_rule(self):
        """Spec §1.5.1: singleton L1 (姓名/年龄/生日…) 不可被写入期 reconciliation
        改写 — 即使新文本严格包含旧文本 (enrichment containment 命中)."""
        existing = _record(
            id="old-name",
            content="我叫小伴",
            source="ai",
            main="身份",
            sub="姓名",
            level=1,
            importance=0.95,
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="ai",
                workspace_id="ws1",
                content="我叫小伴，大家也叫我昕昕",
                embedding=[0.1],
                main_category="身份",
                sub_category="姓名",
                entities=["小伴"],
                topics=["姓名"],
                allow_llm=False,
            )

        assert decision.action != "update_existing"
        assert decision.action != "merge_existing"

    async def test_singleton_l1_never_mutated_by_llm_adjudication(self):
        """LLM 裁决想 update/merge singleton L1 也必须被拒 (keep separate)."""
        existing = _record(
            id="old-bday",
            content="我生日是2004年3月8日",
            source="ai",
            main="身份",
            sub="生日",
            level=1,
            importance=0.9,
        )
        llm_decision = ReconciliationDecision(
            action="update_existing",
            merged_content="我生日是2004年3月8日，是双鱼座",
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
            patch("app.services.memory.storage.reconciliation._relation", return_value="keep_separate"),
            patch("app.services.memory.storage.reconciliation._related_enough_for_llm", return_value=True),
            patch("app.services.memory.storage.reconciliation._llm_adjudicate", new_callable=AsyncMock, return_value=llm_decision),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="ai",
                workspace_id="ws1",
                content="我生日是2004年3月8日，是双鱼座",
                embedding=[0.1],
                main_category="身份",
                sub_category="生日",
                entities=[],
                topics=["生日"],
            )

        assert decision.action == "insert_new"

    async def test_profile_seed_row_never_updated_even_non_singleton(self):
        """Phase 2 provenance: profile_seed 行是人设 ground truth, 任何类别都
        不可被写入期 reconciliation 改写 — enrichment 分开存."""
        existing = _record(
            id="seed-pet",
            content="我养了一只叫芝麻的黑猫",
            source="ai",
            main="身份",
            sub="宠物",  # 非 singleton 子类
            level=1,
            importance=0.85,
            provenance="profile_seed",
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="ai",
                workspace_id="ws1",
                content="我养了一只叫芝麻的黑猫，是三年前在小区捡的",
                embedding=[0.1],
                main_category="身份",
                sub_category="宠物",
                entities=["芝麻"],
                topics=["宠物"],
                allow_llm=False,
            )

        assert decision.action != "update_existing"
        assert decision.action != "merge_existing"

    async def test_non_singleton_l1_enrichment_still_updates(self):
        """非 singleton L1 (如 宠物) 的 containment enrichment 政策保留."""
        existing = _record(
            id="old-pet",
            content="我养了一只叫芝麻的黑猫",
            source="ai",
            main="身份",
            sub="宠物",
            level=1,
            importance=0.85,
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="ai",
                workspace_id="ws1",
                content="我养了一只叫芝麻的黑猫，是三年前在小区捡的",
                embedding=[0.1],
                main_category="身份",
                sub_category="宠物",
                entities=["芝麻"],
                topics=["宠物"],
                allow_llm=False,
            )

        assert decision.action == "update_existing"
        assert decision.existing_id == "old-pet"

    async def test_ambiguous_related_pair_can_use_llm_merge_decision(self):
        existing = _record(
            id="old-life",
            content="用户周末经常去花鸟市场逛干花摊",
            source="user",
            main="生活",
            sub="生活",
            level=2,
            importance=0.7,
        )
        llm_decision = ReconciliationDecision(
            action="merge_existing",
            merged_content="用户周末会去花鸟市场逛干花摊，也买过多肉植物",
        )
        with (
            patch("app.services.memory.storage.reconciliation.memory_repo.find_many", new_callable=AsyncMock, return_value=[existing]),
            patch("app.services.memory.storage.reconciliation.search_by_embedding", new_callable=AsyncMock, return_value=[]),
            patch("app.services.memory.storage.reconciliation._relation", return_value="keep_separate"),
            patch("app.services.memory.storage.reconciliation._related_enough_for_llm", return_value=True),
            patch("app.services.memory.storage.reconciliation._llm_adjudicate", new_callable=AsyncMock, return_value=llm_decision) as mock_llm,
        ):
            decision = await resolve_memory_write(
                user_id="u1",
                source="user",
                workspace_id="ws1",
                content="用户在花鸟市场买过多肉植物",
                embedding=[0.1],
                main_category="生活",
                sub_category="生活",
                entities=["花鸟市场", "多肉植物"],
                topics=["花鸟市场", "植物"],
            )

        assert decision.action == "merge_existing"
        assert decision.existing_id == "old-life"
        assert decision.merged_content == "用户周末会去花鸟市场逛干花摊，也买过多肉植物"
        mock_llm.assert_awaited_once()

    async def test_store_memory_updates_existing_when_reconciliation_says_update(self):
        existing = _record(
            id="old-pref",
            content="用户喜欢咖啡",
            source="user",
            main="偏好",
            sub="饮食喜好",
            level=2,
            importance=0.7,
        )
        decision = ReconciliationDecision(
            action="update_existing",
            existing_id="old-pref",
            existing_record=existing,
            merged_content="用户喜欢研究咖啡豆，尤其关注浅烘埃塞豆",
        )
        P = "app.services.memory.storage.persistence"
        with (
            patch(f"{P}.resolve_workspace_id", new_callable=AsyncMock, return_value="ws1"),
            patch(f"{P}.generate_embedding", new_callable=AsyncMock, return_value=[0.1]),
            patch(f"{P}.resolve_memory_write", new_callable=AsyncMock, return_value=decision),
            patch(f"{P}.store_embedding", new_callable=AsyncMock) as mock_store_embedding,
            patch(f"{P}.memory_repo.update", new_callable=AsyncMock) as mock_update,
            patch(f"{P}.memory_repo.create", new_callable=AsyncMock) as mock_create,
            patch(f"{P}.log_memory_changelog", new_callable=AsyncMock),
        ):
            result = await store_memory(
                user_id="u1",
                content="用户喜欢研究咖啡豆，尤其关注浅烘埃塞豆",
                level=2,
                importance=0.8,
                main_category="偏好",
                sub_category="饮食喜好",
                source="user",
            )

        assert result == "old-pref"
        mock_store_embedding.assert_awaited_once_with("old-pref", [0.1])
        mock_update.assert_awaited_once()
        mock_create.assert_not_called()

    async def test_store_memory_reembeds_merged_content_when_llm_merges(self):
        existing = _record(
            id="old-life",
            content="用户周末经常去花鸟市场逛干花摊",
            source="user",
            main="生活",
            sub="生活",
            level=2,
            importance=0.7,
        )
        decision = ReconciliationDecision(
            action="merge_existing",
            existing_id="old-life",
            existing_record=existing,
            merged_content="用户周末会去花鸟市场逛干花摊，也买过多肉植物",
        )
        P = "app.services.memory.storage.persistence"
        with (
            patch(f"{P}.resolve_workspace_id", new_callable=AsyncMock, return_value="ws1"),
            patch(f"{P}.generate_embedding", new_callable=AsyncMock, side_effect=[[0.1], [0.2]]) as mock_embed,
            patch(f"{P}.resolve_memory_write", new_callable=AsyncMock, return_value=decision),
            patch(f"{P}.store_embedding", new_callable=AsyncMock) as mock_store_embedding,
            patch(f"{P}.memory_repo.update", new_callable=AsyncMock),
            patch(f"{P}.memory_repo.create", new_callable=AsyncMock) as mock_create,
            patch(f"{P}.log_memory_changelog", new_callable=AsyncMock),
        ):
            result = await store_memory(
                user_id="u1",
                content="用户在花鸟市场买过多肉植物",
                level=2,
                importance=0.8,
                main_category="生活",
                sub_category="生活",
                source="user",
            )

        assert result == "old-life"
        assert mock_embed.await_count == 2
        mock_embed.assert_any_await("用户周末会去花鸟市场逛干花摊，也买过多肉植物")
        mock_store_embedding.assert_awaited_once_with("old-life", [0.2])
        mock_create.assert_not_called()


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
        patch(f"{P}.resolve_memory_write", new_callable=AsyncMock, return_value=ReconciliationDecision(action="insert_new")),
        patch(f"{P}.memory_repo.update", new_callable=AsyncMock) as mock_update,
        patch(f"{P}.memory_repo.create", new_callable=AsyncMock, return_value=MagicMock(id=create_id)) as mock_create,
        patch(f"{P}.store_embedding", new_callable=AsyncMock),
        patch(f"{P}.log_memory_changelog", new_callable=AsyncMock),
    ):
        yield {"embed": mock_embed, "create": mock_create, "update": mock_update}


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


@pytest.mark.asyncio
class TestProvenancePassthrough:
    """Phase 2: store_memory 把合法 provenance 写入行, 非法值落 NULL."""

    async def test_valid_provenance_written(self):
        with _patch_storage_chain() as mocks:
            result = await store_memory(
                user_id="u1", content="用户喜欢咖啡", level=2, importance=0.7,
                main_category="偏好", sub_category="饮食喜好", source="user",
                provenance="user_stated",
            )
        assert result == "new-id"
        create_kwargs = mocks["create"].call_args.kwargs
        assert create_kwargs["provenance"] == "user_stated"

    async def test_invalid_provenance_dropped(self):
        with _patch_storage_chain() as mocks:
            await store_memory(
                user_id="u1", content="用户喜欢咖啡", level=2, importance=0.7,
                main_category="偏好", sub_category="饮食喜好", source="user",
                provenance="made_up_source",
            )
        create_kwargs = mocks["create"].call_args.kwargs
        assert "provenance" not in create_kwargs

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

    async def test_user_singleton_replaces_old_current_value(self):
        """用户侧 L1 singleton 新事实应替换旧当前值，而不是被旧 L1 永久挡住."""
        existing = MagicMock(id="old-name-id")
        existing.content = "用户叫花卷"
        with _patch_storage_chain(existing_l1=[existing]) as mocks:
            result = await store_memory(
                user_id="u1", content="用户叫馒头", level=1, importance=0.9,
                main_category="身份", sub_category="姓名", source="user",
            )
        assert result == "new-id"
        mocks["update"].assert_awaited_once()
        assert mocks["update"].await_args.kwargs["isArchived"] is True
        mocks["create"].assert_called_once()

    async def test_user_singleton_same_text_still_blocked(self):
        """同一条 singleton 文本重复写入仍应短路，避免无意义 churn."""
        existing = MagicMock(id="old-name-id")
        existing.content = "用户叫花卷"
        with _patch_storage_chain(existing_l1=[existing]) as mocks:
            result = await store_memory(
                user_id="u1", content="用户叫花卷", level=1, importance=0.9,
                main_category="身份", sub_category="姓名", source="user",
            )
        assert result is None
        mocks["update"].assert_not_called()
        mocks["create"].assert_not_called()


# --- A2: L1 SINGLETON 写锁 (TOCTOU 修复) ---


@pytest.mark.asyncio
class TestSingletonWriteLock:
    """singleton 检查→create 之间隔着 embedding + reconciliation (秒级窗口),
    并发写会双双通过 find_many 检查 → L1 重复. 修复后整段被 per-类目
    distributed_lock 串行化.
    """

    async def test_concurrent_singleton_writes_only_one_insert(self):
        """并发两条同类目 singleton 写入 → 只有 1 条 create, 另一条被闸门拦下.

        用 asyncio.Lock 模拟 distributed_lock 的互斥语义; generate_embedding
        故意 sleep 拉宽 TOCTOU 窗口 — 修复前该场景两条都会 create (生产
        case 2026-05-07 复现), 修复后 create 恰好 1 次.
        """
        import asyncio
        from contextlib import asynccontextmanager

        P = "app.services.memory.storage.persistence"
        created: list[str] = []
        mutex = asyncio.Lock()

        @asynccontextmanager
        async def fake_lock(name, **kwargs):
            assert name.startswith("singleton_write:")
            async with mutex:
                yield True

        async def fake_find_many(*args, **kwargs):
            return [MagicMock(id="m1")] if created else []

        async def fake_create(**kwargs):
            created.append(f"m{len(created) + 1}")
            return MagicMock(id=created[-1])

        async def slow_embed(content):
            await asyncio.sleep(0.05)  # 拉宽检查→create 的竞争窗口
            return [0.1]

        with (
            patch(f"{P}.distributed_lock", fake_lock),
            patch(f"{P}.memory_repo.find_many", side_effect=fake_find_many),
            patch(f"{P}.resolve_workspace_id", new_callable=AsyncMock, return_value="ws1"),
            patch(f"{P}.generate_embedding", side_effect=slow_embed),
            patch(f"{P}.resolve_memory_write", new_callable=AsyncMock,
                  return_value=ReconciliationDecision(action="insert_new")),
            patch(f"{P}.memory_repo.create", side_effect=fake_create),
            patch(f"{P}.memory_repo.update", new_callable=AsyncMock),
            patch(f"{P}.store_embedding", new_callable=AsyncMock),
            patch(f"{P}.log_memory_changelog", new_callable=AsyncMock),
        ):
            r1, r2 = await asyncio.gather(
                store_memory(
                    user_id="u1", content="我今年28岁", level=1, importance=0.9,
                    main_category="身份", sub_category="年龄", source="user",
                ),
                store_memory(
                    user_id="u1", content="我今年28岁，生日是3月15号", level=1,
                    importance=0.9, main_category="身份", sub_category="年龄",
                    source="user",
                ),
            )

        assert len(created) == 1  # 恰好一条入库
        assert sorted([r1, r2], key=lambda x: (x is None, x)) == ["m1", None]

    async def test_singleton_lock_contention_drops_write(self):
        """等锁超时 (另一写入者持锁 >10s) → 按重复丢弃, 不 create."""
        from app.services.runtime.distributed_lock import DistributedLockNotAcquired

        P = "app.services.memory.storage.persistence"

        def raise_not_acquired(*args, **kwargs):
            raise DistributedLockNotAcquired("lock held")

        with _patch_storage_chain(existing_l1=[]) as mocks:
            with patch(f"{P}.distributed_lock", raise_not_acquired):
                result = await store_memory(
                    user_id="u1", content="我今年28岁", level=1, importance=0.9,
                    main_category="身份", sub_category="年龄", source="user",
                )
        assert result is None
        mocks["create"].assert_not_called()

    async def test_non_singleton_write_does_not_touch_lock(self):
        """非 singleton 写入不应产生锁开销 (热路径性能契约)."""
        P = "app.services.memory.storage.persistence"
        with _patch_storage_chain(existing_l1=[]) as mocks:
            with patch(f"{P}.distributed_lock") as mock_lock:
                result = await store_memory(
                    user_id="u1", content="我喜欢吃辣", level=2, importance=0.7,
                    main_category="偏好", sub_category="饮食喜好", source="user",
                )
        assert result == "new-id"
        mock_lock.assert_not_called()
        mocks["create"].assert_called_once()


@pytest.mark.asyncio
async def test_store_memory_skip_reconciliation_bypasses_adjudication():
    """skip_reconciliation=True 必须绕过 resolve_memory_write, 强制插入新行 —
    整合摘要靠它避免被 update_existing 并进无关记忆 (2026-07-20 review)."""
    from types import SimpleNamespace
    from app.services.memory.storage import persistence as pers
    from app.services.memory.taxonomy import resolve_taxonomy

    tax = resolve_taxonomy(
        main_category="生活", sub_category="其他",
        legacy_type=None, source="ai", level=3,
    )
    resolve_spy = AsyncMock()
    with (
        patch.object(pers, "resolve_taxonomy", return_value=tax),
        patch.object(pers, "resolve_workspace_id", AsyncMock(return_value="ws1")),
        patch.object(pers, "generate_embedding", AsyncMock(return_value=[0.1] * 4)),
        patch.object(pers, "store_embedding", AsyncMock()),
        patch.object(pers, "resolve_memory_write", resolve_spy),
        patch.object(pers, "log_memory_changelog", AsyncMock()),
        patch.object(
            pers.memory_repo, "create",
            AsyncMock(return_value=SimpleNamespace(id="new-row-1")),
        ),
    ):
        new_id = await pers.store_memory(
            user_id="u1", content="十月里我常在早晨散步、买咖啡",
            level=3, importance=0.4, main_category="生活", sub_category="其他",
            source="ai", workspace_id="ws1",
            provenance="consolidated", skip_reconciliation=True,
        )

    assert new_id == "new-row-1"
    resolve_spy.assert_not_awaited()  # reconciliation entirely bypassed
