"""Regression: apply_contradiction_resolution.

History:
- v1 bug (2026-04-29 trace 019dd7fa): apply 没用 LLM 输出的 new_memory 字段,
  新事实丢失. 修复: 在 apply 内调 store_memory 写新条目.

- Phase 0.3 (2026-05-07): importance 硬编 0.95 + 老条目仅 demote (留在检索通路)
  → 用户口误/玩笑/被诱导 → 错信息直接污染 L1 永久. 修复:
  * 变化: importance=max(0.85, old-0.05) = L1, 老条目 archive (isArchived=True)
  * 错误: importance=0.7 = L2 (等用户反复提到自然 promote), 老条目 archive
  * 新增: importance=0.85 = L1, 老条目不动 (无冲突)
  * archive 而非 demote: 防止双重事实留在检索通路 ('AI 时而说苏州时而上海')
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_apply_错误_archives_old_writes_l2_new():
    """Phase 0.3: change_type=错误 → archive old + 写 L2 新条目 (importance=0.7).

    用户纠正过去说错的事 → 不直接 L1 (防口误/玩笑污染), 等用户反复提到自然 promote.
    老条目 archive (从检索消失), 不再 demote 留在通路.
    """
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="4c882b2b-old-id",
        userId="u1",
        importance=0.95,
        content="用户今年 28 岁",
        mainCategory="身份",
        subCategory="年龄",
        source="user",
        workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock, return_value="new-29-id",
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.log_memory_changelog",
            new_callable=AsyncMock,
        ),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "4c882b2b-old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    # Phase 0.3: 老条目 archive (isArchived=True), 不再 demote 留 L2
    mock_update.assert_called_once()
    update_kwargs = mock_update.call_args.kwargs
    assert update_kwargs["isArchived"] is True
    assert "level" not in update_kwargs  # 不再改 level
    assert "importance" not in update_kwargs  # 不再改 importance

    # 新条目入 L2 (importance=0.7), 不直接 L1
    mock_store.assert_called_once()
    store_kwargs = mock_store.call_args.kwargs
    assert store_kwargs["content"] == "用户今年 29 岁"
    assert store_kwargs["level"] == 2  # L2, 不是 L1
    assert store_kwargs["importance"] == pytest.approx(0.70, abs=0.01)
    assert store_kwargs["main_category"] == "身份"
    assert store_kwargs["sub_category"] == "年龄"
    assert store_kwargs["source"] == "user"
    assert store_kwargs["workspace_id"] == "ws1"


@pytest.mark.asyncio
async def test_apply_变化_archives_old_writes_l1_inheriting_importance():
    """Phase 0.3: change_type=变化 → archive old + 写 L1 (importance ≈ old-0.05).

    用户正常生活更新 (搬家/换工作), 直接 L1, 微降 0.05 标记 transition.
    """
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="old-id", userId="u1", importance=0.95,
        content="住在苏州",
        mainCategory="身份", subCategory="居住地",
        source="user", workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock, return_value="new-id",
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.log_memory_changelog",
            new_callable=AsyncMock,
        ),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "变化", "new_memory": "搬到上海"},
        )

    # 老条目 archive
    update_kwargs = mock_update.call_args.kwargs
    assert update_kwargs["isArchived"] is True

    # 新条目 L1 (importance=0.90 = max(0.85, 0.95-0.05))
    store_kwargs = mock_store.call_args.kwargs
    assert store_kwargs["level"] == 1
    assert store_kwargs["importance"] == pytest.approx(0.90, abs=0.01)


@pytest.mark.parametrize(
    ("old_content", "updated_memory", "main_category", "sub_category"),
    [
        ("用户叫花卷", "用户叫馒头", "身份", "姓名"),
        ("用户住在苏州", "用户住在上海", "身份", "居住地"),
    ],
)
@pytest.mark.asyncio
async def test_apply_变化_uses_updated_memory_when_new_memory_empty(
    old_content, updated_memory, main_category, sub_category,
):
    """矛盾分析常把替换事实放在 updated_memory；不能 archive 旧条目后丢新事实."""
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="old-memory-id", userId="u1", importance=0.9,
        content=old_content,
        mainCategory=main_category, subCategory=sub_category,
        source="user", workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock, return_value="new-name-id",
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.log_memory_changelog",
            new_callable=AsyncMock,
        ),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-memory-id"},
            analysis={
                "change_type": "变化",
                "updated_memory": updated_memory,
                "new_memory": "",
                "new_memory_main_category": "",
                "new_memory_sub_category": "",
            },
        )

    assert mock_update.call_args.kwargs["isArchived"] is True
    mock_store.assert_called_once()
    store_kwargs = mock_store.call_args.kwargs
    assert store_kwargs["content"] == updated_memory
    assert store_kwargs["main_category"] == main_category
    assert store_kwargs["sub_category"] == sub_category
    assert store_kwargs["level"] == 1


@pytest.mark.asyncio
async def test_apply_skips_new_when_new_memory_empty():
    """new_memory 空字符串 → 只 archive 老条目, 不调 store_memory."""
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="old-id", userId="u1", importance=0.9,
        content="住在苏州",
        mainCategory="身份", subCategory="居住地",
        source="user", workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock,
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.log_memory_changelog",
            new_callable=AsyncMock,
        ),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "变化", "new_memory": ""},
        )

    # 老条目 archive 仍发生
    mock_update.assert_called_once()
    assert mock_update.call_args.kwargs["isArchived"] is True
    # 没新条目可写
    mock_store.assert_not_called()


@pytest.mark.asyncio
async def test_apply_新增_writes_new_with_llm_categories():
    """change_type=新增 → 老 L1 不 demote, 用 LLM 给的类目写新条目.

    spec §4.4: "新增: 新内容作为新增记忆按正常流程拆分/打分/存对应层级 (可与
    原L1共存)". 之前代码完全不操作 (注释说由 extraction pipeline 处理), 但
    extraction 跑在用户的"确认"消息上抽不到新事实, 同 bug 跟变化/错误 case.
    """
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="独生女-id", userId="u1", importance=0.95,
        mainCategory="身份", subCategory="亲属关系",  # 老条目"独生女"也归亲属关系
        source="user", workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock, return_value="new-mei-id",
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.log_memory_changelog",
            new_callable=AsyncMock,
        ),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "独生女-id"},
            analysis={
                "change_type": "新增",
                "new_memory": "用户有个妹妹叫小芳",
                "new_memory_main_category": "身份",
                "new_memory_sub_category": "亲属关系",
            },
        )

    # 新增 case: 老 L1 NOT modified (无冲突, 共存)
    mock_update.assert_not_called()
    # 但新条目要入库
    mock_store.assert_called_once()
    kwargs = mock_store.call_args.kwargs
    assert kwargs["content"] == "用户有个妹妹叫小芳"
    assert kwargs["level"] == 1  # L1 (importance=0.85)
    assert kwargs["importance"] == pytest.approx(0.85, abs=0.01)
    assert kwargs["main_category"] == "身份"  # 用 LLM 给的, 不是老条目
    assert kwargs["sub_category"] == "亲属关系"


@pytest.mark.asyncio
async def test_apply_变化_prefers_llm_categories_over_old():
    """change_type=变化 + LLM 给了 new categories → 用 LLM 的, 不复用老条目.

    一般 case 变化 LLM 应该给跟老一样的类目 (e.g. 28岁→29岁 都是 身份/年龄).
    若 LLM 给不一样的 (罕见, 可能 LLM 误判), 还是用 LLM 给的, 让 taxonomy
    resolver 兜底校正. 老条目类目作为 fallback 仅在 LLM 没给时用.
    """
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="old-id", userId="u1", importance=0.95,
        mainCategory="身份", subCategory="年龄",
        source="user", workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock, return_value="new-id",
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.log_memory_changelog",
            new_callable=AsyncMock,
        ),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={
                "change_type": "变化",
                "new_memory": "用户今年 29 岁",
                "new_memory_main_category": "身份",
                "new_memory_sub_category": "年龄",
            },
        )

    kwargs = mock_store.call_args.kwargs
    assert kwargs["main_category"] == "身份"
    assert kwargs["sub_category"] == "年龄"


@pytest.mark.asyncio
async def test_apply_falls_back_to_old_categories_when_llm_omits():
    """LLM 没输出 new_memory_main/sub_category → fallback 复用老条目类目."""
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="old-id", userId="u1", importance=0.95,
        mainCategory="身份", subCategory="年龄",
        source="user", workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock, return_value="new-id",
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.log_memory_changelog",
            new_callable=AsyncMock,
        ),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
            # 没传 new_memory_main_category / new_memory_sub_category
        )

    kwargs = mock_store.call_args.kwargs
    assert kwargs["main_category"] == "身份"  # fallback 老的
    assert kwargs["sub_category"] == "年龄"


@pytest.mark.asyncio
async def test_apply_handles_old_mem_not_found():
    """老条目已被删 → 静默退出, 不报错."""
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=None,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock,
        ) as mock_store,
        patch(
            "app.services.memory.interaction.contradiction.best_effort_create_memory_repair_item",
            new_callable=AsyncMock,
        ) as mock_repair,
    ):
        # Should not raise
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "ghost-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    mock_update.assert_not_called()
    mock_store.assert_not_called()
    mock_repair.assert_awaited_once()
    assert mock_repair.await_args.kwargs["source_type"] == "contradiction_missing_old_memory"


@pytest.mark.asyncio
async def test_apply_logs_warning_when_store_returns_none():
    """store_memory 返回 None (dedup hit / taxonomy block) → 记 warning 不崩."""
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="old-id", userId="u1", importance=0.9,
        mainCategory="身份", subCategory="年龄",
        source="user", workspaceId="ws1",
    )

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock, return_value=old_mem,
        ),
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock, return_value=None,  # 模拟 dedup 命中
        ),
        patch(
            "app.services.memory.interaction.contradiction.logger.warning",
        ) as mock_warning,
        patch(
            "app.services.memory.interaction.contradiction.best_effort_create_memory_repair_item",
            new_callable=AsyncMock,
        ) as mock_repair,
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    # 应该有 warning 日志, 标识"new_blocked" (Phase 0.3 重命名 outcome)
    assert mock_warning.called
    found = False
    for call in mock_warning.call_args_list:
        msg = call.args[0] if call.args else ""
        extra = call.kwargs.get("extra", {})
        if (
            "new_memory" in msg.lower() or "未入库" in msg
            or extra.get("outcome") in ("new_blocked", "new_l1_blocked")
        ):
            found = True
            break
    assert found, "must log warning about new memory not being stored"
    mock_repair.assert_awaited_once()
    assert mock_repair.await_args.kwargs["source_type"] == "contradiction_new_memory_blocked"
