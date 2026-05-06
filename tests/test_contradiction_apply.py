"""Regression: apply_contradiction_resolution 写新 L1 (spec §4.4 step 2).

生产 bug 复现 (2026-04-29 trace 019dd7fa, 用户 690d80fc):
  用户说"我其实29岁" 跟现有 L1 "用户今年28岁" 冲突. 流程跑完后:
  - ✓ 老 28岁 L1 → L2 imp 0.75 (apply 已正确降级)
  - ✗ 新 29岁 L1 **未入库** (apply 没用 LLM 输出的 new_memory 字段)

根因: contradiction_analysis prompt 输出 new_memory 字段, 但 apply 函数
之前注释说"由 extraction pipeline 创建" — 但 extraction 跑在用户的"确认"
消息(如"说错了之前")上, 文本不含新事实抽不到. 而且 trace_1 extraction 抽到
29岁后, store_memory 被 L1 SINGLETON gate 拦下 (老 28岁还在 L1).

修复: apply 在 demote 老 L1 后, 用 analysis.new_memory 直接调 store_memory
写新 L1, 复用老条目的 main/sub_category.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_apply_writes_new_l1_for_change_type():
    """change_type=变化/错误 + new_memory 非空 → demote old + write new L1."""
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    old_mem = MagicMock(
        id="4c882b2b-old-id",
        userId="u1",
        importance=0.95,
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
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "4c882b2b-old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    # 老 L1 → L2
    mock_update.assert_called_once()
    update_kwargs = mock_update.call_args.kwargs
    assert update_kwargs["level"] == 2
    assert update_kwargs["importance"] == pytest.approx(0.75, abs=0.01)  # 0.95 - 0.20

    # 新 L1 入库
    mock_store.assert_called_once()
    store_kwargs = mock_store.call_args.kwargs
    assert store_kwargs["content"] == "用户今年 29 岁"
    assert store_kwargs["summary"] == "用户今年 29 岁"
    assert store_kwargs["level"] == 1
    assert store_kwargs["importance"] == 0.95
    assert store_kwargs["main_category"] == "身份"  # 从老条目复用
    assert store_kwargs["sub_category"] == "年龄"
    assert store_kwargs["source"] == "user"
    assert store_kwargs["workspace_id"] == "ws1"


@pytest.mark.asyncio
async def test_apply_skips_new_l1_when_new_memory_empty():
    """new_memory 空字符串 → 只 demote, 不调 store_memory."""
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
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock,
        ) as mock_store,
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "变化", "new_memory": ""},
        )

    mock_update.assert_called_once()
    mock_store.assert_not_called()


@pytest.mark.asyncio
async def test_apply_skips_demote_for_新增():
    """change_type=新增 → 不 demote, 不 写新 L1 (走 extraction pipeline)."""
    from app.services.memory.interaction.contradiction import apply_contradiction_resolution

    with (
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.find_unique",
            new_callable=AsyncMock,
        ) as mock_find,
        patch(
            "app.services.memory.interaction.contradiction.memory_repo.update",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "app.services.memory.interaction.contradiction.store_memory",
            new_callable=AsyncMock,
        ) as mock_store,
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "新增", "new_memory": "用户有个表妹叫小芳"},
        )

    # 新增 case: 老 L1 不动, 新条目由 extraction pipeline 处理
    mock_find.assert_not_called()
    mock_update.assert_not_called()
    mock_store.assert_not_called()


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
    ):
        # Should not raise
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "ghost-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    mock_update.assert_not_called()
    mock_store.assert_not_called()


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
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    # 应该有 warning 日志, 标识"new_l1_blocked"
    assert mock_warning.called
    found = False
    for call in mock_warning.call_args_list:
        msg = call.args[0] if call.args else ""
        extra = call.kwargs.get("extra", {})
        if "new_l1" in msg.lower() or extra.get("outcome") == "new_l1_blocked":
            found = True
            break
    assert found, "must log warning about new_l1 not being stored"
