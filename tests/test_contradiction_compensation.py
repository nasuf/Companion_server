"""Contradiction resolution must not leave a memory hole.

Phase 4 fix: if the old L1 is archived but the replacement write fails or is
blocked (dedup/taxonomy), un-archive the old memory so the user's core fact
doesn't silently vanish.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from app.services.memory.interaction.contradiction import apply_contradiction_resolution


def _old_mem():
    return MagicMock(
        id="old-id",
        userId="u1",
        importance=0.95,
        content="用户今年 28 岁",
        mainCategory="身份",
        subCategory="年龄",
        source="user",
        workspaceId="ws1",
    )


@pytest.mark.asyncio
async def test_unarchive_when_new_memory_blocked():
    """store_memory 返回 None (dedup/taxonomy 拦截) → 老条目 archive 被撤销."""
    with (
        patch("app.services.memory.interaction.contradiction.memory_repo.find_unique",
              new_callable=AsyncMock, return_value=_old_mem()),
        patch("app.services.memory.interaction.contradiction.memory_repo.update",
              new_callable=AsyncMock) as mock_update,
        patch("app.services.memory.interaction.contradiction.store_memory",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.memory.interaction.contradiction.log_memory_changelog",
              new_callable=AsyncMock),
        patch("app.services.memory.interaction.contradiction.best_effort_create_memory_repair_item",
              new_callable=AsyncMock),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    # 先 archive=True, 失败后 archive=False (补偿) → 两次 update
    archived_calls = [c for c in mock_update.call_args_list
                      if c.kwargs.get("isArchived") is True]
    unarchived_calls = [c for c in mock_update.call_args_list
                        if c.kwargs.get("isArchived") is False]
    assert len(archived_calls) == 1
    assert len(unarchived_calls) == 1


@pytest.mark.asyncio
async def test_unarchive_when_new_memory_write_raises():
    """store_memory 抛异常 → 老条目 archive 被撤销."""
    with (
        patch("app.services.memory.interaction.contradiction.memory_repo.find_unique",
              new_callable=AsyncMock, return_value=_old_mem()),
        patch("app.services.memory.interaction.contradiction.memory_repo.update",
              new_callable=AsyncMock) as mock_update,
        patch("app.services.memory.interaction.contradiction.store_memory",
              new_callable=AsyncMock, side_effect=Exception("db down")),
        patch("app.services.memory.interaction.contradiction.log_memory_changelog",
              new_callable=AsyncMock),
        patch("app.services.memory.interaction.contradiction.best_effort_create_memory_repair_item",
              new_callable=AsyncMock),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    unarchived = [c for c in mock_update.call_args_list
                  if c.kwargs.get("isArchived") is False]
    assert len(unarchived) == 1


@pytest.mark.asyncio
async def test_no_unarchive_on_success():
    """新条目成功写入 → 不撤销 archive (老条目保持归档)."""
    with (
        patch("app.services.memory.interaction.contradiction.memory_repo.find_unique",
              new_callable=AsyncMock, return_value=_old_mem()),
        patch("app.services.memory.interaction.contradiction.memory_repo.update",
              new_callable=AsyncMock) as mock_update,
        patch("app.services.memory.interaction.contradiction.store_memory",
              new_callable=AsyncMock, return_value="new-id"),
        patch("app.services.memory.interaction.contradiction.log_memory_changelog",
              new_callable=AsyncMock),
    ):
        await apply_contradiction_resolution(
            conflict={"conflicting_memory_id": "old-id"},
            analysis={"change_type": "错误", "new_memory": "用户今年 29 岁"},
        )

    unarchived = [c for c in mock_update.call_args_list
                  if c.kwargs.get("isArchived") is False]
    assert len(unarchived) == 0
