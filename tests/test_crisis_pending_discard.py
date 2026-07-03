"""A3 修复回归: crisis 回合显式丢弃跨消息 pending 状态.

背景: crisis_care_turn=True 时 orchestrator 跳过 preflight (求救信号优先),
旧行为把 pending 矛盾追问/删除确认留在 Redis 等 TTL 过期 — 用户脱离危机后的
第一条消息会被误解析成"对矛盾追问的回答"或"删除确认", 造成误路由.
新契约: 危机回合调 discard_pending_states_for_crisis 显式清除.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.preflight import discard_pending_states_for_crisis

P = "app.services.chat.preflight"


@pytest.mark.asyncio
async def test_discard_clears_both_pending_states():
    with (
        patch(f"{P}.clear_pending_contradiction", new_callable=AsyncMock) as c_contra,
        patch(f"{P}.clear_pending_deletion", new_callable=AsyncMock) as c_del,
    ):
        await discard_pending_states_for_crisis("conv1")

    c_contra.assert_awaited_once_with("conv1")
    c_del.assert_awaited_once_with("conv1")


@pytest.mark.asyncio
async def test_discard_swallows_redis_errors():
    """清理失败不得阻塞危机回复主路径 (危机回复 > 状态整洁)."""
    with (
        patch(
            f"{P}.clear_pending_contradiction",
            new_callable=AsyncMock,
            side_effect=RuntimeError("redis down"),
        ),
        patch(f"{P}.clear_pending_deletion", new_callable=AsyncMock),
    ):
        # 不抛异常即通过
        await discard_pending_states_for_crisis("conv1")


def test_save_pending_action_is_module_level_import():
    """顺手修的 NameError 回归: preflight.py 的 update_reminder_content 分支
    使用 save_pending_action, 必须有模块级 import (旧代码只在另一分支做了
    函数内 import, 走到这个分支直接 NameError)."""
    from app.services.chat import preflight
    from app.services.memory.interaction.deletion import save_pending_action

    assert preflight.save_pending_action is save_pending_action
