"""_check_promotion_conditions 必须按 workspaceId 隔离 L1 冲突.

同一 user 的不同 agent (workspace) 各自有独立 L1 空间, 不应互相误判冲突."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mem(**kwargs):
    defaults = dict(
        id="mem-1",
        userId="user-1",
        workspaceId="ws-A",
        mainCategory="身份",
        subCategory="姓名",
        content="我叫张三",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_promotion_no_longer_requires_user_emphasis():
    """晋升改为纯值驱动 —— "用户曾说过一定要记住" 不再是一票否决项。

    旧规则把它和分数、频率做 AND, 而 user_emphasized 只在用户说出"一定要记住"
    这类话时才写入, 生产上历史晋升次数为 0 —— 等于根本没有晋升路径, 分层只剩
    下降通道。用户强调仍然有用, 只是改在录入期抬高 importance。
    """
    from app.services.memory.lifecycle.l2_dynamics import _check_promotion_conditions

    mem = _mem()
    count_mock = AsyncMock(return_value=0)  # 从未被强调过
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db:
        mock_db.memorychangelog = MagicMock(count=count_mock)
        mock_db.usermemory = MagicMock(find_many=AsyncMock(return_value=[]))
        allowed = await _check_promotion_conditions(mem, side="user")

    assert allowed is True, "从未被强调的记忆仍被拒绝晋升"
    count_mock.assert_not_called()


@pytest.mark.asyncio
async def test_l1_conflict_query_scoped_by_workspace():
    """L1 冲突查询必须限定在 mem 所属 workspace, 避免阻塞跨 workspace 升级."""
    from app.services.memory.lifecycle.l2_dynamics import _check_promotion_conditions

    mem = _mem(workspaceId="ws-A")
    find_many_mock = AsyncMock(return_value=[])

    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db:
        mock_db.memorychangelog = MagicMock(count=AsyncMock(return_value=1))
        mock_db.usermemory = MagicMock(find_many=find_many_mock)
        await _check_promotion_conditions(mem, side="user")

    where = find_many_mock.call_args.kwargs["where"]
    assert where["userId"] == "user-1"
    assert where["workspaceId"] == "ws-A"
    assert where["level"] == 1


@pytest.mark.asyncio
async def test_null_workspace_passes_through_without_crash():
    """workspaceId=None 的旧记忆应当 IS NULL 过滤, 不崩."""
    from app.services.memory.lifecycle.l2_dynamics import _check_promotion_conditions

    mem = _mem(workspaceId=None)
    find_many_mock = AsyncMock(return_value=[])

    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db:
        mock_db.memorychangelog = MagicMock(count=AsyncMock(return_value=1))
        mock_db.usermemory = MagicMock(find_many=find_many_mock)
        result = await _check_promotion_conditions(mem, side="user")

    assert result is True
    assert find_many_mock.call_args.kwargs["where"]["workspaceId"] is None


@pytest.mark.asyncio
async def test_adjust_side_delegates_to_scoped_decay_backstop():
    from app.services.memory.lifecycle import l2_dynamics
    expected={"total":2,"promoted":0,"demoted":1,"adjusted":1}
    with patch("app.services.memory.lifecycle.lazy_update.sweep_stale_values", new=AsyncMock(return_value={"user":expected})) as sweep:
        assert await l2_dynamics._adjust_side("user","user-1") == expected
    sweep.assert_awaited_once_with(sources=("user",),user_id="user-1")
