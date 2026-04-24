"""_check_promotion_conditions 必须按 workspaceId 隔离 L1 冲突.

同一 user 的不同 agent (workspace) 各自有独立 L1 空间, 不应互相误判冲突."""

from __future__ import annotations

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
        summary="我叫张三",
        content="我叫张三",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_emphasis_count_query_scoped_by_user_and_workspace():
    from app.services.memory.lifecycle.l2_dynamics import _check_promotion_conditions

    mem = _mem()
    count_mock = AsyncMock(return_value=0)
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db:
        mock_db.memorychangelog = MagicMock(count=count_mock)
        await _check_promotion_conditions(mem, side="user")

    where = count_mock.call_args.kwargs["where"]
    assert where["memoryId"] == "mem-1"
    assert where["operation"] == "user_emphasized"
    assert where["userId"] == "user-1"
    assert where["workspaceId"] == "ws-A"


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
