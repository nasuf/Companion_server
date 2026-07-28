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
async def test_adjust_side_counts_access_with_snake_case_columns():
    """Raw SQL must use actual DB column names so L2 frequency factor works."""
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side

    mem = _mem(
        importance=0.6,
        updatedAt=datetime.now(UTC),
        createdAt=datetime.now(UTC),
    )
    find_many_mock = AsyncMock(return_value=[mem])
    update_mock = AsyncMock()
    query_raw_mock = AsyncMock(return_value=[{"memory_id": "mem-1", "cnt": 4}])

    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db, \
         patch("app.services.memory.lifecycle.l2_dynamics._track_low_score_streak", AsyncMock(return_value=False)), \
         patch("app.services.memory.lifecycle.l2_dynamics._check_promotion_conditions", AsyncMock(return_value=False)):
        mock_db.usermemory = MagicMock(find_many=find_many_mock, update=update_mock)
        mock_db.query_raw = query_raw_mock

        await _adjust_side("user", "user-1")

    sql = query_raw_mock.await_args.args[0]
    assert "memory_id" in sql
    assert "created_at" in sql
    assert '"memoryId"' not in sql
    assert '"createdAt"' not in sql
    update_data = update_mock.await_args.kwargs["data"]
    # importance is the immutable initial score; the dynamic score goes to
    # its own column (writing it back to importance compounded nightly).
    assert "importance" not in update_data
    assert update_data["currentScore"] == pytest.approx(0.66)
