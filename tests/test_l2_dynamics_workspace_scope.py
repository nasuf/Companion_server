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
    assert update_data["importance"] == pytest.approx(0.66)
