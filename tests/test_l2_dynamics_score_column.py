"""L2 dynamic score must never compound into `importance`.

Legacy formula tests must not require rewards during pure decay.
The old implementation wrote current_score back into importance, so the next
nightly run used last night's product as the base: frequently-accessed rows
inflated ×ff nightly, idle rows decayed ×tf nightly. Now `importance` is the
immutable initial score, the computed score lands in `current_score`, and the
singleton promotion gate refuses a second L1 outright.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mem(**kwargs):
    now = datetime.now(UTC)
    defaults = dict(
        id="mem-1",
        userId="user-1",
        workspaceId="ws-A",
        mainCategory="生活",
        subCategory="工作",
        content="用户在做一个副业项目",
        importance=0.6,
        createdAt=now,
        updatedAt=now,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["user", "ai"])
async def test_maintenance_compatibility_entry_delegates_to_pure_decay(side):
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side
    stats = {"scanned": 3, "adjusted": 2, "promoted": 0, "demoted": 1}
    with patch("app.services.memory.lifecycle.lazy_update.sweep_stale_values", AsyncMock(return_value={side: stats})) as sweep:
        assert await _adjust_side(side, "user-1") == stats
    sweep.assert_awaited_once_with(user_id="user-1", sources=(side,))


def test_decay_never_rewards_idle_memory_or_exceeds_initial_value():
    from app.services.memory.lifecycle.value import decayed_value
    assert decayed_value(0.6, 0) == pytest.approx(0.6)
    assert 0 < decayed_value(0.6, 180) < 0.6
    assert decayed_value(0.6, 240) == pytest.approx(0.3)
    # Real score/importance immutability, concurrency and clocks are also
    # asserted by test_memory_lifecycle_postgres.py on migrated PostgreSQL.


@pytest.mark.asyncio
async def test_singleton_promotion_blocked_even_when_similar():
    """A singleton sub with ANY existing L1 must refuse promotion — the old
    char-overlap heuristic let near-duplicates through, creating a second
    姓名/生日 L1 row that bypassed store_memory's singleton gate."""
    from app.services.memory.lifecycle.l2_dynamics import _check_promotion_conditions

    mem = _mem(
        mainCategory="身份", subCategory="姓名",
        summary="我叫张三啊", content="我叫张三啊",
    )
    existing_l1 = SimpleNamespace(id="l1-1", summary="我叫张三", content="我叫张三")
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db:
        mock_db.memorychangelog = MagicMock(count=AsyncMock(return_value=1))
        mock_db.usermemory = MagicMock(find_many=AsyncMock(return_value=[existing_l1]))
        result = await _check_promotion_conditions(mem, side="user")

    assert result is False


@pytest.mark.asyncio
async def test_non_singleton_promotion_not_blocked_by_l1_presence():
    from app.services.memory.lifecycle.l2_dynamics import _check_promotion_conditions

    mem = _mem(mainCategory="生活", subCategory="工作")
    find_many_mock = AsyncMock(return_value=[SimpleNamespace(id="l1-x")])
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db:
        mock_db.memorychangelog = MagicMock(count=AsyncMock(return_value=1))
        mock_db.usermemory = MagicMock(find_many=find_many_mock)
        result = await _check_promotion_conditions(mem, side="user")

    assert result is True
    find_many_mock.assert_not_awaited()  # non-singleton skips the L1 lookup
