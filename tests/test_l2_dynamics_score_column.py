"""L2 dynamic score must never compound into `importance`.

The old implementation wrote current_score back into importance, so the next
nightly run used last night's product as the base: frequently-accessed rows
inflated ×ff nightly, idle rows decayed ×tf nightly. Now `importance` is the
immutable initial score, the computed score lands in `current_score`, and the
singleton promotion gate refuses a second L1 outright.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
async def test_importance_never_overwritten_on_adjustment():
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side

    mem = _mem(importance=0.6)
    update_mock = AsyncMock()
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db, \
         patch("app.services.memory.lifecycle.l2_dynamics._track_low_score_streak",
               AsyncMock(return_value=False)):
        mock_db.usermemory = MagicMock(
            find_many=AsyncMock(return_value=[mem]), update=update_mock,
        )
        mock_db.query_raw = AsyncMock(return_value=[
            {"memory_id": "mem-1", "cnt": 4},  # ff=1.1 → score 0.66
        ])
        await _adjust_side("user", "user-1")

    data = update_mock.await_args.kwargs["data"]
    assert "importance" not in data
    assert data["currentScore"] == pytest.approx(0.66)


@pytest.mark.asyncio
async def test_no_write_when_score_unchanged():
    """tf=ff=qf=1.0 and currentScore already equals the product → no update."""
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side

    mem = _mem(importance=0.6, currentScore=0.6)
    update_mock = AsyncMock()
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db, \
         patch("app.services.memory.lifecycle.l2_dynamics._track_low_score_streak",
               AsyncMock(return_value=False)):
        mock_db.usermemory = MagicMock(
            find_many=AsyncMock(return_value=[mem]), update=update_mock,
        )
        mock_db.query_raw = AsyncMock(return_value=[
            {"memory_id": "mem-1", "cnt": 1},  # ff=1.0
        ])
        stats = await _adjust_side("user", "user-1")

    update_mock.assert_not_awaited()
    assert stats["adjusted"] == 0


@pytest.mark.asyncio
async def test_current_score_clamped_to_one():
    """importance 0.8 × ff 1.3 × qf 1.1 would exceed 1.0 — must clamp."""
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side

    mem = _mem(importance=0.8)
    update_mock = AsyncMock()
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db, \
         patch("app.services.memory.lifecycle.l2_dynamics._track_low_score_streak",
               AsyncMock(return_value=False)), \
         patch("app.services.memory.lifecycle.l2_dynamics._check_promotion_conditions",
               AsyncMock(return_value=False)):
        mock_db.usermemory = MagicMock(
            find_many=AsyncMock(return_value=[mem]), update=update_mock,
        )
        mock_db.query_raw = AsyncMock(side_effect=[
            [{"memory_id": "mem-1", "cnt": 12}],  # ff=1.3
            [{"memory_id": "mem-1", "corrections": 0, "evidence_links": 4}],  # qf=1.1
        ])
        await _adjust_side("user", "user-1")

    data = update_mock.await_args.kwargs["data"]
    assert data["currentScore"] <= 1.0


@pytest.mark.asyncio
async def test_promotion_writes_l1_band_importance_and_changelog():
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side

    mem = _mem(importance=0.8)
    update_mock = AsyncMock()
    changelog_mock = AsyncMock()
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db, \
         patch("app.services.memory.lifecycle.l2_dynamics._track_low_score_streak",
               AsyncMock(return_value=False)), \
         patch("app.services.memory.lifecycle.l2_dynamics._check_promotion_conditions",
               AsyncMock(return_value=True)), \
         patch("app.services.memory.storage.persistence.log_memory_changelog",
               changelog_mock):
        mock_db.usermemory = MagicMock(
            find_many=AsyncMock(return_value=[mem]), update=update_mock,
        )
        mock_db.query_raw = AsyncMock(side_effect=[
            [{"memory_id": "mem-1", "cnt": 12}],  # ff=1.3 → score 1.0 (clamped)
            [],
        ])
        stats = await _adjust_side("user", "user-1")

    assert stats["promoted"] == 1
    data = update_mock.await_args.kwargs["data"]
    assert data["level"] == 1
    assert 0.85 <= data["importance"] <= 1.0
    changelog_mock.assert_awaited_once()
    assert changelog_mock.await_args.args[2] == "promote"


@pytest.mark.asyncio
async def test_demotion_keeps_importance_untouched():
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side

    mem = _mem(importance=0.55, createdAt=datetime.now(UTC) - timedelta(days=200))
    update_mock = AsyncMock()
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db, \
         patch("app.services.memory.lifecycle.l2_dynamics._track_low_score_streak",
               AsyncMock(return_value=True)):
        mock_db.usermemory = MagicMock(
            find_many=AsyncMock(return_value=[mem]), update=update_mock,
        )
        mock_db.query_raw = AsyncMock(return_value=[])
        stats = await _adjust_side("user", "user-1")

    assert stats["demoted"] == 1
    data = update_mock.await_args.kwargs["data"]
    assert data["level"] == 3
    assert "importance" not in data


@pytest.mark.asyncio
async def test_last_access_read_from_changelog_not_updated_at():
    """updatedAt is touched by this cron itself; the true last access is the
    changelog MAX(created_at). An old access must yield a decayed time factor
    even when updatedAt is fresh."""
    from app.services.memory.lifecycle.l2_dynamics import _adjust_side

    now = datetime.now(UTC)
    mem = _mem(
        importance=0.6,
        updatedAt=now,  # freshly touched by a previous cron write
        createdAt=now - timedelta(days=400),
    )
    update_mock = AsyncMock()
    with patch("app.services.memory.lifecycle.l2_dynamics.db") as mock_db, \
         patch("app.services.memory.lifecycle.l2_dynamics._track_low_score_streak",
               AsyncMock(return_value=False)):
        mock_db.usermemory = MagicMock(
            find_many=AsyncMock(return_value=[mem]), update=update_mock,
        )
        mock_db.query_raw = AsyncMock(side_effect=[
            [{
                "memory_id": "mem-1",
                "cnt": 0,
                "last_access": (now - timedelta(days=100)).isoformat(),
            }],
            [],
        ])
        await _adjust_side("user", "user-1")

    data = update_mock.await_args.kwargs["data"]
    # 100 days since access → tf=0.8 → 0.6 × 0.8 = 0.48 (NOT 0.6 from fresh updatedAt)
    assert data["currentScore"] == pytest.approx(0.48)


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
