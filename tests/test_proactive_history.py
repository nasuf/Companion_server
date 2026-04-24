"""proactive_count / proactive_2day 的 DB fallback 测试.

Redis miss/down 时应从 proactive_counters 表读取, 避免计数丢失导致主动消息超发.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _row(count: int, date: str = "20260424"):
    return SimpleNamespace(id=f"row-{date}", count=count, date=date)


@pytest.fixture
def history_mocks():
    """安装 db + redis mock, 返回 (mock_db, mock_redis)."""
    mock_redis = AsyncMock()
    mock_db = MagicMock()
    mock_db.proactivecounter = MagicMock()
    mock_db.proactivecounter.find_unique = AsyncMock(return_value=None)
    mock_db.proactivecounter.find_many = AsyncMock(return_value=[])
    mock_db.proactivecounter.create = AsyncMock()
    mock_db.proactivecounter.update = AsyncMock()

    with patch("app.services.proactive.history.db", mock_db), \
         patch("app.services.proactive.history.get_redis",
               AsyncMock(return_value=mock_redis)):
        yield mock_db, mock_redis


@pytest.mark.asyncio
async def test_can_send_proactive_redis_hit_fast_path(history_mocks):
    from app.services.proactive.history import can_send_proactive

    mock_db, mock_redis = history_mocks
    mock_redis.get = AsyncMock(return_value="2")  # < MAX 3

    assert await can_send_proactive("a1", "u1") is True
    # DB 未被触达
    mock_db.proactivecounter.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_can_send_proactive_redis_miss_falls_back_to_db(history_mocks):
    from app.services.proactive.history import can_send_proactive

    mock_db, mock_redis = history_mocks
    mock_redis.get = AsyncMock(return_value=None)  # Redis miss
    mock_redis.set = AsyncMock()
    mock_db.proactivecounter.find_unique = AsyncMock(return_value=_row(count=1))

    assert await can_send_proactive("a1", "u1") is True
    mock_db.proactivecounter.find_unique.assert_awaited_once()
    # 回填 Redis
    mock_redis.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_can_send_proactive_redis_error_falls_back_to_db(history_mocks):
    """Redis get 抛异常 → DB fallback, 不向上抛."""
    import redis.asyncio as redis_async

    from app.services.proactive.history import can_send_proactive

    mock_db, mock_redis = history_mocks
    mock_redis.get = AsyncMock(side_effect=redis_async.ConnectionError("down"))
    mock_redis.set = AsyncMock(side_effect=redis_async.ConnectionError("down"))
    mock_db.proactivecounter.find_unique = AsyncMock(return_value=_row(count=0))

    assert await can_send_proactive("a1", "u1") is True  # count=0 < MAX 3


@pytest.mark.asyncio
async def test_can_send_proactive_at_limit_returns_false(history_mocks):
    """Redis miss + DB count=3 (MAX) → False."""
    from app.services.proactive.history import can_send_proactive

    mock_db, mock_redis = history_mocks
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_db.proactivecounter.find_unique = AsyncMock(return_value=_row(count=3))

    assert await can_send_proactive("a1", "u1") is False


@pytest.mark.asyncio
async def test_can_send_proactive_2day_sums_today_and_yesterday(history_mocks):
    """2day Redis miss → DB sum(today, yesterday)."""
    from app.services.proactive.history import can_send_proactive_2day

    mock_db, mock_redis = history_mocks
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    # DB 返 2 行: 今天 2, 昨天 1 → sum = 3 < MAX 4 → True
    mock_db.proactivecounter.find_many = AsyncMock(return_value=[
        _row(count=2, date="20260424"),
        _row(count=1, date="20260423"),
    ])

    assert await can_send_proactive_2day("a1", "u1") is True
    # 参数包含 date IN [today, yesterday]
    call_kwargs = mock_db.proactivecounter.find_many.call_args.kwargs
    assert call_kwargs["where"]["agentId"] == "a1"
    assert "in" in call_kwargs["where"]["date"]


@pytest.mark.asyncio
async def test_increment_triggers_db_upsert(history_mocks):
    """incr 后 asyncio.create_task 触发 _upsert_counter, DB 有新行."""
    from app.services.proactive.history import increment_proactive_count

    mock_db, mock_redis = history_mocks
    mock_redis.incr = AsyncMock(return_value=1)
    mock_redis.expire = AsyncMock()
    mock_db.proactivecounter.find_unique = AsyncMock(return_value=None)  # 无存量
    mock_db.proactivecounter.create = AsyncMock()

    await increment_proactive_count("a1", "u1")
    # 让后台 task 跑完
    await asyncio.sleep(0.01)

    mock_db.proactivecounter.create.assert_awaited_once()
    create_data = mock_db.proactivecounter.create.call_args.kwargs["data"]
    assert create_data["agentId"] == "a1"
    assert create_data["userId"] == "u1"
    assert create_data["count"] == 1


@pytest.mark.asyncio
async def test_increment_redis_down_still_upserts_db(history_mocks):
    """Redis incr 抛异常也不阻止 DB upsert (持久化保底)."""
    import redis.asyncio as redis_async

    from app.services.proactive.history import increment_proactive_count

    mock_db, mock_redis = history_mocks
    mock_redis.incr = AsyncMock(side_effect=redis_async.ConnectionError("down"))
    mock_db.proactivecounter.find_unique = AsyncMock(return_value=None)
    mock_db.proactivecounter.create = AsyncMock()

    await increment_proactive_count("a1", "u1")
    await asyncio.sleep(0.01)

    mock_db.proactivecounter.create.assert_awaited_once()
