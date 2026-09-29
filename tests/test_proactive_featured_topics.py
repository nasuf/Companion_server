"""featured_topics.py 单测 (2026-09-14): 最近 featured 过的热点标题按 workspace 排除.

修 bug: admin 反复触发主动消息, DailyHot 热榜前几名短时间内不变 + V3 分类器
100% 确定性, 结果永远推"同一件事". 加"最近 featured 排除", 按 workspace_id
(即 user × agent) 隔离, 6h TTL. 不影响其他用户对同一热点的首次曝光.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest



class TestFeaturedTopicsRedis:
    """featured_topics.py Redis CRUD."""

    @pytest.mark.asyncio
    async def test_get_empty_when_no_workspace(self):
        from app.services.proactive.featured_topics import get_recent_featured
        assert await get_recent_featured("") == set()
        assert await get_recent_featured(None) == set()

    @pytest.mark.asyncio
    async def test_get_reads_and_parses(self):
        from app.services.proactive import featured_topics as m
        redis = MagicMock()
        redis.lrange = AsyncMock(return_value=[
            "赵雷当爸爸".encode(),
            "iPhone 秒空".encode(),
            "第三条",
        ])
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            result = await m.get_recent_featured("ws-1")
        assert result == {"赵雷当爸爸", "iPhone 秒空", "第三条"}

    @pytest.mark.asyncio
    async def test_get_swallows_redis_failure(self):
        """Redis 挂 → 返空集合 (放行, 不阻塞主动消息发送)."""
        from app.services.proactive import featured_topics as m
        redis = MagicMock()
        redis.lrange = AsyncMock(side_effect=RuntimeError("redis down"))
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            assert await m.get_recent_featured("ws-1") == set()

    @pytest.mark.asyncio
    async def test_remember_writes_and_trims(self):
        from app.services.proactive import featured_topics as m
        pipe = MagicMock()
        pipe.lpush = MagicMock(); pipe.ltrim = MagicMock(); pipe.expire = MagicMock()
        pipe.execute = AsyncMock()
        redis = MagicMock()
        redis.pipeline = MagicMock(return_value=pipe)
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            await m.remember_featured("ws-1", "赵雷当爸爸")
        pipe.lpush.assert_called_once()
        pipe.ltrim.assert_called_once_with(
            m._FEATURED_KEY.format(workspace_id="ws-1"),
            0, m._FEATURED_MAX - 1,
        )
        pipe.expire.assert_called_once_with(
            m._FEATURED_KEY.format(workspace_id="ws-1"),
            m._FEATURED_TTL_S,
        )
        pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remember_no_op_on_empty(self):
        from app.services.proactive import featured_topics as m
        redis = MagicMock()
        redis.pipeline = MagicMock()  # 不该被调
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            await m.remember_featured("", "some title")
            await m.remember_featured("ws-1", "")
        redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_user_isolation(self):
        """workspace_A 记账不影响 workspace_B 的读取 (per-user 隔离验证)."""
        from app.services.proactive import featured_topics as m
        # 模拟两个 workspace 的 Redis 键各自独立
        store: dict[str, list] = {}

        async def fake_lrange(key, start, end):
            return [t.encode() for t in store.get(key, [])[start:end + 1]]

        async def fake_pipe_exec():
            return [None, None, None]

        pipe_a = MagicMock()
        pipe_a.execute = AsyncMock(side_effect=fake_pipe_exec)
        def _lpush_a(key, val):
            store.setdefault(key, []).insert(0, val)
        pipe_a.lpush = MagicMock(side_effect=_lpush_a)
        pipe_a.ltrim = MagicMock()
        pipe_a.expire = MagicMock()

        redis = MagicMock()
        redis.lrange = AsyncMock(side_effect=fake_lrange)
        redis.pipeline = MagicMock(return_value=pipe_a)

        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            # user A 记了一个热点
            await m.remember_featured("ws-user-A", "赵雷当爸爸")
            # user B 读自己的 → 空 (关键验证: 隔离)
            b_result = await m.get_recent_featured("ws-user-B")
            # user A 读自己的 → 有
            a_result = await m.get_recent_featured("ws-user-A")

        assert b_result == set(), "user B 不该看到 user A 的 featured 记账"
        assert a_result == {"赵雷当爸爸"}
