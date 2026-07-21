from unittest.mock import AsyncMock, patch

import pytest


class _FakeZSetRedis:
    """Minimal in-memory Redis supporting the ZSET ops presence.py uses."""

    def __init__(self):
        self.zsets: dict[str, dict[str, float]] = {}

    async def zadd(self, key, mapping):
        z = self.zsets.setdefault(key, {})
        z.update(mapping)

    async def zremrangebyscore(self, key, min_score, max_score):
        z = self.zsets.get(key)
        if not z:
            return 0
        # presence.py calls with ("-inf", cutoff): drop members with score <= cutoff.
        cutoff = float(max_score)
        stale = [m for m, s in z.items() if s <= cutoff]
        for m in stale:
            del z[m]
        return len(stale)

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))


@pytest.mark.asyncio
async def test_record_online_counts_distinct_users_across_platforms():
    """record_online (WS/auth/App 共用) 后, count_online_users 计入不同用户."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_online("app-user")
        await presence.record_online("h5-user")
        await presence.record_online("h5-user")  # 重复不双计

        count, ok = await presence.count_online_users()

    assert ok is True
    assert count == 2


@pytest.mark.asyncio
async def test_count_online_prunes_stale_members():
    """超过 TTL 的成员在计数时被剔除."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        # 手动放一个陈旧成员 (远早于 now) + 一个新鲜成员.
        fake.zsets[presence._ONLINE_ZKEY] = {"stale": 1.0}
        await presence.record_online("fresh")

        count, ok = await presence.count_online_users()

    assert ok is True
    assert count == 1  # stale 被 zremrangebyscore 剔除


@pytest.mark.asyncio
async def test_count_online_redis_failure_degrades():
    """Redis 异常时返回 (0, False), 不抛."""
    from app.services.notifications import presence

    with patch.object(
        presence, "get_redis", new_callable=AsyncMock, side_effect=RuntimeError("down")
    ):
        count, ok = await presence.count_online_users()

    assert (count, ok) == (0, False)


@pytest.mark.asyncio
async def test_record_online_ignores_empty_user():
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_online(None)
        await presence.record_online("")
        count, _ = await presence.count_online_users()

    assert count == 0
