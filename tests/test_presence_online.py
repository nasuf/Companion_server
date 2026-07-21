from unittest.mock import AsyncMock, patch

import pytest


class _FakeZSetRedis:
    """Minimal in-memory Redis supporting the ZSET ops presence.py uses."""

    def __init__(self):
        self.zsets: dict[str, dict[str, float]] = {}

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zrem(self, key, *members):
        z = self.zsets.get(key)
        if not z:
            return 0
        removed = 0
        for m in members:
            if m in z:
                del z[m]
                removed += 1
        return removed

    async def zremrangebyscore(self, key, min_score, max_score):
        z = self.zsets.get(key)
        if not z:
            return 0
        cutoff = float(max_score)  # presence.py calls ("-inf", cutoff): drop score <= cutoff
        stale = [m for m, s in z.items() if s <= cutoff]
        for m in stale:
            del z[m]
        return len(stale)

    async def zrange(self, key, start, end):
        z = self.zsets.get(key, {})
        members = [m for m, _ in sorted(z.items(), key=lambda kv: kv[1])]
        if end == -1:
            end = len(members) - 1
        return members[start:end + 1]

    async def zcard(self, key):
        return len(self.zsets.get(key, {}))


@pytest.mark.asyncio
async def test_heartbeat_pool_counts_distinct_users():
    """record_online (App 前台 / H5 心跳 / 登录) 计入不同用户, 重复不双计."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_online("app-user")
        await presence.record_online("h5-user")
        await presence.record_online("h5-user")
        count, ok = await presence.count_online_users()

    assert ok is True
    assert count == 2


@pytest.mark.asyncio
async def test_ws_pool_add_and_instant_remove():
    """WS 连接即在线, 断开即下线 (连接数语义, 不等 TTL)."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_ws_online("u1", "conn-a")
        await presence.record_ws_online("u2", "conn-b")
        count_before, _ = await presence.count_online_users()

        await presence.remove_ws_online("u1", "conn-a")
        count_after, _ = await presence.count_online_users()

    assert count_before == 2
    assert count_after == 1  # u1 掉线立即反映


@pytest.mark.asyncio
async def test_multi_connection_user_counts_once_until_all_closed():
    """同一用户多个 WS 连接算 1 个在线; 关掉一个仍在线, 全关才下线."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_ws_online("u1", "tab-1")
        await presence.record_ws_online("u1", "tab-2")
        assert (await presence.count_online_users())[0] == 1

        await presence.remove_ws_online("u1", "tab-1")
        assert (await presence.count_online_users())[0] == 1  # tab-2 仍在

        await presence.remove_ws_online("u1", "tab-2")
        assert (await presence.count_online_users())[0] == 0


@pytest.mark.asyncio
async def test_union_dedupes_ws_and_heartbeat_pools():
    """同一用户同时在 WS 池和心跳池只计 1 次."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_ws_online("u1", "conn")
        await presence.record_online("u1")  # 同一人心跳
        await presence.record_online("u2")  # 只有心跳
        count, _ = await presence.count_online_users()

    assert count == 2


@pytest.mark.asyncio
async def test_remove_online_drops_heartbeat_user():
    """remove_online (App 切后台 / H5 隐藏) 从心跳池移除."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_online("u1")
        assert (await presence.count_online_users())[0] == 1
        await presence.remove_online("u1")
        assert (await presence.count_online_users())[0] == 0


@pytest.mark.asyncio
async def test_count_online_prunes_stale_members():
    """超过 TTL 的成员在计数时被剔除 (两个池都剪)."""
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        fake.zsets[presence._ONLINE_HB_ZKEY] = {"stale-hb": 1.0}
        fake.zsets[presence._ONLINE_WS_ZKEY] = {"stale-ws|conn": 1.0}
        await presence.record_online("fresh")
        count, ok = await presence.count_online_users()

    assert ok is True
    assert count == 1  # 仅 fresh, 两个陈旧成员被剔除


@pytest.mark.asyncio
async def test_count_online_redis_failure_degrades():
    from app.services.notifications import presence

    with patch.object(
        presence, "get_redis", new_callable=AsyncMock, side_effect=RuntimeError("down")
    ):
        count, ok = await presence.count_online_users()

    assert (count, ok) == (0, False)


@pytest.mark.asyncio
async def test_record_functions_ignore_empty_ids():
    from app.services.notifications import presence

    fake = _FakeZSetRedis()
    with patch.object(presence, "get_redis", new_callable=AsyncMock, return_value=fake):
        await presence.record_online(None)
        await presence.record_online("")
        await presence.record_ws_online(None, "c")
        await presence.record_ws_online("u", "")
        count, _ = await presence.count_online_users()

    assert count == 0


def test_heartbeat_endpoint_marks_online(api_client):
    """POST /auth/heartbeat 只刷新心跳池在线, 供 H5/web 与 App 同等被检测."""
    from app.api.jwt_auth import require_user
    from app.main import app

    app.dependency_overrides[require_user] = lambda: {"sub": "h5-user", "role": "user"}
    recorder = AsyncMock()
    try:
        with patch("app.api.public.auth.record_online", recorder):
            resp = api_client.post("/auth/heartbeat")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        recorder.assert_awaited_once_with("h5-user")
    finally:
        app.dependency_overrides.pop(require_user, None)


def test_offline_endpoint_removes_online(api_client):
    """POST /auth/offline 显式下线 (H5 页面隐藏/关闭)."""
    from app.api.jwt_auth import require_user
    from app.main import app

    app.dependency_overrides[require_user] = lambda: {"sub": "h5-user", "role": "user"}
    remover = AsyncMock()
    try:
        with patch("app.api.public.auth.remove_online", remover):
            resp = api_client.post("/auth/offline")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        remover.assert_awaited_once_with("h5-user")
    finally:
        app.dependency_overrides.pop(require_user, None)
