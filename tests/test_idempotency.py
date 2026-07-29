"""防重复提交.

要防的是双击、网络重试、移动端弱网自动重发 —— 几秒内内容完全相同的两个请求, 结果
是用户列表里凭空多出一条。多 worker 之后两个请求真正并行, 更容易撞上。

这层是**体验保护而非正确性保护**: Redis 挂了必须放行, 为它拦掉用户的正常创建是本末
倒置。下面的用例把这个取舍也钉住。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.api.idempotency import (
    DEFAULT_WINDOW_S,
    SubmissionGuard,
    claim_submission,
    fingerprint,
)


class _Redis:
    """够用的内存 Redis: SET NX / GET / DELETE."""

    def __init__(self, *, broken: bool = False):
        self.store: dict[str, str] = {}
        self.broken = broken

    async def set(self, key, value, *, nx=False, ex=None):
        if self.broken:
            raise ConnectionError("redis down")
        if nx and key in self.store:
            return None
        self.store[key] = str(value)
        return True

    async def get(self, key):
        if self.broken:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def delete(self, key):
        if self.broken:
            raise ConnectionError("redis down")
        return int(self.store.pop(key, None) is not None)


def _patch(redis: _Redis):
    return patch("app.redis_client.get_redis", new=AsyncMock(return_value=redis))


class TestFingerprint:
    def test_key_order_does_not_change_the_fingerprint(self):
        """客户端两次序列化的字段顺序可能不同; 不排序去重就失效了."""
        assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})

    def test_different_content_differs(self):
        assert fingerprint({"a": 1}) != fingerprint({"a": 2})

    def test_handles_non_serialisable_values(self):
        from datetime import datetime

        assert fingerprint({"at": datetime(2026, 7, 29)})


class TestClaim:
    @pytest.mark.asyncio
    async def test_first_submission_is_not_duplicate(self):
        with _patch(_Redis()):
            g = await claim_submission("capsule.create", "u1", {"x": 1})
        assert not g.is_duplicate

    @pytest.mark.asyncio
    async def test_second_identical_submission_is_duplicate(self):
        redis = _Redis()
        with _patch(redis):
            first = await claim_submission("capsule.create", "u1", {"x": 1})
            await first.record("cap-123")
            second = await claim_submission("capsule.create", "u1", {"x": 1})
        assert second.is_duplicate
        assert second.duplicate_of == "cap-123"

    @pytest.mark.asyncio
    async def test_different_users_do_not_collide(self):
        redis = _Redis()
        with _patch(redis):
            await claim_submission("capsule.create", "u1", {"x": 1})
            other = await claim_submission("capsule.create", "u2", {"x": 1})
        assert not other.is_duplicate

    @pytest.mark.asyncio
    async def test_different_scopes_do_not_collide(self):
        redis = _Redis()
        with _patch(redis):
            await claim_submission("capsule.create", "u1", {"x": 1})
            other = await claim_submission("will.create", "u1", {"x": 1})
        assert not other.is_duplicate

    @pytest.mark.asyncio
    async def test_in_flight_duplicate_has_no_resource_yet(self):
        """并发的第二个请求会看到占位值, 知道有人在建但拿不到 id."""
        redis = _Redis()
        with _patch(redis):
            await claim_submission("capsule.create", "u1", {"x": 1})
            second = await claim_submission("capsule.create", "u1", {"x": 1})
        assert second.is_duplicate and second.duplicate_of == "-"


class TestRelease:
    @pytest.mark.asyncio
    async def test_release_lets_the_user_retry_immediately(self):
        """创建失败不撤占位的话, 用户重试会被判成重复而卡死 —— 既没建成又不让重试."""
        redis = _Redis()
        with _patch(redis):
            first = await claim_submission("capsule.create", "u1", {"x": 1})
            await first.release()
            retry = await claim_submission("capsule.create", "u1", {"x": 1})
        assert not retry.is_duplicate

    @pytest.mark.asyncio
    async def test_release_on_a_duplicate_guard_is_a_noop(self):
        """重复请求不持有占位, 它 release 会把别人的占位删掉."""
        redis = _Redis()
        with _patch(redis):
            await claim_submission("capsule.create", "u1", {"x": 1})
            second = await claim_submission("capsule.create", "u1", {"x": 1})
            await second.release()
            third = await claim_submission("capsule.create", "u1", {"x": 1})
        assert third.is_duplicate, "重复请求不该有权撤销首个请求的占位"


class TestDegradation:
    @pytest.mark.asyncio
    async def test_redis_down_allows_the_request(self):
        """这层保护体验不保护正确性; Redis 挂了拦下用户的创建是本末倒置."""
        with _patch(_Redis(broken=True)):
            g = await claim_submission("capsule.create", "u1", {"x": 1})
        assert not g.is_duplicate

    @pytest.mark.asyncio
    async def test_record_failure_is_swallowed(self):
        g = SubmissionGuard(key="k", _claimed=True)
        with _patch(_Redis(broken=True)):
            await g.record("id-1")     # 不该抛

    @pytest.mark.asyncio
    async def test_window_is_short_enough_to_allow_intentional_repeats(self):
        """窗口太长会挡住"用户确实想再建一条同样的"这种正当意图."""
        assert 5 <= DEFAULT_WINDOW_S <= 60


class TestCapsuleWiring:
    def test_create_capsule_guards_and_releases(self):
        import inspect

        from app.api.public import time_capsules

        src = inspect.getsource(time_capsules.create_capsule)
        assert "claim_submission" in src, "建胶囊没有防重复提交"
        assert "guard.release()" in src, "创建失败没有撤占位, 用户重试会被卡死"
        assert "guard.record(" in src, "没有回写资源 id, 重复请求拿不到既有胶囊"

    def test_open_capsule_is_conditional_in_sql(self):
        """"只开一次"要写在 WHERE 里, 不能只靠前置的状态判断."""
        import inspect

        from app.api.public import time_capsules

        src = inspect.getsource(time_capsules.open_capsule)
        assert "opened_at IS NULL" in src
