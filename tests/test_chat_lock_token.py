"""Per-conversation chat lock must be ownership-safe (CAS release).

Without a token, a worker whose lock TTL expired could DELETE a lock a
different worker has since acquired for the same conversation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services.interaction import delayed_queue as dq
from tests.conftest import FakeRedis


@pytest.mark.asyncio
async def test_lock_returns_token_and_blocks_second_acquire():
    redis = FakeRedis()
    with patch.object(dq, "get_redis", return_value=redis):
        token1 = await dq.try_lock_conversation("c1", ttl=60)
        token2 = await dq.try_lock_conversation("c1", ttl=60)
    assert token1 is not None
    assert token2 is None  # already held


@pytest.mark.asyncio
async def test_unlock_with_token_releases():
    redis = FakeRedis()
    with patch.object(dq, "get_redis", return_value=redis):
        token = await dq.try_lock_conversation("c1", ttl=60)
        await dq.unlock_conversation("c1", token)
        # released → can re-acquire
        token2 = await dq.try_lock_conversation("c1", ttl=60)
    assert token2 is not None


@pytest.mark.asyncio
async def test_unlock_with_wrong_token_does_not_release():
    """CAS: 用错误 token unlock 不应删除他人持有的锁."""
    redis = FakeRedis()
    with patch.object(dq, "get_redis", return_value=redis):
        token = await dq.try_lock_conversation("c1", ttl=60)
        assert token is not None
        # 另一 worker 用错误 token 尝试释放 → 不生效
        await dq.unlock_conversation("c1", "some-other-token")
        # 原锁仍在 → 无法重新获取
        token2 = await dq.try_lock_conversation("c1", ttl=60)
    assert token2 is None


@pytest.mark.asyncio
async def test_unlock_legacy_no_token_still_deletes():
    """未传 token 的 legacy 调用仍 best-effort 删除 (向后兼容)."""
    redis = FakeRedis()
    with patch.object(dq, "get_redis", return_value=redis):
        await dq.try_lock_conversation("c1", ttl=60)
        await dq.unlock_conversation("c1")  # no token
        token2 = await dq.try_lock_conversation("c1", ttl=60)
    assert token2 is not None
