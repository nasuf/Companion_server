from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.runtime.distributed_lock import (
    DistributedLockNotAcquired,
    distributed_lock,
    lock_key,
)


class FakeRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, bool, int]] = []
        self.eval_calls: list[tuple[str, str]] = []

    async def set(self, key, value, *, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def eval(self, _script, _n_keys, key, token):
        self.eval_calls.append((key, token))
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0


@pytest.mark.asyncio
async def test_distributed_lock_acquires_and_releases_by_token():
    fake = FakeRedis()

    with patch(
        "app.services.runtime.distributed_lock.get_redis",
        AsyncMock(return_value=fake),
    ):
        async with distributed_lock("job:a", ttl_s=30) as acquired:
            assert acquired is True
            assert lock_key("job:a") in fake.values

    assert lock_key("job:a") not in fake.values
    assert fake.eval_calls


@pytest.mark.asyncio
async def test_distributed_lock_busy_raises_without_waiting():
    fake = FakeRedis()
    fake.values[lock_key("job:a")] = "other-token"

    with patch(
        "app.services.runtime.distributed_lock.get_redis",
        AsyncMock(return_value=fake),
    ):
        with pytest.raises(DistributedLockNotAcquired):
            async with distributed_lock("job:a", ttl_s=30, wait_timeout_s=0):
                pass


@pytest.mark.asyncio
async def test_distributed_lock_fail_open_on_redis_error():
    with (
        patch(
            "app.services.runtime.distributed_lock.is_redis_healthy",
            return_value=True,
        ),
        patch(
            "app.services.runtime.distributed_lock.mark_redis_healthy",
        ) as mark_health,
        patch(
            "app.services.runtime.distributed_lock.get_redis",
            AsyncMock(side_effect=RuntimeError("redis down")),
        ),
    ):
        async with distributed_lock("job:a", ttl_s=30, fail_open=True) as acquired:
            assert acquired is False

    mark_health.assert_called_once_with(False)
