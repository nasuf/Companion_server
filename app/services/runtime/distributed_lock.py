"""Small Redis-backed distributed locks for cross-process background work."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator

from app.redis_client import get_redis, is_redis_healthy, mark_redis_healthy

logger = logging.getLogger(__name__)

_KEY_PREFIX = "runtime:lock"

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class DistributedLockNotAcquired(RuntimeError):
    """The lock is currently held by another worker."""


class DistributedLockUnavailable(RuntimeError):
    """Redis lock backend is unavailable."""


def lock_key(name: str) -> str:
    return f"{_KEY_PREFIX}:{name}"


@contextlib.asynccontextmanager
async def distributed_lock(
    name: str,
    *,
    ttl_s: int,
    wait_timeout_s: float = 0.0,
    retry_interval_s: float = 0.25,
    fail_open: bool = False,
) -> AsyncIterator[bool]:
    """Acquire a Redis SET NX lock and release it with token compare/delete.

    Yields True when a real Redis lock was acquired. If Redis is unavailable and
    fail_open=True, yields False so callers can deliberately fall back to local
    behavior while preserving the same control flow.
    """
    key = lock_key(name)
    token = uuid.uuid4().hex
    deadline = time.monotonic() + max(0.0, wait_timeout_s)
    acquired = False

    while True:
        if fail_open and not is_redis_healthy():
            yield False
            return
        try:
            redis = await get_redis()
            acquired = bool(await redis.set(key, token, nx=True, ex=max(1, int(ttl_s))))
        except Exception as e:
            mark_redis_healthy(False)
            if fail_open:
                logger.warning(
                    "Distributed lock unavailable; running without Redis lock",
                    extra={
                        "lock_name": name,
                        "lock_key": key,
                        "error_type": type(e).__name__,
                    },
                )
                yield False
                return
            raise DistributedLockUnavailable(str(e)) from e

        if acquired:
            break
        if wait_timeout_s <= 0 or time.monotonic() >= deadline:
            raise DistributedLockNotAcquired(f"lock held: {name}")
        await asyncio.sleep(max(0.01, retry_interval_s))

    try:
        yield True
    finally:
        try:
            redis = await get_redis()
            await redis.eval(_RELEASE_LUA, 1, key, token)
        except Exception as e:
            logger.debug(
                "Distributed lock release failed",
                extra={
                    "lock_name": name,
                    "lock_key": key,
                    "error_type": type(e).__name__,
                },
            )
