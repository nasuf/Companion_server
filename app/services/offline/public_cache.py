"""Bounded single-flight caching for public discovery facts, never user context."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from copy import deepcopy
from uuid import uuid4

from app.redis_client import get_redis

logger = logging.getLogger(__name__)
_flights: dict[tuple, asyncio.Task] = {}
_RELEASE = "if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('DEL',KEYS[1]) end; return 0"


async def public_cached(
    namespace: str,
    identity,
    load,
    *,
    valid,
    ttl=900,
    timeout: float = 18,
    fresh: bool = False,
    redis_factory=None,
):
    """Reuse one bounded fetch across callers and Redis-connected workers.

    Fresh source checks always fetch independently. A cancelled caller cannot
    cancel the shared fetch; the fetch itself has a deadline. Redis failure
    degrades to one fetch per process. Failed loads are never cached.
    """
    key = (
        "offline:public:v1:"
        + namespace
        + ":"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    )
    redis_factory = redis_factory or get_redis

    async def fetch():
        async with asyncio.timeout(timeout):
            redis = None
            token = uuid4().hex
            owned = False

            async def read():
                payload = await redis.get(key)
                if payload:
                    try:
                        value = json.loads(payload)
                        if valid(value):
                            return value
                    except (ValueError, TypeError, KeyError):
                        pass
                return None

            try:
                try:
                    redis = await redis_factory()
                    if not fresh:
                        value = await read()
                        if value is not None:
                            return value
                        while True:
                            owned = bool(
                                await redis.set(
                                    key + ":lock", token, nx=True, ex=int(timeout) + 5
                                )
                            )
                            # Double-check after taking the lease: another owner
                            # may have published between GET and SET NX.
                            value = await read()
                            if value is not None:
                                return value
                            if owned:
                                break
                            await asyncio.sleep(0.2)
                except Exception as exc:
                    logger.debug("[offline-cache] unavailable (%s)", type(exc).__name__)
                    redis = None
                value = await load()
                if redis is not None and valid(value):
                    try:
                        await redis.set(
                            key,
                            json.dumps(value, ensure_ascii=False, allow_nan=False),
                            ex=ttl(value) if callable(ttl) else ttl,
                        )
                    except Exception:
                        pass
                return value
            finally:
                if owned and redis is not None:
                    try:
                        # Cleanup must have its own deadline: the outer timeout
                        # may already have delivered cancellation to load().
                        async with asyncio.timeout(1):
                            await redis.eval(_RELEASE, 1, key + ":lock", token)
                    except Exception:
                        pass  # The lease expires even if cleanup is unavailable.

    if fresh:
        return await fetch()
    flight_key = (asyncio.get_running_loop(), key)
    task = _flights.get(flight_key)
    if task is None:
        task = asyncio.create_task(fetch())
        _flights[flight_key] = task

        def complete(done):
            if _flights.get(flight_key) is done:
                _flights.pop(flight_key, None)
            if not done.cancelled():
                done.exception()  # Also consume errors if every caller cancelled.

        task.add_done_callback(complete)
    result = await asyncio.shield(task)
    # Independent copies prevent caller mutations from contaminating another
    # user's candidate list, dates, distances or source provenance.
    return deepcopy(result)
