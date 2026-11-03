"""Rewrite circuit breaker: stop rewriting when it fires on too many turns.

Each main reply that reaches the post-generation check counts one turn; each
rewrite actually generated counts one rewrite. Counts live in 5-minute Redis
buckets and the rate is read over the last 12 (one hour). Above the policy rate
with enough turns, rewriting is skipped (prompt sections and the backstop keep
working); ops/service_health reports the same reading so the existing alert
outlet notifies. Redis trouble never blocks a reply: the state is "unknown"
and rewriting stays allowed, since each turn rewrites at most once.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

BUCKET_SECONDS = 300
# The client's socket timeout is 5 s; a counter on the reply path must not wait that long.
HOT_PATH_TIMEOUT_SECONDS = 0.3
WINDOW_BUCKETS = 12
_TTL_SECONDS = BUCKET_SECONDS * (WINDOW_BUCKETS + 1)
_PREFIX = "reply_repetition"


def _bucket(now: datetime) -> int:
    return int(now.timestamp()) // BUCKET_SECONDS


def _keys(kind: str, now: datetime) -> list[str]:
    last = _bucket(now)
    return [f"{_PREFIX}:{kind}:{b}" for b in range(last - WINDOW_BUCKETS + 1, last + 1)]


async def _redis():
    from app.redis_client import get_redis

    return await get_redis()


async def record(kind: str, now: datetime | None = None) -> None:
    """kind: "turns" or "rewrites". Failures are swallowed; a lost count is not an outage."""
    now = now or datetime.now(UTC)
    key = f"{_PREFIX}:{kind}:{_bucket(now)}"
    try:
        async with asyncio.timeout(HOT_PATH_TIMEOUT_SECONDS):
            redis = await _redis()
            pipe = redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, _TTL_SECONDS)
            await pipe.execute()
    except Exception:
        pass


def _total(values) -> int:
    return sum(int(v) for v in values if v is not None)


async def window(now: datetime | None = None) -> dict:
    """Last-hour counts; raises when Redis can't be read (callers decide what unknown means)."""
    now = now or datetime.now(UTC)
    redis = await _redis()
    turns = _total(await redis.mget(_keys("turns", now)))
    rewrites = _total(await redis.mget(_keys("rewrites", now)))
    return {"turns": turns, "rewrites": rewrites, "window_seconds": BUCKET_SECONDS * WINDOW_BUCKETS}


def judge(counts: dict, policy) -> dict:
    turns, rewrites = counts["turns"], counts["rewrites"]
    rate = rewrites / turns if turns else None
    open_ = bool(turns >= policy.breaker_min_turns and rate is not None and rate > policy.breaker_rate)
    return {**counts, "rate": None if rate is None else round(rate, 4), "state": "open" if open_ else "closed",
            "threshold": policy.breaker_rate, "min_turns": policy.breaker_min_turns,
            "enough_turns": turns >= policy.breaker_min_turns}


async def state(policy, now: datetime | None = None) -> dict:
    try:
        async with asyncio.timeout(HOT_PATH_TIMEOUT_SECONDS):
            counts = await window(now)
        return judge(counts, policy)
    except Exception as exc:
        return {"state": "unknown", "error_type": type(exc).__name__}


async def measure(now: datetime, carry: dict) -> dict:
    """ops/service_health check: alerts while the breaker is open; unreadable counts raise (unknown)."""
    from app.services.chat.reply_repetition.policy import current

    reading = judge(await window(now), current())
    return {"count": reading["rewrites"], "turns": reading["turns"], "rate": reading["rate"],
            "threshold": reading["threshold"], "min_turns": reading["min_turns"],
            "window_seconds": reading["window_seconds"], "alerting": reading["state"] == "open"}
