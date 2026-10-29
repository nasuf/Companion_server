"""Container-scoped API worker heartbeat; no process signals or business reads.

A new Docker hostname is a new deployment scope, so old deployment workers do
not appear as crashes. A process incarnation prevents PID reuse hiding a crash.
The bounded registry survives heartbeat expiry long enough to prove an exit.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime
from uuid import uuid4

from app.config import settings
from app.redis_client import get_redis

_HOST = socket.gethostname()
_INCARNATION = f"{os.getpid()}:{uuid4().hex}"
_PREFIX = f"ops:workers:{_HOST}"
_REGISTRY = _PREFIX + ":registry"
_START = _PREFIX + ":started"
_LIMIT = 64

_RECORD = """
local now = redis.call('TIME')
local at = tonumber(now[1]) + tonumber(now[2])/1000000
local old = redis.call('HGET',KEYS[1],ARGV[1])
local began = at
if old then began = cjson.decode(old).started_at end
redis.call('SET',KEYS[2],at,'NX','EX',86400)
redis.call('EXPIRE',KEYS[2],86400)
local rows = redis.call('HGETALL',KEYS[1])
for i=1,#rows,2 do
  if cjson.decode(rows[i+1]).last_seen < at-tonumber(ARGV[3]) then
    redis.call('HDEL',KEYS[1],rows[i])
  end
end
if not old and redis.call('HLEN',KEYS[1]) >= tonumber(ARGV[4]) then return 0 end
redis.call('HSET',KEYS[1],ARGV[1],cjson.encode({started_at=began,last_seen=at,closed=ARGV[2]=='closed'}))
redis.call('EXPIRE',KEYS[1],86400)
if ARGV[2]=='closed' then redis.call('DEL',KEYS[3])
else redis.call('SET',KEYS[3],'1','EX',ARGV[5]) end
return 1
"""


async def heartbeat(*, closed: bool = False) -> None:
    redis = await get_redis()
    kept = await redis.eval(
        _RECORD,
        3,
        _REGISTRY,
        _START,
        _PREFIX + ":" + _INCARNATION,
        _INCARNATION,
        "closed" if closed else "alive",
        settings.ops_worker_exit_window_seconds
        + settings.ops_worker_heartbeat_seconds
        + 300,
        _LIMIT,
        settings.ops_worker_heartbeat_seconds,
    )
    if not kept:
        raise RuntimeError("worker_registry_full")


def classify(rows: list[dict], started: float, now: float, carry: dict) -> dict:
    ttl = settings.ops_worker_heartbeat_seconds
    grace = now - started < settings.ops_worker_startup_grace_seconds
    live = [r for r in rows if not r["closed"] and now - r["last_seen"] <= ttl]
    # A missing heartbeat alone can be a Redis outage. A newer incarnation on
    # this host is required to call the old incarnation an abnormal exit.
    crashed = [
        r
        for r in rows
        if not r["closed"]
        and ttl < now - r["last_seen"] <= settings.ops_worker_exit_window_seconds
        and any(n["started_at"] > r["last_seen"] for n in live)
    ]
    unhealthy = len(live) < settings.web_concurrency or bool(crashed)
    bad = int(carry.get("bad_samples", 0)) + 1 if unhealthy and not grace else 0
    good = int(carry.get("good_samples", 0)) + 1 if not unhealthy and not grace else 0
    alerting = not grace and (
        bool(crashed)
        or bad >= settings.ops_worker_samples
        or bool(carry.get("alerting"))
        and good < settings.ops_worker_samples
    )
    return {
        "count": len(live),
        "expected": settings.web_concurrency,
        "abnormal_exits": len(crashed),
        "heartbeat_seconds": ttl,
        "window_seconds": settings.ops_worker_exit_window_seconds,
        "startup_grace_seconds": settings.ops_worker_startup_grace_seconds,
        "required_samples": settings.ops_worker_samples,
        "bad_samples": bad,
        "good_samples": good,
        "sample_state": "startup_grace" if grace else "measured",
        "alerting": alerting,
        "_carry": {"bad_samples": bad, "good_samples": good, "alerting": alerting},
    }


async def measure(now: datetime, carry: dict) -> dict:
    redis = await get_redis()
    # HLEN before HGETALL limits materialization even if the namespace is corrupt.
    size = await redis.hlen(_REGISTRY)
    if not 0 < size <= _LIMIT:
        raise RuntimeError("worker_registry_unavailable")
    raw, started, clock = (
        await redis.hgetall(_REGISTRY),
        await redis.get(_START),
        await redis.time(),
    )
    if not started or len(raw) > _LIMIT:
        raise RuntimeError("worker_registry_unavailable")
    rows = [json.loads(value) for value in raw.values()]
    sampled_at = float(clock[0]) + float(clock[1]) / 1e6
    # A deployment or missing sampling interval cannot count as consecutive
    # observations. Keep the incident latch, but restart its sample counters.
    if (
        carry.get("host") != _HOST
        or not 0 < sampled_at - float(carry.get("sampled_at", 0)) <= 240
    ):
        carry = (
            {"alerting": carry.get("alerting", False)}
            if carry.get("host") == _HOST
            else {}
        )
    result = classify(rows, float(started), sampled_at, carry)
    result["_carry"].update(host=_HOST, sampled_at=sampled_at)
    return result
