"""Small Redis-backed runtime job queue.

This is intentionally lightweight: jobs are JSON payloads stored in Redis,
with ready/delayed/running/dead-letter indexes. It gives long-running backend
work a recoverable status path without adding a database migration yet.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.config import settings
from app.redis_client import get_redis
from app.services.runtime.distributed_lock import (
    DistributedLockNotAcquired,
    distributed_lock,
)

logger = logging.getLogger(__name__)

JobHandler = Callable[[dict[str, Any]], Awaitable[None]]

_READY_KEY = "runtime:jobs:ready"
_DELAYED_KEY = "runtime:jobs:delayed"
_RUNNING_KEY = "runtime:jobs:running"
_DLQ_KEY = "runtime:jobs:dlq"
_SUCCEEDED_KEY = "runtime:jobs:succeeded"
_JOB_KEY_PREFIX = "runtime:job:"
_IDEMP_KEY_PREFIX = "runtime:job_idem:"
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_DELAY_S = 30
_DEFAULT_JOB_TTL_S = 7 * 24 * 3600
_HANDLERS: dict[str, JobHandler] = {}


def register_job_handler(job_type: str, handler: JobHandler) -> None:
    _HANDLERS[job_type] = handler


def _job_key(job_id: str) -> str:
    return f"{_JOB_KEY_PREFIX}{job_id}"


def _idempotency_key(key: str) -> str:
    return f"{_IDEMP_KEY_PREFIX}{key}"


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_hash(raw: dict[Any, Any]) -> dict[str, str]:
    return {str(_decode(k)): str(_decode(v)) for k, v in raw.items() if _decode(k) is not None}


async def enqueue_runtime_job(
    job_type: str,
    payload: dict[str, Any],
    *,
    idempotency_key: str | None = None,
    delay_s: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> str:
    redis = await get_redis()
    job_id = uuid.uuid4().hex
    if idempotency_key:
        idem_key = _idempotency_key(idempotency_key)
        existing = _decode(await redis.get(idem_key))
        if existing:
            return existing
        claimed = await redis.set(idem_key, job_id, nx=True, ex=_DEFAULT_JOB_TTL_S)
        if not claimed:
            existing = _decode(await redis.get(idem_key))
            if existing:
                return existing

    now = int(time.time())
    await redis.hset(
        _job_key(job_id),
        mapping={
            "id": job_id,
            "type": job_type,
            "payload": json.dumps(payload, ensure_ascii=False),
            "status": "queued",
            "attempts": "0",
            "max_attempts": str(max(1, max_attempts)),
            "created_at": str(now),
            "updated_at": str(now),
            "last_error": "",
        },
    )
    await redis.expire(_job_key(job_id), _DEFAULT_JOB_TTL_S)

    if delay_s > 0:
        await redis.zadd(_DELAYED_KEY, {job_id: now + delay_s})
    else:
        await redis.lpush(_READY_KEY, job_id)
    return job_id


async def process_runtime_jobs(max_jobs: int = 10, stale_after_s: int = 15 * 60) -> int:
    await _recover_stale_running_jobs(stale_after_s)
    await _promote_due_jobs()
    redis = await get_redis()
    processed = 0
    for _ in range(max(1, max_jobs)):
        job_id = _decode(await redis.rpop(_READY_KEY))
        if not job_id:
            break
        await _run_job(job_id)
        processed += 1
    return processed


async def _promote_due_jobs() -> None:
    redis = await get_redis()
    now = int(time.time())
    due = [_decode(v) for v in await redis.zrangebyscore(_DELAYED_KEY, 0, now)]
    due = [v for v in due if v]
    if not due:
        return
    await redis.zrem(_DELAYED_KEY, *due)
    for job_id in due:
        await redis.lpush(_READY_KEY, job_id)


async def _recover_stale_running_jobs(stale_after_s: int) -> None:
    redis = await get_redis()
    cutoff = int(time.time()) - max(1, stale_after_s)
    stale = [_decode(v) for v in await redis.zrangebyscore(_RUNNING_KEY, 0, cutoff)]
    stale = [v for v in stale if v]
    if not stale:
        return
    await redis.zrem(_RUNNING_KEY, *stale)
    now = str(int(time.time()))
    for job_id in stale:
        await redis.hset(_job_key(job_id), mapping={"status": "queued", "updated_at": now})
        await redis.lpush(_READY_KEY, job_id)


async def _run_job(job_id: str) -> None:
    try:
        async with distributed_lock(
            f"runtime_job:{job_id}",
            ttl_s=600,
            fail_open=not settings.is_production(),
        ):
            await _run_job_with_lock(job_id)
    except DistributedLockNotAcquired:
        redis = await get_redis()
        await redis.zadd(_DELAYED_KEY, {job_id: int(time.time()) + 5})


async def _run_job_with_lock(job_id: str) -> None:
    redis = await get_redis()
    raw = await redis.hgetall(_job_key(job_id))
    if not raw:
        return
    job = _decode_hash(raw)
    job_type = job.get("type", "")
    handler = _HANDLERS.get(job_type)
    attempts = int(job.get("attempts") or "0") + 1
    max_attempts = int(job.get("max_attempts") or _DEFAULT_MAX_ATTEMPTS)
    now = int(time.time())

    await redis.hset(
        _job_key(job_id),
        mapping={"status": "running", "attempts": str(attempts), "updated_at": str(now)},
    )
    await redis.zadd(_RUNNING_KEY, {job_id: now})

    try:
        if handler is None:
            raise RuntimeError(f"No handler registered for runtime job type: {job_type}")
        payload = json.loads(job.get("payload") or "{}")
        await handler(payload)
    except Exception as e:
        await redis.zrem(_RUNNING_KEY, job_id)
        if attempts >= max_attempts:
            await redis.hset(
                _job_key(job_id),
                mapping={
                    "status": "dead_letter",
                    "last_error": str(e)[:500],
                    "updated_at": str(int(time.time())),
                },
            )
            await redis.lpush(_DLQ_KEY, job_id)
            logger.warning(
                f"Runtime job dead-lettered: {job_id} type={job_type} error={e}",
                extra={"event": "runtime_job", "job_type": job_type, "job_id": job_id, "phase": "dead_letter"},
            )
            return
        delay = _DEFAULT_RETRY_DELAY_S * attempts
        await redis.hset(
            _job_key(job_id),
            mapping={
                "status": "queued",
                "last_error": str(e)[:500],
                "updated_at": str(int(time.time())),
            },
        )
        await redis.zadd(_DELAYED_KEY, {job_id: int(time.time()) + delay})
        logger.warning(
            f"Runtime job failed; retry scheduled: {job_id} type={job_type} error={e}",
            extra={"event": "runtime_job", "job_type": job_type, "job_id": job_id, "phase": "retry"},
        )
        return

    await redis.zrem(_RUNNING_KEY, job_id)
    await redis.hset(
        _job_key(job_id),
        mapping={"status": "succeeded", "updated_at": str(int(time.time())), "last_error": ""},
    )
    await redis.lpush(_SUCCEEDED_KEY, job_id)
    await _safe_ltrim(redis, _SUCCEEDED_KEY, 0, 499)


async def inspect_runtime_job(job_id: str) -> dict[str, Any] | None:
    redis = await get_redis()
    raw = await redis.hgetall(_job_key(job_id))
    return _serialize_job(_decode_hash(raw)) if raw else None


async def list_runtime_jobs(
    *,
    status: str | None = None,
    job_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    redis = await get_redis()
    limit = max(1, min(limit, 200))
    status_keys = [status] if status else ["queued", "delayed", "running", "dead_letter", "succeeded"]
    ids: list[str] = []
    for key in status_keys:
        if key == "queued":
            ids.extend(await _list_ids(redis, _READY_KEY, limit))
        elif key == "delayed":
            ids.extend(await _zset_ids(redis, _DELAYED_KEY, limit))
        elif key == "running":
            ids.extend(await _zset_ids(redis, _RUNNING_KEY, limit))
        elif key in {"dead_letter", "dlq", "failed"}:
            ids.extend(await _list_ids(redis, _DLQ_KEY, limit))
        elif key == "succeeded":
            ids.extend(await _list_ids(redis, _SUCCEEDED_KEY, limit))
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for job_id in ids:
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        item = await inspect_runtime_job(job_id)
        if not item:
            continue
        if job_type and item.get("type") != job_type:
            continue
        items.append(item)
        if len(items) >= limit:
            break
    counts = await runtime_job_counts()
    return {"items": items, "count": len(items), "limit": limit, "counts": counts}


async def retry_runtime_job(job_id: str) -> dict[str, Any] | None:
    redis = await get_redis()
    job = await inspect_runtime_job(job_id)
    if job is None:
        return None
    now = str(int(time.time()))
    await redis.hset(
        _job_key(job_id),
        mapping={"status": "queued", "updated_at": now, "last_error": ""},
    )
    await redis.zrem(_RUNNING_KEY, job_id)
    await redis.zrem(_DELAYED_KEY, job_id)
    await _safe_lrem(redis, _DLQ_KEY, job_id)
    await _safe_lrem(redis, _SUCCEEDED_KEY, job_id)
    await redis.lpush(_READY_KEY, job_id)
    return await inspect_runtime_job(job_id)


async def retry_runtime_jobs(job_ids: list[str]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for job_id in list(dict.fromkeys(job_ids))[:200]:
        job = await retry_runtime_job(job_id)
        if job is None:
            missing.append(job_id)
        else:
            results.append(job)
    return {
        "items": results,
        "retried_count": len(results),
        "missing_ids": missing,
    }


async def resolve_runtime_job(job_id: str) -> dict[str, Any] | None:
    redis = await get_redis()
    job = await inspect_runtime_job(job_id)
    if job is None:
        return None
    await redis.hset(
        _job_key(job_id),
        mapping={"status": "resolved", "updated_at": str(int(time.time()))},
    )
    await redis.zrem(_RUNNING_KEY, job_id)
    await redis.zrem(_DELAYED_KEY, job_id)
    await _safe_lrem(redis, _DLQ_KEY, job_id)
    await _safe_lrem(redis, _SUCCEEDED_KEY, job_id)
    return await inspect_runtime_job(job_id)


async def runtime_job_counts() -> dict[str, int]:
    redis = await get_redis()
    return {
        "queued": int(await redis.llen(_READY_KEY) or 0),
        "delayed": int(await redis.zcard(_DELAYED_KEY) or 0),
        "running": int(await redis.zcard(_RUNNING_KEY) or 0),
        "dead_letter": int(await redis.llen(_DLQ_KEY) or 0),
        "succeeded": int(await redis.llen(_SUCCEEDED_KEY) or 0),
    }


async def _list_ids(redis, key: str, limit: int) -> list[str]:
    if hasattr(redis, "lrange"):
        raw = await redis.lrange(key, 0, limit - 1)
        return [v for v in (_decode(item) for item in raw) if v]
    values = getattr(redis, "lists", {}).get(key, [])
    return [str(v) for v in values[:limit]]


async def _zset_ids(redis, key: str, limit: int) -> list[str]:
    if hasattr(redis, "zrange"):
        raw = await redis.zrange(key, 0, limit - 1)
        return [v for v in (_decode(item) for item in raw) if v]
    zset = getattr(redis, "zsets", {}).get(key, {})
    return [
        str(member)
        for member, _score in sorted(zset.items(), key=lambda item: item[1])[:limit]
    ]


async def _safe_lrem(redis, key: str, value: str) -> None:
    if hasattr(redis, "lrem"):
        await redis.lrem(key, 0, value)
        return
    values = getattr(redis, "lists", {}).get(key)
    if isinstance(values, list):
        while value in values:
            values.remove(value)


async def _safe_ltrim(redis, key: str, start: int, end: int) -> None:
    if hasattr(redis, "ltrim"):
        await redis.ltrim(key, start, end)
        return
    values = getattr(redis, "lists", {}).get(key)
    if isinstance(values, list):
        stop = None if end == -1 else end + 1
        values[:] = values[start:stop]


def _serialize_job(job: dict[str, str]) -> dict[str, Any]:
    payload: Any = {}
    try:
        payload = json.loads(job.get("payload") or "{}")
    except Exception:
        payload = {}
    created_at = _ts_iso(job.get("created_at"))
    updated_at = _ts_iso(job.get("updated_at"))
    return {
        "id": job.get("id"),
        "type": job.get("type"),
        "status": job.get("status"),
        "attempts": int(job.get("attempts") or 0),
        "max_attempts": int(job.get("max_attempts") or 0),
        "created_at": created_at,
        "updated_at": updated_at,
        "last_error": job.get("last_error") or "",
        "payload": payload,
    }


def _ts_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime_from_epoch(int(value))
    except Exception:
        return None


def datetime_from_epoch(value: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
