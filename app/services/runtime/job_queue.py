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
