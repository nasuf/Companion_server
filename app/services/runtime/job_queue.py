"""Small Redis-backed runtime job queue.

This is intentionally lightweight: jobs are JSON payloads stored in Redis,
with ready/delayed/running/dead-letter indexes. It gives long-running backend
work a recoverable status path without adding a database migration yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from redis.exceptions import RedisError

from app.redis_client import get_redis
from app.services.runtime.distributed_lock import lock_key
from app.services.runtime.job_queue_scripts import (
    ENQUEUE as _ENQUEUE_JOB_LUA,
    CLAIM as _CLAIM_JOB_LUA,
    RENEW as _RENEW_JOB_LUA,
    FINISH as _FINISH_JOB_LUA,
    RECOVER as _RECOVER_JOB_LUA,
    PROMOTE as _PROMOTE_JOB_LUA,
    RECONCILE as _RECONCILE_JOB_LUA,
    DIAGNOSE as _DIAGNOSE_JOB_LUA,
    DIAGNOSE_IDEMPOTENCY as _DIAGNOSE_IDEMPOTENCY_LUA,
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
_DEFAULT_LEASE_S = 60
_HEARTBEAT_INTERVAL_S = 15
_RENEW_TIMEOUT_S = 5
_INDEX_BATCH_SIZE = 100
_RECONCILE_CURSOR = 0
_RECONCILE_PENDING: deque[str] = deque()
_RECOVERY_OFFSET = 0
_HANDLERS: dict[str, JobHandler] = {}
_LEGACY_NO_DELAY_HANDLERS: set[str] = set()


_MANUAL_JOB_LUA = """
local function check(key, expected)
    local kind = redis.call('TYPE', key).ok
    if kind ~= 'none' and kind ~= expected then
        error('Runtime queue key type mismatch: expected ' .. expected)
    end
end
check(KEYS[1], 'hash')
check(KEYS[2], 'zset')
check(KEYS[3], 'zset')
check(KEYS[4], 'list')
check(KEYS[5], 'list')
check(KEYS[6], 'list')
local state = redis.call('HGET', KEYS[1], 'status')
if not state then return {'missing'} end
local retry = ARGV[2] == 'retry'
if retry and state ~= 'dead_letter' and state ~= 'failed' then
    return {'conflict', state}
end
if not retry and state ~= 'queued' and state ~= 'dead_letter'
   and state ~= 'failed' and state ~= 'resolved' then
    return {'conflict', state}
end
if retry then
    redis.call('HSET', KEYS[1], 'status', 'queued',
               'updated_at', ARGV[3], 'last_error', '', 'not_before', ARGV[3])
elseif state ~= 'resolved' then
    redis.call('HSET', KEYS[1], 'status', 'resolved', 'updated_at', ARGV[3])
end
redis.call('HDEL', KEYS[1], 'lease_token', 'lease_expires_at')
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
redis.call('LREM', KEYS[4], 0, ARGV[1])
redis.call('LREM', KEYS[5], 0, ARGV[1])
redis.call('LREM', KEYS[6], 0, ARGV[1])
if retry then redis.call('LPUSH', KEYS[6], ARGV[1]) end
return {'ok', redis.call('HGETALL', KEYS[1])}
"""

class RuntimeJobConflict(RuntimeError):
    def __init__(self, job_id: str, status: str, action: str):
        self.job_id = job_id
        self.status = status
        label = "重试" if action == "retry" else "标记解决"
        super().__init__(f"任务当前状态为 {status}，不能{label}。")


class RuntimeJobOrphaned(RuntimeError):
    """An existing idempotency binding has no verifiable task record.

    Do not silently recreate or locally replay an unknown historical operation.
    The caller must surface the failure for explicit operator reconciliation.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__("任务幂等记录存在，但任务正文缺失，需要核对后恢复。")


class RuntimeJobEnqueueUncertain(RuntimeError):
    """EVAL may have committed before its response was lost; never replay locally."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__("任务入队结果暂时无法确认，需要核对队列后恢复。")


def _flat_hash(values: list[Any]) -> dict[str, str]:
    return _decode_hash(dict(zip(values[::2], values[1::2], strict=True)))


def register_job_handler(job_type: str, handler: JobHandler, *, legacy_no_delay: bool = False) -> None:
    _HANDLERS[job_type] = handler
    if legacy_no_delay:
        _LEGACY_NO_DELAY_HANDLERS.add(job_type)
    else:
        _LEGACY_NO_DELAY_HANDLERS.discard(job_type)


def _job_key(job_id: str) -> str:
    return f"{_JOB_KEY_PREFIX}{job_id}"


def _idempotency_key(key: str) -> str:
    return f"{_IDEMP_KEY_PREFIX}{key}"


def _job_lock_key(job_id: str) -> str:
    # Keep the legacy lock name so an old worker cannot overlap a new claim.
    return lock_key(f"runtime_job:{job_id}")


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
    if not isinstance(payload, dict):
        raise ValueError("Runtime job payload must be an object")
    redis = await get_redis()
    candidate_id = uuid.uuid4().hex
    content = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    maximum = max(1, min(int(max_attempts), 1000000000))
    delay = max(0, int(delay_s))
    if delay >= _DEFAULT_JOB_TTL_S:
        raise ValueError("Runtime job delay must be shorter than its retention period")
    idem_key = _idempotency_key(idempotency_key or "")
    # Read only to supply every accessed key explicitly. The script checks this
    # expected binding before writing, including expiry and concurrent creators.
    for _ in range(16):
        existing = _decode(await redis.get(idem_key)) if idempotency_key else None
        job_id = existing or candidate_id
        try:
            result = await redis.eval(
                _ENQUEUE_JOB_LUA, 4, _job_key(job_id), idem_key, _READY_KEY, _DELAYED_KEY,
                "1" if idempotency_key else "0", existing or "", job_id, job_type,
                content, maximum, delay, _DEFAULT_JOB_TTL_S,
            )
        except RedisError as error:
            # Even a connection error may follow a successful server commit.
            # A caller's uncoordinated local fallback could duplicate execution.
            raise RuntimeJobEnqueueUncertain(job_id) from error
        outcome = _decode(result[0])
        if outcome == "changed":
            continue
        if outcome == "orphan":
            raise RuntimeJobOrphaned(job_id)
        return _decode(result[1]) or job_id
    raise RuntimeError("Runtime job idempotency binding keeps changing")


async def process_runtime_jobs(max_jobs: int = 10, stale_after_s: int = 15 * 60) -> int:
    await reconcile_runtime_job_indexes()
    await _recover_stale_running_jobs(stale_after_s)
    await _promote_due_jobs()
    redis = await get_redis()
    processed = 0
    for _ in range(max(1, min(max_jobs, 200))):
        # Peeking is read-only. Removal, lease and running index commit together.
        job_id = _decode(await redis.lindex(_READY_KEY, -1))
        if not job_id:
            break
        await _run_job_with_lock(job_id, require_ready=True)
        processed += 1
    return processed


async def _promote_due_jobs() -> None:
    redis = await get_redis()
    now = await _server_time(redis)
    due = [_decode(v) for v in await redis.zrangebyscore(
        _DELAYED_KEY, 0, now, start=0, num=_INDEX_BATCH_SIZE,
    )]
    due = [v for v in due if v]
    if not due:
        return
    for job_id in due:
        await redis.eval(
            _PROMOTE_JOB_LUA, 3, _job_key(job_id), _DELAYED_KEY, _READY_KEY, job_id,
        )


async def _recover_stale_running_jobs(stale_after_s: int) -> None:
    global _RECOVERY_OFFSET
    redis = await get_redis()
    now = await _server_time(redis)
    cutoff = now - max(1, stale_after_s)
    stale = [_decode(v) for v in await redis.zrangebyscore(
        _RUNNING_KEY, 0, now, start=_RECOVERY_OFFSET, num=_INDEX_BATCH_SIZE,
    )]
    stale = [v for v in stale if v]
    _RECOVERY_OFFSET = _RECOVERY_OFFSET + len(stale) if len(stale) == _INDEX_BATCH_SIZE else 0
    if not stale:
        return
    for job_id in stale:
        recovered = await redis.eval(
            _RECOVER_JOB_LUA, 6, _job_key(job_id), _RUNNING_KEY, _READY_KEY,
            _DELAYED_KEY, _job_lock_key(job_id), _DLQ_KEY, job_id, cutoff,
        )
        if recovered:
            logger.warning("Recovered interrupted runtime job", extra={
                "event": "runtime_job", "job_id": job_id, "phase": "lease_recovered",
            })


async def _server_time(redis) -> float:
    seconds, micros = await redis.time()
    return int(seconds) + int(micros) / 1000000


async def reconcile_runtime_job_indexes(*, limit: int = _INDEX_BATCH_SIZE) -> dict[str, int]:
    """Restore recoverable records left without an index by older processes.

    One SCAN page per tick; oversized pages are drained across later ticks.
    Payloads, terminal states, idempotency bindings and existing TTLs stay intact.
    """
    global _RECONCILE_CURSOR
    redis = await get_redis()
    if not _RECONCILE_PENDING:
        _RECONCILE_CURSOR, keys = await redis.scan(
            _RECONCILE_CURSOR, match=_JOB_KEY_PREFIX + "*", count=_INDEX_BATCH_SIZE,
        )
        _RECONCILE_PENDING.extend(value for key in keys if (value := _decode(key)))
    inspected = repaired = unverified = 0
    for _ in range(max(1, min(limit, 200))):
        if not _RECONCILE_PENDING:
            break
        key = _RECONCILE_PENDING.popleft()
        if not key.startswith(_JOB_KEY_PREFIX):
            continue
        job_id = key[len(_JOB_KEY_PREFIX):]
        inspected += 1
        job_type = _decode(await redis.hget(key, "type"))
        result = int(await redis.eval(
            _RECONCILE_JOB_LUA, 5, key, _READY_KEY, _DELAYED_KEY,
            _RUNNING_KEY, _job_lock_key(job_id), job_id,
            "1" if job_type in _LEGACY_NO_DELAY_HANDLERS else "0",
        ))
        repaired += result == 1
        unverified += result == -1
    if repaired:
        logger.warning("Restored runtime job indexes", extra={
            "event": "runtime_job", "phase": "index_repaired", "count": repaired,
        })
    if unverified:
        logger.warning("Runtime jobs need explicit legacy delay reconciliation", extra={
            "event": "runtime_job", "phase": "index_unverified", "count": unverified,
        })
    return {"inspected": inspected, "repaired": repaired, "unverified": unverified}


async def _run_job(job_id: str) -> None:
    await _run_job_with_lock(job_id)


async def _claim_job(job_id: str, *, require_ready: bool = False) -> dict[str, str] | None:
    redis = await get_redis()
    result = await redis.eval(
        _CLAIM_JOB_LUA, 6, _job_key(job_id), _READY_KEY, _RUNNING_KEY,
        _DELAYED_KEY, _job_lock_key(job_id), _DLQ_KEY,
        job_id, uuid.uuid4().hex, "1" if require_ready else "0",
        max(1000, int(_DEFAULT_LEASE_S * 1000)),
    )
    if _decode(result[0]) != "claimed":
        return None
    return _flat_hash(result[1])


async def _run_job_with_lock(job_id: str, *, require_ready: bool = False) -> None:
    redis = await get_redis()
    job = await _claim_job(job_id, require_ready=require_ready)
    if job is None:
        return
    await _execute_claimed_job(redis, job)


async def _renew_job_lease(redis, job: dict[str, str]) -> bool:
    return bool(await redis.eval(
        _RENEW_JOB_LUA, 3, _job_key(job["id"]), _RUNNING_KEY,
        _job_lock_key(job["id"]), job["id"], job["attempts"], job["lease_token"],
        max(1000, int(_DEFAULT_LEASE_S * 1000)),
    ))


async def _heartbeat_job(redis, job: dict[str, str]) -> None:
    while True:
        await asyncio.sleep(min(_HEARTBEAT_INTERVAL_S, _DEFAULT_LEASE_S / 3))
        renewed = await asyncio.wait_for(
            _renew_job_lease(redis, job), timeout=min(_RENEW_TIMEOUT_S, _DEFAULT_LEASE_S / 3),
        )
        if not renewed:
            return


async def _execute_claimed_job(redis, job: dict[str, str]) -> None:
    job_id = job["id"]
    job_type = job.get("type", "")
    handler = _HANDLERS.get(job_type)
    attempts = int(job["attempts"])
    max_attempts = int(job.get("max_attempts") or _DEFAULT_MAX_ATTEMPTS)

    async def invoke():
        if handler is None:
            raise RuntimeError(f"No handler registered for runtime job type: {job_type}")
        payload = json.loads(job.get("payload") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Runtime job payload must be an object")
        await handler(payload)

    task = asyncio.create_task(invoke())
    heartbeat = asyncio.create_task(_heartbeat_job(redis, job))
    try:
        done, _ = await asyncio.wait((task, heartbeat), return_when=asyncio.FIRST_COMPLETED)
        if heartbeat in done:
            error = heartbeat.exception()
            logger.warning("Runtime job lease lost; stopping handler", extra={
                "event": "runtime_job", "job_id": job_id, "phase": "lease_lost",
                "error_type": type(error).__name__ if error else "LeaseExpired",
            })
            return
        try:
            await task
        except Exception as error:
            status = "dead_letter" if attempts >= max_attempts else "queued"
            committed = await _finish_job(
                redis, job_id, attempts, status, str(error)[:500],
                retry_at=int(await _server_time(redis)) + _DEFAULT_RETRY_DELAY_S * attempts,
                lease_token=job["lease_token"],
            )
            if committed:
                logger.warning(
                    f"Runtime job failed: {job_id} type={job_type} error={error}",
                    extra={"event": "runtime_job", "job_type": job_type,
                           "job_id": job_id, "phase": "dead_letter" if status == "dead_letter" else "retry"},
                )
            return
        await _finish_job(redis, job_id, attempts, "succeeded", "", lease_token=job["lease_token"])
    finally:
        heartbeat.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, heartbeat, return_exceptions=True)


async def _finish_job(
    redis, job_id: str, attempts: int, status: str, error: str, *, retry_at: int = 0,
    lease_token: str = "",
) -> bool:
    if status not in {"queued", "dead_letter", "succeeded"}:
        raise ValueError("Invalid runtime job completion state")
    committed = bool(await redis.eval(
        _FINISH_JOB_LUA, 7, _job_key(job_id), _RUNNING_KEY,
        _DELAYED_KEY, _DLQ_KEY, _SUCCEEDED_KEY, _job_lock_key(job_id), _READY_KEY,
        job_id, attempts, status, int(time.time()), error, retry_at, lease_token,
    ))
    if not committed:
        logger.warning(
            "Ignored obsolete runtime job result",
            extra={"event": "runtime_job", "job_id": job_id, "phase": "stale_result_ignored"},
        )
    return committed


async def inspect_runtime_job(job_id: str) -> dict[str, Any] | None:
    redis = await get_redis()
    raw = await redis.hgetall(_job_key(job_id))
    return _serialize_job(_decode_hash(raw)) if raw else None


async def diagnose_runtime_job_queue(
    *, cursor: int = 0, idempotency_cursor: int = 0, limit: int = 50,
) -> dict[str, Any]:
    """Bounded read-only pages; no payloads, binding names or ownership tokens."""
    redis = await get_redis()
    limit = max(1, min(limit, 200))
    next_cursor, keys = await redis.scan(cursor, match=_JOB_KEY_PREFIX + "*", count=limit)
    next_idem_cursor, bindings = await redis.scan(
        idempotency_cursor, match=_IDEMP_KEY_PREFIX + "*", count=limit,
    )
    truncated = len(keys) > limit or len(bindings) > limit
    anomalies: list[dict[str, str]] = []
    for raw in keys[:limit]:
        key = _decode(raw) or ""
        job_id = key[len(_JOB_KEY_PREFIX):]
        values = [_decode(v) or "" for v in await redis.eval(
            _DIAGNOSE_JOB_LUA, 5, key, _READY_KEY, _DELAYED_KEY,
            _RUNNING_KEY, _job_lock_key(job_id), job_id,
        )]
        issue = None
        if values[0] != "ok":
            issue = "record_" + values[0]
        elif values[2] == "queued" and values[4] == "0" and not values[5]:
            issue = "missing_pending_index"
            if not values[3] and values[1] not in _LEGACY_NO_DELAY_HANDLERS:
                issue = "legacy_delay_unverified"
        elif values[2] == "running" and not values[6]:
            issue = "missing_running_index"
        if issue:
            anomalies.append({"id": job_id, "issue": issue})
    for raw in bindings[:limit]:
        key = _decode(raw) or ""
        job_id = _decode(await redis.get(key))
        if job_id and await redis.eval(
            _DIAGNOSE_IDEMPOTENCY_LUA, 2, key, _job_key(job_id), job_id,
        ):
            anomalies.append({"id": job_id, "issue": "idempotency_record_missing"})
    return {
        "anomalies": anomalies, "inspected_records": min(len(keys), limit),
        "inspected_bindings": min(len(bindings), limit),
        "next_cursor": int(next_cursor), "next_idempotency_cursor": int(next_idem_cursor),
        "truncated": truncated,
        "scan_complete": next_cursor == 0 and next_idem_cursor == 0 and not truncated,
        "read_only": True,
    }


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


async def _manual_job_transition(job_id: str, action: str) -> dict[str, Any] | None:
    redis = await get_redis()
    result = await redis.eval(
        _MANUAL_JOB_LUA, 6, _job_key(job_id), _RUNNING_KEY, _DELAYED_KEY,
        _DLQ_KEY, _SUCCEEDED_KEY, _READY_KEY, job_id, action, int(await _server_time(redis)),
    )
    outcome = _decode(result[0])
    if outcome == "missing":
        return None
    if outcome == "conflict":
        raise RuntimeJobConflict(job_id, _decode(result[1]) or "unknown", action)
    return _serialize_job(_flat_hash(result[1]))


async def retry_runtime_job(job_id: str) -> dict[str, Any] | None:
    return await _manual_job_transition(job_id, "retry")


async def retry_runtime_jobs(job_ids: list[str]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    conflicts: list[dict[str, str]] = []
    for job_id in list(dict.fromkeys(job_ids))[:200]:
        try:
            job = await retry_runtime_job(job_id)
        except RuntimeJobConflict as error:
            conflicts.append({"id": job_id, "status": error.status, "detail": str(error)})
            continue
        if job is None:
            missing.append(job_id)
        else:
            results.append(job)
    return {
        "items": results,
        "retried_count": len(results),
        "missing_ids": missing,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
    }


async def resolve_runtime_job(job_id: str) -> dict[str, Any] | None:
    return await _manual_job_transition(job_id, "resolve")


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


def _serialize_job(job: dict[str, str]) -> dict[str, Any]:
    payload: Any = {}
    try:
        payload = json.loads(job.get("payload") or "{}")
    except Exception:
        payload = {}
    created_at = _ts_iso(job.get("created_at"))
    updated_at = _ts_iso(job.get("updated_at"))
    expires = job.get("lease_expires_at")
    lease_state = None
    if job.get("status") == "running":
        lease_state = "legacy"
        if job.get("lease_token"):
            try:
                lease_state = "active" if float(expires or 0) > time.time() else "expired"
            except ValueError:
                lease_state = "unknown"
    return {
        "id": job.get("id"),
        "type": job.get("type"),
        "status": job.get("status"),
        "attempts": _safe_int(job.get("attempts")),
        "max_attempts": _safe_int(job.get("max_attempts")),
        "created_at": created_at,
        "updated_at": updated_at,
        "last_error": job.get("last_error") or "",
        "payload": payload,
        "queue_version": _safe_int(job.get("queue_version")) or 1,
        "lease_state": lease_state,
        "lease_expires_at": _ts_iso(expires),
        "heartbeat_at": _ts_iso(job.get("heartbeat_at")),
        "recoveries": _safe_int(job.get("recoveries")),
        "last_recovered_at": _ts_iso(job.get("last_recovered_at")),
        "not_before": _ts_iso(job.get("not_before")),
    }


def _safe_int(value: str | None) -> int:
    try:
        return max(0, int(value or 0))
    except (ValueError, TypeError):
        return 0


def _ts_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime_from_epoch(float(value))
    except Exception:
        return None


def datetime_from_epoch(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
