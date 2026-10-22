"""Bounded operational observations, deliberately separate from replayable jobs.

Redis loss means observations are unavailable, never that processing succeeded.
No chat text, memory text, provider error or authentication material is stored.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
import asyncio
import json
import logging
from uuid import uuid4

from app.redis_client import get_redis
from app.db import db
from app.services.runtime import user_erasure
from app.services.memory.recording.outcomes import MemorySideOutcome

logger = logging.getLogger(__name__)
_INDEX = "memory:processing:observations:index"
_DATA = "memory:processing:observations:data"
_RETENTION = 7 * 86400
_MAX_RECORDS = 200
_REASONS = frozenset({
    "processed", "no_new_messages", "reserved_messages", "intentional_skip",
    "prompt_disabled", "source_invalid", "extraction_failed", "processing_failed",
    "scope_unavailable", "scope_invalid", "lock_unavailable", "backlog_unavailable",
    "watermark_read_unavailable", "watermark_write_unavailable", "watermark_invalid",
    "timestamp_page_boundary", "watermark_conflict",
    "source_identity_missing", "source_identity_invalid", "source_read_unavailable", "source_batch_limit",
})
_WRITE = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[2] - ARGV[4])
for _, id in ipairs(expired) do redis.call('HDEL', KEYS[2], id) end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[2] - ARGV[4])
redis.call('HSET', KEYS[2], ARGV[1], ARGV[3])
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
local excess = redis.call('ZRANGE', KEYS[1], 0, -tonumber(ARGV[5])-1)
for _, id in ipairs(excess) do
  redis.call('HDEL', KEYS[2], id)
  redis.call('ZREM', KEYS[1], id)
end
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[4])
return 1
"""


def _safe_side(result: MemorySideOutcome) -> dict:
    item = asdict(result)
    if result.side not in {'user', 'ai'}:
        raise ValueError('invalid_side')
    if result.status not in {'completed', 'no_new_messages', 'excluded', 'failed', 'disabled'}:
        item['status'] = 'unknown'
    if result.progress not in {'advanced', 'unchanged', 'not_applicable', 'unknown'}:
        item['progress'] = 'unknown'
    if type(result.memory_count) is not int or result.memory_count < 0:
        item['memory_count'] = None
    if type(result.source_count) is not int or not 0 <= result.source_count <= 100:
        item['source_count'] = None
    if result.source_binding not in {'persisted_messages', 'unlinked_input', 'unknown'}:
        item['source_binding'] = 'unknown'
    if result.cursor_kind not in {'initial_window', 'timestamp_cohort', 'message_position', 'not_applicable'}:
        item['cursor_kind'] = 'unknown'
    if type(result.inspected_count) is not int or not 0 <= result.inspected_count <= 100:
        item['inspected_count'] = None
    if type(result.has_more) is not bool:
        item['has_more'] = None
    item['reason'] = result.reason if result.reason in _REASONS else 'processing_failed'
    return item


def _project_record(value: str, now: datetime) -> dict:
    item = json.loads(value)
    for key in ('id', 'conversation_id', 'started_at', 'updated_at'):
        if key == 'conversation_id' and item[key] is None:
            continue
        if not isinstance(item[key], str) or not 1 <= len(item[key]) <= 200:
            raise ValueError('invalid_record_field')
    updated = datetime.fromisoformat(item['updated_at'])
    if updated.tzinfo is None or datetime.fromisoformat(item['started_at']).tzinfo is None:
        raise ValueError('invalid_record_time')
    status = item['status']
    if status not in {'running', 'completed', 'incomplete', 'cancelled'}:
        status = 'unknown'
    if status == 'running' and (now - updated).total_seconds() > 900:
        status = 'unconfirmed'
    if not isinstance(item['sides'], list) or len(item['sides']) > 2:
        raise ValueError('invalid_side_count')
    sides = [_safe_side(MemorySideOutcome(
        **{key: side[key] for key in ('side', 'status', 'reason', 'memory_count', 'progress')},
        source_count=side.get('source_count'), source_binding=side.get('source_binding', 'unknown'),
        cursor_kind=side.get('cursor_kind', 'unknown'), inspected_count=side.get('inspected_count'),
        has_more=side.get('has_more'),
    )) for side in item['sides']]
    if len({side['side'] for side in sides}) != len(sides):
        raise ValueError('duplicate_side')
    return {key: item[key] for key in ('id', 'conversation_id', 'started_at', 'updated_at')} | {
        'status': status, 'sides': sides}


async def _write(record: dict) -> None:
    try:
        conversation_id = record.get('conversation_id')
        if conversation_id:
            owners = await db.query_raw('SELECT user_id FROM conversations WHERE id=$1', conversation_id)
            if not owners:
                return
            async with user_erasure.owner_write_scope(owners[0]['user_id'], database=db):
                redis = await get_redis()
                await redis.eval(_WRITE, 2, _INDEX, _DATA, record['id'],
                    datetime.fromisoformat(record['updated_at']).timestamp(),
                    json.dumps(record, ensure_ascii=False), _RETENTION, _MAX_RECORDS)
        else:
            redis = await get_redis()
            await redis.eval(_WRITE, 2, _INDEX, _DATA, record['id'],
                datetime.fromisoformat(record['updated_at']).timestamp(),
                json.dumps(record, ensure_ascii=False), _RETENTION, _MAX_RECORDS)
    except Exception as exc:
        # Telemetry is best effort; it must never prevent or retry a memory write.
        logger.warning("Memory processing observation unavailable (%s)", type(exc).__name__)


@asynccontextmanager
async def observe_memory_run(conversation_id: str | None):
    now = datetime.now(UTC).isoformat()
    record = {'id': str(uuid4()), 'conversation_id': conversation_id,
              'started_at': now, 'updated_at': now, 'status': 'running', 'sides': []}
    outcomes: list[MemorySideOutcome] = []
    await _write(record)
    try:
        yield outcomes
        record['status'] = ('completed' if len(outcomes) == 2 and all(
            item.status not in {'failed', 'disabled'} for item in outcomes) else 'incomplete')
    except asyncio.CancelledError:
        record['status'] = 'cancelled'
        raise
    except BaseException:
        record['status'] = 'incomplete'
        raise
    finally:
        record.update(updated_at=datetime.now(UTC).isoformat(), sides=[_safe_side(r) for r in outcomes])
        await _write(record)


async def inspect_memory_processing(*, limit: int = 20) -> dict:
    if not 1 <= limit <= 50:
        raise ValueError('limit_out_of_bounds')
    redis = await get_redis()
    now = datetime.now(UTC)
    ids = await redis.zrevrangebyscore(_INDEX, '+inf', now.timestamp() - _RETENTION,
                                     start=0, num=limit)
    raw = await redis.hmget(_DATA, ids) if ids else []
    records = []
    missing = 0
    for value in raw:
        if value is None:
            missing += 1
            continue
        # Allowlist the projection as well as the writer. A future producer
        # cannot accidentally expose a new field in an existing admin API.
        try:
            records.append(_project_record(value, now))
        except (ValueError, TypeError, KeyError):
            missing += 1
    return {'records': records, 'sampled_at': now.isoformat(), 'limit': limit,
            'sample_count': len(records), 'retention_days': 7, 'max_retained': _MAX_RECORDS,
            'missing_records': missing,
            'collection_status': 'partial' if missing else 'available' if records else 'no_retained_records',
            'replay_supported': False}
