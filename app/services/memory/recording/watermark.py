"""Bounded extraction progress, not a durable or replayable write receipt.

Legacy timestamps consume the entire timestamp cohort. V2 positions consume
only through (created_at, id), allowing a cohort to span bounded pages. The ID
orders processing, never events in the real world. Backdated/late commits are
not covered by this cursor; durable dispatch must qualify them separately.

The same Redis key/30-day retention and completed-cohort timestamp format are
retained. Only an unfinished cohort needs V2 JSON. Old code fails closed on
that position; rollback must retain this reader while any cohort is unfinished,
not delete progress or truncate it to a timestamp. CAS rejects stale versions;
the existing account fence also protects initial writes against erasure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from uuid import uuid4

from app.redis_client import get_redis
from app.services.memory.recording.outcomes import MemoryProgressError
from app.services.runtime import user_erasure

_KEY_FMT = "mem:extract:wm:{conv}:{side}"
_TTL_SEC = 30 * 86400
_UNSET = object()
_CAS = """
local current = redis.call('GET', KEYS[1])
if (ARGV[1] == 'absent' and current) or
   (ARGV[1] == 'present' and current ~= ARGV[2]) then return 0 end
redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[4])
return 1
"""


def _key(conversation_id: str, side: str) -> str:
    if not isinstance(conversation_id, str) or not 1 <= len(conversation_id) <= 200 or side not in {'user', 'ai'}:
        raise MemoryProgressError('watermark_invalid')
    return _KEY_FMT.format(conv=conversation_id, side=side)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True)
class MessageProgress:
    timestamp: datetime
    message_id: str | None = None  # None: all IDs at this timestamp were consumed.
    raw: str = field(default='', repr=False, compare=False)

    def includes(self, timestamp: datetime, message_id: str) -> bool:
        ts = _utc(timestamp)
        return ts < self.timestamp or (ts == self.timestamp and (
            self.message_id is None or message_id <= self.message_id))


def progress_time(value: MessageProgress | datetime | None) -> datetime | None:
    """Timestamp inputs remain valid for legacy callers; normalize UTC once."""
    return _utc(value.timestamp if isinstance(value, MessageProgress) else value) if value is not None else None


def includes(value: MessageProgress | datetime | None, timestamp: datetime, message_id: str) -> bool:
    if value is None:
        return False
    cursor = value if isinstance(value, MessageProgress) else MessageProgress(_utc(value))
    return cursor.includes(timestamp, message_id)


async def get_watermark(conversation_id: str, side: str) -> MessageProgress | None:
    key = _key(conversation_id, side)
    try:
        redis = await get_redis()
        raw = await redis.get(key)
    except UnicodeError as exc:
        raise MemoryProgressError('watermark_invalid') from exc
    except Exception as exc:
        raise MemoryProgressError('watermark_read_unavailable') from exc
    if raw is None:
        return None
    try:
        text = raw.decode() if isinstance(raw, bytes) else raw
        if not isinstance(text, str) or len(text) > 2048:
            raise ValueError('invalid_encoding')
        if not text.startswith('{'):
            return MessageProgress(_utc(datetime.fromisoformat(text)), raw=text)
        data = json.loads(text)
        if (type(data.get('v')) is not int or data['v'] != 2
                or data.get('conversation') != conversation_id or data.get('side') != side
                or not isinstance(data.get('revision'), str) or len(data['revision']) != 32):
            raise ValueError('invalid_scope_or_version')
        mid = data['message_id']
        if mid is not None and (not isinstance(mid, str) or not 1 <= len(mid) <= 200):
            raise ValueError('invalid_message_id')
        ts = datetime.fromisoformat(data['timestamp'])
        if ts.tzinfo is None:
            raise ValueError('missing_timezone')
        return MessageProgress(_utc(ts), mid, text)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise MemoryProgressError('watermark_invalid') from exc


async def set_watermark(conversation_id: str, side: str, ts: datetime, *,
                        message_id: str | None = None,
                        expected: MessageProgress | datetime | None | object = _UNSET) -> None:
    """CAS the exact read version. Failure after extraction never means zero writes."""
    key = _key(conversation_id, side)
    if message_id is not None and (not isinstance(message_id, str) or not 1 <= len(message_id) <= 200):
        raise MemoryProgressError('watermark_invalid')
    if expected is _UNSET:
        expected = await get_watermark(conversation_id, side)
    if expected is not None and not isinstance(expected, (MessageProgress, datetime)):
        raise MemoryProgressError('watermark_invalid')
    ts = _utc(ts)
    prior_time = progress_time(expected)
    if prior_time is not None and (ts < prior_time or (ts == prior_time and (
        not isinstance(expected, MessageProgress) or expected.message_id is None
        or (message_id is not None and message_id <= expected.message_id)
    ))):
        raise MemoryProgressError('watermark_conflict')
    previous = expected.raw if isinstance(expected, MessageProgress) else expected.isoformat() if expected else ''
    value = ts.isoformat() if message_id is None else json.dumps({
        'v': 2, 'conversation': conversation_id, 'side': side,
        'timestamp': ts.isoformat(), 'message_id': message_id, 'revision': uuid4().hex})
    try:
        # Even a first-window writer must not recreate private state after an
        # account erasure. This is the existing short DB fence, not a new lock
        # around extraction/provider I/O or per-memory deletion generation.
        async with user_erasure.conversation_write_scope(conversation_id):
            redis = await get_redis()
            result = await redis.eval(_CAS, 1, key, 'present' if expected else 'absent', previous, value, _TTL_SEC)
    except Exception as exc:
        raise MemoryProgressError('watermark_write_unavailable') from exc
    if result != 1:
        raise MemoryProgressError('watermark_conflict')
