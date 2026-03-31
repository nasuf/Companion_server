"""Delayed reply queue for normal user messages.

Stores normal chat messages in Redis sorted-set based delayed delivery,
so scheduler can consume them later and generate replies asynchronously.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.redis_client import get_redis
from app.services.reply_context import merge_reply_contexts

_DELAYED_LIST_KEY = "delayed:msgs:{cid}"
_DELAYED_ZSET_KEY = "delayed:due"
_DELAYED_TTL = 86400

# Lua script: atomically read + remove due items from per-conversation ZSET,
# then update or clean the global index ZSET.  Prevents race conditions when
# multiple scheduler scans overlap.
_FLUSH_LUA = """
local list_key = KEYS[1]
local index_key = KEYS[2]
local conv_id = ARGV[1]
local due_before = tonumber(ARGV[2])

local rows = redis.call('ZRANGEBYSCORE', list_key, 0, due_before)
if #rows == 0 then
    return nil
end
redis.call('ZREM', list_key, unpack(rows))

local next_due = redis.call('ZRANGE', list_key, 0, 0, 'WITHSCORES')
if #next_due >= 2 then
    redis.call('ZADD', index_key, tonumber(next_due[2]), conv_id)
else
    redis.call('DEL', list_key)
    redis.call('ZREM', index_key, conv_id)
end
return rows
"""

async def enqueue_delayed_message(
    conversation_id: str,
    payload: dict[str, Any],
    delay_seconds: float,
) -> None:
    """Append a delayed message payload and schedule its earliest due time."""
    redis = await get_redis()
    list_key = _DELAYED_LIST_KEY.format(cid=conversation_id)
    due_at = time.time() + max(0.0, delay_seconds)
    stored_payload = dict(payload)
    stored_payload["due_at"] = due_at

    pipe = redis.pipeline()
    pipe.zadd(list_key, {json.dumps(stored_payload, ensure_ascii=False): due_at})
    pipe.expire(list_key, _DELAYED_TTL)
    current_score = await redis.zscore(_DELAYED_ZSET_KEY, conversation_id)
    if current_score is None or due_at < float(current_score):
        pipe.zadd(_DELAYED_ZSET_KEY, {conversation_id: due_at})
    await pipe.execute()


async def flush_due_delayed_messages(conversation_id: str, now: float | None = None) -> list[dict[str, Any]]:
    """Atomically return and clear due delayed payloads for a conversation.

    Uses a Lua script to guarantee that read + remove is a single atomic
    Redis operation, preventing duplicate processing when scheduler scans
    overlap.
    """
    redis = await get_redis()
    list_key = _DELAYED_LIST_KEY.format(cid=conversation_id)
    due_before = now if now is not None else time.time()

    rows = await redis.eval(
        _FLUSH_LUA, 2,
        list_key, _DELAYED_ZSET_KEY,
        conversation_id, str(due_before),
    )
    if not rows:
        return []

    payloads: list[dict[str, Any]] = []
    for row in rows:
        raw = row if isinstance(row, str) else row.decode()
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            payloads.append(item)
    return payloads


async def scan_due_delayed_messages() -> list[tuple[str, list[dict[str, Any]]]]:
    """Scan all due conversations and return their queued payloads."""
    redis = await get_redis()
    now = time.time()
    conv_ids = await redis.zrangebyscore(_DELAYED_ZSET_KEY, 0, now)
    results: list[tuple[str, list[dict[str, Any]]]] = []
    for conv_id in conv_ids:
        cid = conv_id if isinstance(conv_id, str) else conv_id.decode()
        payloads = await flush_due_delayed_messages(cid, now=now)
        if payloads:
            results.append((cid, payloads))
    return results


def merge_delayed_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Merge queued payloads into a single processing task."""
    if not payloads:
        return None

    base = payloads[0]
    latest = payloads[-1]
    reply_context = None
    texts: list[str] = []
    for item in payloads:
        reply_context = merge_reply_contexts(reply_context, item.get("reply_context"))
        text = str(item.get("message", "")).strip()
        if text:
            texts.append(text)

    merged_text = " ".join(texts).strip()
    if not merged_text:
        merged_text = str(latest.get("message", "")).strip()

    return {
        "conversation_id": str(base.get("conversation_id", "")),
        "user_id": str(base.get("user_id", "")),
        "agent_id": str(base.get("agent_id", "")),
        "user_message": merged_text,
        "user_message_id": latest.get("message_id"),
        "reply_context": reply_context,
        "queued_messages": texts,
    }

# --- Concurrency Control ---

_LOCK_KEY = "lock:chat:{cid}"

async def try_lock_conversation(conversation_id: str, ttl: int = 60) -> bool:
    """Try to acquire a distributed lock for a conversation.
    
    Returns True if lock acquired, False otherwise.
    TTL prevents deadlocks if a process crashes.
    """
    redis = await get_redis()
    lock_key = _LOCK_KEY.format(cid=conversation_id)
    # nx=True: only set if key doesn't exist
    return bool(await redis.set(lock_key, "locked", ex=ttl, nx=True))


async def unlock_conversation(conversation_id: str) -> None:
    """Release the lock for a conversation."""
    redis = await get_redis()
    lock_key = _LOCK_KEY.format(cid=conversation_id)
    await redis.delete(lock_key)
