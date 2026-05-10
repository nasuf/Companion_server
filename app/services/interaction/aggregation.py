"""Low-level Redis storage for user-message aggregation windows.

Public chat entrypoints should use `user_turn_aggregation.py` instead of calling
this module directly.  This file owns the Redis key layout and primitive
operations for the two internal strategies:

- fragment window: incomplete ≤2 character fragments, 5 second window
- turn window: rapid complete user messages, short quiet window

Key scope: 所有 key 和 ZSET 成员都以 (agent_id, user_id) 双维度隔离，
防止同一用户并行与两个 agent 会话时碎片串扰（pending:msgs:{A}:{uid}
与 pending:msgs:{B}:{uid} 独立, flush 时互不合并）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.observability.events import EVT_AGG_FLUSHED, EVT_AGG_PUSHED, EVT_AGG_SCAN
from app.redis_client import get_redis
from app.services.interaction.reply_context import merge_reply_contexts
from app.services.interaction.turn_coalescing import coalesce_turn_messages

logger = logging.getLogger(__name__)

# 常用应答词集合（spec §1.2）：长度≤2 但属于应答词 → 不视为碎片，直接进入边界系统。
COMMON_RESPONSES = {
    "嗯", "哦", "啊", "噢", "哎", "喂", "嗨", "哈",
    "好", "是", "不", "对", "行",
    "ok", "OK", "嗯嗯", "好的", "是的", "对啊", "不行",
    "可以", "没事", "知道", "明白",
    "额", "呃", "嗯哼",
}

# Lua脚本: 原子读取conv_id + 取出msgs + 清理所有key
_AGGREGATE_LUA = """
local msgs = redis.call('LRANGE', KEYS[1], 0, -1)
if #msgs == 0 then
    return nil
end
local conv_id = redis.call('GET', KEYS[3])
local ctx = redis.call('GET', KEYS[4])
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('DEL', KEYS[3])
redis.call('DEL', KEYS[4])
return {conv_id, ctx, unpack(msgs)}
"""

_PENDING_MSG_KEY = "pending:msgs:{aid}:{uid}"
_PENDING_CONV_KEY = "pending:conv:{aid}:{uid}"
_PENDING_CTX_KEY = "pending:ctx:{aid}:{uid}"
_PENDING_DELAYED_KEY = "pending:delayed"
_AGGREGATION_WINDOW = 5  # seconds
_PENDING_TTL = 30  # seconds, fallback TTL

_TURN_MSG_KEY = "turn:msgs:{aid}:{uid}"
_TURN_CONV_KEY = "turn:conv:{aid}:{uid}"
_TURN_FIRST_AT_KEY = "turn:first_at:{aid}:{uid}"
_TURN_DELAYED_KEY = "turn:delayed"
_TURN_QUIET_WINDOW = 1.2
_TURN_MAX_WAIT = 4.0
_TURN_TTL = 30


def _scope_token(agent_id: str, user_id: str) -> str:
    """ZSET 成员编码：{agent_id}:{user_id}, UUID 不含 ':' 可安全 split."""
    return f"{agent_id}:{user_id}"


def _parse_scope_token(token: str) -> tuple[str, str] | None:
    """scan_expired 解码 ZSET 成员。非法格式 → None 跳过。"""
    if ":" not in token:
        return None
    agent_id, _, user_id = token.partition(":")
    if not agent_id or not user_id:
        return None
    return agent_id, user_id


def is_short_message(text: str) -> bool:
    """PRD §3.4: len≤2 且不在常用应答词集合。"""
    text = text.strip()
    if text in COMMON_RESPONSES:
        return False
    return len(text) <= 2


async def push_pending(
    *,
    agent_id: str,
    user_id: str,
    conversation_id: str,
    text: str,
    reply_context: dict | None = None,
    message_id: str | None = None,
) -> bool:
    """将碎片消息加入 (agent, user) scoped 聚合队列。

    kwargs-only: 三个前置位置参数都是 UUID 字符串, 位置传参很容易把
    agent_id / user_id / conversation_id 写反导致静默路由错误。

    Returns True 表示成功入队 (caller 可安心回 'aggregating'), False 表示
    Redis 挂, caller 应走同步 LLM 跳过聚合 (避免用户长时间看不到回应).
    """
    r = await get_redis()
    msg_key = _PENDING_MSG_KEY.format(aid=agent_id, uid=user_id)
    conv_key = _PENDING_CONV_KEY.format(aid=agent_id, uid=user_id)
    ctx_key = _PENDING_CTX_KEY.format(aid=agent_id, uid=user_id)
    token = _scope_token(agent_id, user_id)

    pipe = r.pipeline()
    payload: dict[str, Any] = {"text": text}
    if message_id:
        payload["message_id"] = message_id
    pipe.rpush(msg_key, json.dumps(payload, ensure_ascii=False))
    pipe.expire(msg_key, _PENDING_TTL)
    pipe.set(conv_key, conversation_id, ex=_PENDING_TTL)
    if reply_context:
        pipe.set(ctx_key, json.dumps(reply_context, ensure_ascii=False), ex=_PENDING_TTL)
    pipe.zadd(_PENDING_DELAYED_KEY, {token: time.time() + _AGGREGATION_WINDOW})
    try:
        await pipe.execute()
    except Exception as e:
        logger.warning(
            f"[AGG-PUSH] Redis push failed agent_id={agent_id} user_id={user_id}: {e}"
        )
        return False
    logger.info(
        f"[AGG-PUSH] agent_id={agent_id} user_id={user_id} text={text!r} "
        f"window_sec={_AGGREGATION_WINDOW}",
        extra={
            "event": EVT_AGG_PUSHED,
            "fragment_len": len(text),
            "window_sec": _AGGREGATION_WINDOW,
        },
    )
    return True


async def flush_pending(
    *, agent_id: str, user_id: str,
) -> tuple[str | None, str | None, dict | None, str | None]:
    """取出并清空 (agent, user) scoped 聚合队列。返回 (合并文本, conversation_id, reply_context, latest_message_id)。"""
    r = await get_redis()
    msg_key = _PENDING_MSG_KEY.format(aid=agent_id, uid=user_id)
    conv_key = _PENDING_CONV_KEY.format(aid=agent_id, uid=user_id)
    ctx_key = _PENDING_CTX_KEY.format(aid=agent_id, uid=user_id)
    token = _scope_token(agent_id, user_id)

    try:
        result = await r.eval(
            _AGGREGATE_LUA, 4,
            msg_key, _PENDING_DELAYED_KEY, conv_key, ctx_key,
            token,
        )
    except Exception as e:
        logger.warning(
            f"[AGG-FLUSH] Redis eval failed agent_id={agent_id} user_id={user_id}: {e}"
        )
        return None, None, None, None
    if not result:
        return None, None, None, None

    def _coerce(m):
        if m is None or m is False:
            return None
        return m if isinstance(m, str) else m.decode()

    items = [_coerce(m) for m in result]
    conv_id = items[0] if items else None
    raw_ctx = items[1] if len(items) > 1 else None
    raw_msgs = [m for m in items[2:] if m is not None]
    ctx = None
    if raw_ctx:
        try:
            ctx = json.loads(raw_ctx)
        except json.JSONDecodeError:
            ctx = None
    texts: list[str] = []
    latest_message_id: str | None = None
    for raw in raw_msgs:
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            item = {"text": raw}
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
        msg_id = item.get("message_id")
        if isinstance(msg_id, str) and msg_id.strip():
            latest_message_id = msg_id
    # spec §1.5: 按原始顺序直接连接（中文不加空格）
    combined = "".join(texts) if texts else None
    if combined:
        logger.info(
            f"[AGG-FLUSH] agent_id={agent_id} user_id={user_id} parts={len(texts)} "
            f"combined={combined[:80]!r}",
            extra={
                "event": EVT_AGG_FLUSHED,
                "n_parts": len(texts),
                "combined_len": len(combined),
            },
        )
    return combined, conv_id, ctx, latest_message_id


async def scan_expired() -> list[tuple[str, str, str, str, dict | None, str | None]]:
    """扫描到期的聚合窗口。返回 [(agent_id, user_id, combined_text, conversation_id, reply_context, latest_message_id)]。"""
    r = await get_redis()
    now = time.time()
    try:
        expired = await r.zrangebyscore(_PENDING_DELAYED_KEY, 0, now)
    except Exception as e:
        # scheduler 每秒跑一次, 单次失败跳过, 下一 tick 再试
        logger.warning(f"[AGG-SCAN] Redis zrangebyscore failed: {e}")
        return []
    results = []
    for raw in expired:
        token = raw.decode() if isinstance(raw, bytes) else raw
        parsed = _parse_scope_token(token)
        if parsed is None:
            # 非 "{agent_id}:{user_id}" 格式成员直接 zrem, 防止无限循环
            await r.zrem(_PENDING_DELAYED_KEY, token)
            continue
        agent_id, user_id = parsed
        text, conv_id, ctx, latest_message_id = await flush_pending(agent_id=agent_id, user_id=user_id)
        if text and conv_id:
            results.append((agent_id, user_id, text, conv_id, ctx, latest_message_id))
    if results:
        # 仅有命中才打 — 否则 1s/tick 会刷屏 (scheduler 调度间隔)
        logger.debug(
            f"[AGG-SCAN] flushed {len(results)} expired window(s)",
            extra={"event": EVT_AGG_SCAN, "n_flushed": len(results)},
        )
    return results


async def push_turn_pending(
    *,
    agent_id: str,
    user_id: str,
    conversation_id: str,
    text: str,
    reply_context: dict | None = None,
    message_id: str | None = None,
) -> bool:
    """Append a normal user message to the short turn-level quiet window."""
    r = await get_redis()
    msg_key = _TURN_MSG_KEY.format(aid=agent_id, uid=user_id)
    conv_key = _TURN_CONV_KEY.format(aid=agent_id, uid=user_id)
    first_key = _TURN_FIRST_AT_KEY.format(aid=agent_id, uid=user_id)
    token = _scope_token(agent_id, user_id)
    now = time.time()
    try:
        raw_first_at = await r.get(first_key)
    except Exception as e:
        logger.warning(
            f"[TURN-PUSH] Redis get failed agent_id={agent_id} user_id={user_id}: {e}"
        )
        return False
    try:
        first_at = float(raw_first_at if isinstance(raw_first_at, str) else raw_first_at.decode())
    except (AttributeError, TypeError, ValueError):
        first_at = now

    due_at = min(now + _TURN_QUIET_WINDOW, first_at + _TURN_MAX_WAIT)
    payload: dict[str, Any] = {"text": text}
    if message_id:
        payload["message_id"] = message_id
    if reply_context:
        payload["reply_context"] = reply_context

    pipe = r.pipeline()
    pipe.rpush(msg_key, json.dumps(payload, ensure_ascii=False))
    pipe.expire(msg_key, _TURN_TTL)
    pipe.set(conv_key, conversation_id, ex=_TURN_TTL)
    if raw_first_at is None:
        pipe.set(first_key, str(first_at), ex=_TURN_TTL)
    pipe.zadd(_TURN_DELAYED_KEY, {token: due_at})
    try:
        await pipe.execute()
    except Exception as e:
        logger.warning(
            f"[TURN-PUSH] Redis push failed agent_id={agent_id} user_id={user_id}: {e}"
        )
        return False
    logger.info(
        f"[TURN-PUSH] agent_id={agent_id} user_id={user_id} text={text[:60]!r} "
        f"due_in={max(0, due_at - now):.2f}s",
        extra={
            "event": EVT_AGG_PUSHED,
            "fragment_len": len(text),
            "window_sec": _TURN_QUIET_WINDOW,
        },
    )
    return True


async def has_turn_pending(*, agent_id: str, user_id: str) -> bool:
    """Return whether a normal turn quiet window is already open."""
    r = await get_redis()
    first_key = _TURN_FIRST_AT_KEY.format(aid=agent_id, uid=user_id)
    try:
        return bool(await r.get(first_key))
    except Exception as e:
        logger.warning(
            f"[TURN-CHECK] Redis get failed agent_id={agent_id} user_id={user_id}: {e}"
        )
        return False


async def flush_turn_pending(
    *, agent_id: str, user_id: str,
) -> tuple[str | None, str | None, dict | None, str | None]:
    """Flush one normal user turn. Returns (combined_text, conv_id, merged_context, latest_message_id)."""
    r = await get_redis()
    msg_key = _TURN_MSG_KEY.format(aid=agent_id, uid=user_id)
    conv_key = _TURN_CONV_KEY.format(aid=agent_id, uid=user_id)
    first_key = _TURN_FIRST_AT_KEY.format(aid=agent_id, uid=user_id)
    token = _scope_token(agent_id, user_id)
    try:
        raw_msgs = await r.lrange(msg_key, 0, -1)
        conv_id = await r.get(conv_key)
        if not raw_msgs:
            await r.delete(msg_key, conv_key, first_key)
            await r.zrem(_TURN_DELAYED_KEY, token)
            return None, None, None, None
        await r.delete(msg_key, conv_key, first_key)
        await r.zrem(_TURN_DELAYED_KEY, token)
    except Exception as e:
        logger.warning(
            f"[TURN-FLUSH] Redis flush failed agent_id={agent_id} user_id={user_id}: {e}"
        )
        return None, None, None, None

    def _coerce(value):
        if value is None or value is False:
            return None
        return value if isinstance(value, str) else value.decode()

    texts: list[str] = []
    message_ids: list[str] = []
    reply_context = None
    latest_message_id: str | None = None
    for raw in raw_msgs:
        item_raw = _coerce(raw)
        if item_raw is None:
            continue
        try:
            item = json.loads(item_raw)
        except json.JSONDecodeError:
            item = {"text": item_raw}
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
        reply_context = merge_reply_contexts(reply_context, item.get("reply_context"))
        msg_id = item.get("message_id")
        if isinstance(msg_id, str) and msg_id.strip():
            latest_message_id = msg_id
            message_ids.append(msg_id)

    coalesced_turn = coalesce_turn_messages(texts)
    combined = coalesced_turn.combined_text
    if message_ids or coalesced_turn.metadata:
        reply_context = dict(reply_context or {})
        if message_ids:
            reply_context["turn_message_ids"] = message_ids
        if coalesced_turn.metadata:
            reply_context["turn_coalescing"] = coalesced_turn.metadata
    if combined:
        logger.info(
            f"[TURN-FLUSH] agent_id={agent_id} user_id={user_id} parts={len(texts)} "
            f"combined={combined[:80]!r}",
            extra={
                "event": EVT_AGG_FLUSHED,
                "n_parts": len(texts),
                "n_coalesced": len(coalesced_turn.coalesced),
                "combined_len": len(combined),
            },
        )
    return combined, _coerce(conv_id), reply_context, latest_message_id


async def scan_turn_expired() -> list[tuple[str, str, str, str, dict | None, str | None]]:
    """Scan normal turn quiet windows that are due."""
    r = await get_redis()
    now = time.time()
    try:
        expired = await r.zrangebyscore(_TURN_DELAYED_KEY, 0, now)
    except Exception as e:
        logger.warning(f"[TURN-SCAN] Redis zrangebyscore failed: {e}")
        return []
    results = []
    for raw in expired:
        token = raw.decode() if isinstance(raw, bytes) else raw
        parsed = _parse_scope_token(token)
        if parsed is None:
            await r.zrem(_TURN_DELAYED_KEY, token)
            continue
        agent_id, user_id = parsed
        text, conv_id, ctx, latest_message_id = await flush_turn_pending(
            agent_id=agent_id,
            user_id=user_id,
        )
        if text and conv_id:
            results.append((agent_id, user_id, text, conv_id, ctx, latest_message_id))
    if results:
        logger.debug(
            f"[TURN-SCAN] flushed {len(results)} quiet window(s)",
            extra={"event": EVT_AGG_SCAN, "n_flushed": len(results)},
        )
    return results
