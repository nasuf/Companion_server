"""碎片化消息聚合服务。

短消息(≤2字且非常用应答词)进入聚合队列，等待5秒窗口后合并处理。
PRD §3.4
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.redis_client import get_redis

# 常用应答词集合（不参与聚合，直接处理）
COMMON_RESPONSES = {
    "嗯", "哦", "啊", "好", "行", "对", "是", "哈", "嘿",
    "呢", "吧", "呀", "噢", "嗯嗯", "好的", "可以", "没事",
    "ok", "OK", "好吧", "知道了", "明白", "收到", "是的",
    "对的", "没有", "你好", "您好",
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

_PENDING_MSG_KEY = "pending:msgs:{uid}"
_PENDING_CONV_KEY = "pending:conv:{uid}"
_PENDING_CTX_KEY = "pending:ctx:{uid}"
_PENDING_DELAYED_KEY = "pending:delayed"
_AGGREGATION_WINDOW = 5  # seconds
_PENDING_TTL = 30  # seconds, fallback TTL


def is_short_message(text: str) -> bool:
    """PRD §3.4: len≤2 且不在常用应答词集合。"""
    text = text.strip()
    if text in COMMON_RESPONSES:
        return False
    return len(text) <= 2


async def push_pending(
    user_id: str,
    conversation_id: str,
    text: str,
    reply_context: dict | None = None,
    message_id: str | None = None,
) -> None:
    """将碎片消息加入聚合队列。"""
    r = await get_redis()
    msg_key = _PENDING_MSG_KEY.format(uid=user_id)
    conv_key = _PENDING_CONV_KEY.format(uid=user_id)
    ctx_key = _PENDING_CTX_KEY.format(uid=user_id)

    pipe = r.pipeline()
    payload: dict[str, Any] = {"text": text}
    if message_id:
        payload["message_id"] = message_id
    pipe.rpush(msg_key, json.dumps(payload, ensure_ascii=False))
    pipe.expire(msg_key, _PENDING_TTL)
    pipe.set(conv_key, conversation_id, ex=_PENDING_TTL)
    if reply_context:
        pipe.set(ctx_key, json.dumps(reply_context, ensure_ascii=False), ex=_PENDING_TTL)
    pipe.zadd(_PENDING_DELAYED_KEY, {user_id: time.time() + _AGGREGATION_WINDOW})
    await pipe.execute()


async def flush_pending(user_id: str) -> tuple[str | None, str | None, dict | None, str | None]:
    """取出并清空聚合队列。返回 (合并文本, conversation_id, reply_context, latest_message_id)。"""
    r = await get_redis()
    msg_key = _PENDING_MSG_KEY.format(uid=user_id)
    conv_key = _PENDING_CONV_KEY.format(uid=user_id)
    ctx_key = _PENDING_CTX_KEY.format(uid=user_id)

    result = await r.eval(
        _AGGREGATE_LUA, 4,
        msg_key, _PENDING_DELAYED_KEY, conv_key, ctx_key,
        user_id,
    )
    if not result:
        return None, None, None, None

    items = [m if isinstance(m, str) else m.decode() for m in result]
    conv_id = items[0] if items else None
    raw_ctx = items[1] if len(items) > 1 else None
    raw_msgs = items[2:] if len(items) > 2 else []
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
    return " ".join(texts) if texts else None, conv_id, ctx, latest_message_id


async def has_pending(user_id: str) -> bool:
    """检查用户是否有待聚合消息。"""
    r = await get_redis()
    return await r.exists(_PENDING_MSG_KEY.format(uid=user_id)) > 0


async def scan_expired() -> list[tuple[str, str, str, dict | None, str | None]]:
    """扫描到期的聚合窗口。返回 [(user_id, combined_text, conversation_id, reply_context, latest_message_id)]。"""
    r = await get_redis()
    now = time.time()
    expired = await r.zrangebyscore(_PENDING_DELAYED_KEY, 0, now)
    results = []
    for uid in expired:
        if isinstance(uid, bytes):
            uid = uid.decode()
        text, conv_id, ctx, latest_message_id = await flush_pending(uid)
        if text and conv_id:
            results.append((uid, text, conv_id, ctx, latest_message_id))
    return results
