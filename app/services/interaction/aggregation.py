"""碎片化消息聚合服务。

短消息(≤2字且非常用应答词)进入聚合队列，等待5秒窗口后合并处理。
PRD §3.4

Key scope: 所有 key 和 ZSET 成员都以 (agent_id, user_id) 双维度隔离，
防止同一用户并行与两个 agent 会话时碎片串扰（pending:msgs:{A}:{uid}
与 pending:msgs:{B}:{uid} 独立, flush 时互不合并）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.redis_client import get_redis

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
        f"window_sec={_AGGREGATION_WINDOW}"
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
            f"combined={combined[:80]!r}"
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
    return results
