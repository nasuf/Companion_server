"""Per-conversation last-reply-count state (图灵测试条数变化机制).

产品问题: 主回复 LLM 长期偏向每轮 2-3 条, 节奏同质化. web 版
chat.response_instruction 引入"本轮条数 ≠ 上一轮"约束, 但 LLM 自报条数
([X] 标记) 不可靠 (代码 split/cap 之后的实际气泡数才是用户看到的), 且
标记会泄漏给用户.

机制: 代码权威计数 —
- 写: 每次回复持久化时记录本轮累计气泡数 (orchestrator 主/子意图路径按
  reply_index_offset 累计, 最后一次写入即本轮总数; 短路路径单条).
- 读: 下一轮主回复 prompt 构建时取出, 渲染进 chat.reply_count_variation
  段 ("上一轮你回复了 {y} 条, 本轮不能相同").

隔离: key 按 conversation_id 划分 — 每个用户×agent 的会话独立, 多用户
并发互不影响. TTL 6h: 隔太久的"上一轮"没有节奏约束意义 (重逢场景重新开始).

所有失败静默降级 (返回 None / 不写), 绝不阻塞聊天热路径.
"""

from __future__ import annotations

import logging

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_KEY_TMPL = "reply:last_count:{conversation_id}"
_TTL_S = 6 * 3600
_MAX_REASONABLE = 12  # 防御: 异常大的值不可信, 丢弃


def _key(conversation_id: str) -> str:
    return _KEY_TMPL.format(conversation_id=conversation_id)


async def save_last_reply_count(conversation_id: str, count: int) -> None:
    """Record the (cumulative) visible bubble count for this reply turn."""
    if not conversation_id or count <= 0:
        return
    try:
        redis = await get_redis()
        await redis.set(_key(conversation_id), int(count), ex=_TTL_S)
    except Exception as e:
        logger.debug(f"[REPLY-COUNT] save failed for {conversation_id[:8]}: {e}")


async def load_last_reply_count(conversation_id: str) -> int | None:
    """Return last turn's bubble count, or None (no record / Redis down)."""
    if not conversation_id:
        return None
    try:
        redis = await get_redis()
        raw = await redis.get(_key(conversation_id))
    except Exception as e:
        logger.debug(f"[REPLY-COUNT] load failed for {conversation_id[:8]}: {e}")
        return None
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if not (1 <= value <= _MAX_REASONABLE):
        return None
    return value
