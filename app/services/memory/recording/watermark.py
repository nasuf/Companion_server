"""Per (conversation, side) 抽取水位线.

spec §2.1 / §2.2 管线默认窗口 6 条滑动覆盖, 同一条消息每轮都会被 LLM 抽一次,
embedding dedup 兜底 DB 不脏但 LLM token 浪费 ~3x. 加水位线后:
- 只对 createdAt > watermark 的消息做抽取
- 其余 messages[-6:] 仅作为【历史上下文】提供指代/情境
- 抽取成功后 watermark 推进到本次最新消息的 createdAt

Redis-only 存储; 丢失即退化回 "全部抽一次" 的老行为, is_duplicate 仍然兜底.
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_KEY_FMT = "mem:extract:wm:{conv}:{side}"
_TTL_SEC = 30 * 86400  # 30 天, 远大于常规会话周期


def _key(conversation_id: str, side: str) -> str:
    return _KEY_FMT.format(conv=conversation_id, side=side)


async def get_watermark(conversation_id: str, side: str) -> datetime | None:
    """读水位线. 无或异常 → None (调用方应按"全部抽"处理)."""
    try:
        redis = await get_redis()
        raw = await redis.get(_key(conversation_id, side))
    except Exception as e:
        logger.warning(f"[MEM-WM] get failed conv={conversation_id[:8]} side={side}: {e}")
        return None
    if raw is None:
        return None
    try:
        text = raw.decode() if isinstance(raw, bytes) else raw
        return datetime.fromisoformat(text)
    except (ValueError, TypeError) as e:
        logger.warning(f"[MEM-WM] parse failed value={raw!r}: {e}")
        return None


async def set_watermark(conversation_id: str, side: str, ts: datetime) -> None:
    """写水位线. Redis 挂仅 warning, 下次抽取退化回 "全部抽"."""
    try:
        redis = await get_redis()
        await redis.set(_key(conversation_id, side), ts.isoformat(), ex=_TTL_SEC)
    except Exception as e:
        logger.warning(f"[MEM-WM] set failed conv={conversation_id[:8]} side={side}: {e}")
