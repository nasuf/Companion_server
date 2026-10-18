"""Per (conversation, side) 抽取水位线.

spec §2.1 / §2.2 管线默认窗口 6 条滑动覆盖, 同一条消息每轮都会被 LLM 抽一次,
embedding dedup 兜底 DB 不脏但 LLM token 浪费 ~3x. 加水位线后:
- 只对 createdAt > watermark 的消息做抽取
- 其余 messages[-6:] 仅作为【历史上下文】提供指代/情境
- 抽取成功后 watermark 推进到本次最新消息的 createdAt

Redis-only 存储。缺失只处理调用方最近窗口；读取/写入失败不能伪装为缺失或成功。
该进度不是持久任务凭据，不能据此宣称重放幂等。
"""

from __future__ import annotations

from datetime import datetime

from app.redis_client import get_redis
from app.services.memory.recording.outcomes import MemoryProgressError

_KEY_FMT = "mem:extract:wm:{conv}:{side}"
_TTL_SEC = 30 * 86400  # 30 天, 远大于常规会话周期


def _key(conversation_id: str, side: str) -> str:
    return _KEY_FMT.format(conv=conversation_id, side=side)


async def get_watermark(conversation_id: str, side: str) -> datetime | None:
    """Only an absent key means no watermark; unavailable/corrupt state fails closed."""
    try:
        redis = await get_redis()
        raw = await redis.get(_key(conversation_id, side))
    except Exception as e:
        raise MemoryProgressError("watermark_read_unavailable") from e
    if raw is None:
        return None
    try:
        text = raw.decode() if isinstance(raw, bytes) else raw
        return datetime.fromisoformat(text)
    except (ValueError, TypeError) as e:
        raise MemoryProgressError("watermark_invalid") from e


async def set_watermark(conversation_id: str, side: str, ts: datetime) -> None:
    """A failed write leaves completion uncertain and must be reported to the caller."""
    try:
        redis = await get_redis()
        await redis.set(_key(conversation_id, side), ts.isoformat(), ex=_TTL_SEC)
    except Exception as e:
        raise MemoryProgressError("watermark_write_unavailable") from e
