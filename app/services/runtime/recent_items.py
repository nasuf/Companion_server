"""跨轮"最近用过"去重的公共 Redis helper。

emoji（reply_post_process）与 sticker（sticker.py）的跨轮去重是同构逻辑：
Redis list 记最近 N 项、读时排除、写时 lpush+ltrim+expire pipeline。
抽到这里统一维护——改存储策略/TTL/异常语义时只动一处。

契约：所有操作失败静默（去重是拟人度增强项，绝不阻塞回复主路径）。
"""

from __future__ import annotations

import logging

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_DEFAULT_TTL_S = 6 * 3600


async def load_recent_items(key: str, keep: int) -> set[str]:
    """读最近 keep 项；Redis 不可用时返回空集（等效于不去重）。"""
    try:
        redis = await get_redis()
        items = await redis.lrange(key, 0, keep - 1)
        return {i.decode() if isinstance(i, bytes) else str(i) for i in items}
    except Exception:
        return set()


async def remember_item(
    key: str, item: str, keep: int, ttl_s: int = _DEFAULT_TTL_S,
) -> None:
    """记录一项（pipeline 一次往返）；失败静默。"""
    if not item:
        return
    try:
        redis = await get_redis()
        pipe = redis.pipeline()
        pipe.lpush(key, item)
        pipe.ltrim(key, 0, keep - 1)
        pipe.expire(key, ttl_s)
        await pipe.execute()
    except Exception:
        pass
