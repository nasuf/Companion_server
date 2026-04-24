import logging

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)

_pool: redis.ConnectionPool | None = None
_client: redis.Redis | None = None

# Readonly mode 状态: 启动时 Redis 连失败, 或运行时 redis_health 失败 → False.
# require_redis 依赖据此拒绝写请求 (503). scheduler cron 每 30s 重检, 允许自愈.
_redis_healthy: bool = True

DEFAULT_TTL = 300  # 5 minutes


async def get_redis() -> redis.Redis:
    global _pool, _client
    if _client is None:
        _pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout_s,
            socket_connect_timeout=settings.redis_connect_timeout_s,
            max_connections=settings.redis_max_connections,
        )
        _client = redis.Redis(connection_pool=_pool)
    return _client


async def close_redis():
    global _pool, _client
    if _client:
        await _client.aclose()
        _client = None
    if _pool:
        await _pool.aclose()
        _pool = None


async def redis_health() -> bool:
    try:
        r = await get_redis()
        return await r.ping()
    except Exception:
        return False


def is_redis_healthy() -> bool:
    """require_redis dependency / ws handler 消费此函数判定是否接受写请求."""
    return _redis_healthy


def mark_redis_healthy(healthy: bool) -> None:
    """startup + scheduler cron 调用, 显式更新全局健康状态."""
    global _redis_healthy
    if _redis_healthy != healthy:
        logger.warning(f"[REDIS-HEALTH] state transition: {_redis_healthy} -> {healthy}")
    _redis_healthy = healthy


async def recheck_redis_health() -> bool:
    """ping 一次并更新 _redis_healthy; scheduler 每 30s 调用, 允许自愈."""
    new_state = await redis_health()
    mark_redis_healthy(new_state)
    return new_state
