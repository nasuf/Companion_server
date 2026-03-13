import redis.asyncio as redis

from app.config import settings

_pool: redis.ConnectionPool | None = None
_client: redis.Redis | None = None

DEFAULT_TTL = 300  # 5 minutes


async def get_redis() -> redis.Redis:
    global _pool, _client
    if _client is None:
        _pool = redis.ConnectionPool.from_url(
            settings.redis_url, decode_responses=True
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
