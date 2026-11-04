"""One live voice call per conversation, shared across workers."""

from __future__ import annotations

import uuid

from app.redis_client import get_redis

_LOCK_TTL_SECONDS = 30

_REFRESH = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


def _key(conversation_id: str) -> str:
    return "voice:live:" + conversation_id


async def acquire_call(conversation_id: str) -> str | None:
    token = uuid.uuid4().hex
    redis = await get_redis()
    created = await redis.set(_key(conversation_id), token, nx=True, ex=_LOCK_TTL_SECONDS)
    return token if created else None


async def refresh_call(conversation_id: str, token: str) -> bool:
    redis = await get_redis()
    refreshed = await redis.eval(_REFRESH, 1, _key(conversation_id), token, _LOCK_TTL_SECONDS)
    return bool(refreshed)


async def release_call(conversation_id: str, token: str) -> None:
    redis = await get_redis()
    await redis.eval(_RELEASE, 1, _key(conversation_id), token)
