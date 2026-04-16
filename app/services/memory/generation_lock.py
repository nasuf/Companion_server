"""Redis-based lock for one-time initial memory generation per agent.

防止重入: 前端重试/背景任务重入/双实例部署时同一 agent 并发生成 L1。
用 SET NX EX 实现, 带握有者 token 便于安全释放。
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from typing import AsyncIterator

from app.redis_client import get_redis
from app.services.memory.redis_keys import lock_key

logger = logging.getLogger(__name__)

# 完整 L1 生成: profile 直转 + 5 并发 LLM 调用 + retry + 批量 embedding
# + 写库, 单次可能 5-15 min。TTL 30 min 保留足够安全边际。
_DEFAULT_TTL = 1800

# Atomic compare-and-delete: only delete if the stored value still matches our
# token. Prevents deleting another worker's lock when ours has expired.
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


class MemoryGenerationLocked(RuntimeError):
    """另一个进程/任务正在为该 agent 生成 L1 记忆。"""


@contextlib.asynccontextmanager
async def memory_generation_lock(agent_id: str, ttl: int = _DEFAULT_TTL) -> AsyncIterator[None]:
    """Try acquire lock for agent; raise if held. Release atomically on exit."""
    redis = await get_redis()
    key = lock_key(agent_id)
    token = uuid.uuid4().hex
    acquired = await redis.set(key, token, nx=True, ex=ttl)
    if not acquired:
        raise MemoryGenerationLocked(
            f"agent {agent_id} already has a running generation task"
        )
    try:
        yield
    finally:
        try:
            await redis.eval(_RELEASE_LUA, 1, key, token)
        except Exception as e:
            logger.debug(f"memory_generation_lock release failed for {agent_id}: {e}")
