"""Redis caching layer for expensive operations.

Caches: embeddings, retrieval results, prompt context, graph queries,
and summarizer results.

Invalidation strategy: per (user, workspace) version counter bumped on any
memory write/delete. The counter is embedded in the cache key so stale
entries become unreachable instantly and expire naturally via TTL — no
SCAN/DEL needed.
"""

import hashlib
import json
import logging

from app.redis_client import get_redis, DEFAULT_TTL

logger = logging.getLogger(__name__)

CACHE_PREFIX = "cache:"
VERSION_PREFIX = "cache_ver:"


def _make_key(namespace: str, identifier: str) -> str:
    """Create a namespaced cache key."""
    h = hashlib.md5(identifier.encode()).hexdigest()[:16]
    return f"{CACHE_PREFIX}{namespace}:{h}"


def _version_key(namespace: str, user_id: str, workspace_id: str | None) -> str:
    return f"{VERSION_PREFIX}{namespace}:{user_id}:{workspace_id or 'none'}"


async def get_cache_version(namespace: str, user_id: str, workspace_id: str | None) -> int:
    """Return the current invalidation epoch for (namespace, user, workspace).

    Missing key = 0 (first read), never fails — if Redis is down we return 0
    and let the caller cache under version 0; next write bumps both sides.
    """
    try:
        redis = await get_redis()
        v = await redis.get(_version_key(namespace, user_id, workspace_id))
        return int(v) if v else 0
    except Exception as e:
        logger.debug(f"Cache version read failed: {e}")
        return 0


async def bump_cache_version(user_id: str, workspace_id: str | None) -> None:
    """Invalidate all per-user caches (retrieval + graph) atomically.

    Call this after any memory write/update/delete/archive in the affected
    (user, workspace) scope. Old cache entries remain in Redis until their
    TTL elapses but become unreachable because the key mixes in the version.

    The two INCRs are pipelined so a concurrent reader can't observe a
    state where only one namespace has been bumped.
    """
    try:
        redis = await get_redis()
        pipe = redis.pipeline()
        pipe.incr(_version_key("ret", user_id, workspace_id))
        pipe.incr(_version_key("graph", user_id, workspace_id))
        await pipe.execute()
    except Exception as e:
        logger.debug(f"Cache version bump failed for {user_id}: {e}")


async def cache_get(namespace: str, identifier: str) -> dict | list | None:
    """Get a cached value by namespace and identifier."""
    redis = await get_redis()
    key = _make_key(namespace, identifier)
    try:
        data = await redis.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.debug(f"Cache get failed for {key}: {e}")
    return None


async def cache_set(
    namespace: str,
    identifier: str,
    value: dict | list,
    ttl: int = DEFAULT_TTL,
) -> None:
    """Set a cached value with TTL."""
    redis = await get_redis()
    key = _make_key(namespace, identifier)
    try:
        await redis.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception as e:
        logger.debug(f"Cache set failed for {key}: {e}")


async def cache_embedding(text: str) -> list[float] | None:
    """Get cached embedding for text."""
    result = await cache_get("emb", text)
    if isinstance(result, list):
        return result
    return None


async def cache_set_embedding(text: str, embedding: list[float]) -> None:
    """Cache an embedding (TTL 30 min)."""
    await cache_set("emb", text, embedding, ttl=1800)


async def cache_retrieval(query: str, user_id: str, workspace_id: str | None = None) -> dict | None:
    """Get cached retrieval results (scoped to current cache version)."""
    ver = await get_cache_version("ret", user_id, workspace_id)
    result = await cache_get("ret", f"v{ver}:{user_id}:{workspace_id or 'none'}:{query}")
    if isinstance(result, dict):
        return result
    return None


async def cache_set_retrieval(
    query: str, user_id: str, results: dict, workspace_id: str | None = None
) -> None:
    """Cache retrieval results (TTL 5 min, keyed by current cache version)."""
    ver = await get_cache_version("ret", user_id, workspace_id)
    await cache_set(
        "ret",
        f"v{ver}:{user_id}:{workspace_id or 'none'}:{query}",
        results,
        ttl=DEFAULT_TTL,
    )


async def cache_graph_context(user_id: str, workspace_id: str | None = None) -> dict | None:
    """Get cached graph context (scoped to current cache version)."""
    ver = await get_cache_version("graph", user_id, workspace_id)
    result = await cache_get("graph", f"v{ver}:{user_id}:{workspace_id or 'none'}")
    if isinstance(result, dict):
        return result
    return None


async def cache_set_graph_context(user_id: str, context: dict, workspace_id: str | None = None) -> None:
    """Cache graph context (TTL 5 min, keyed by current cache version)."""
    ver = await get_cache_version("graph", user_id, workspace_id)
    await cache_set(
        "graph",
        f"v{ver}:{user_id}:{workspace_id or 'none'}",
        context,
        ttl=DEFAULT_TTL,
    )


async def cache_summarizer(conv_hash: str) -> dict | None:
    """Get cached summarizer results."""
    result = await cache_get("sum", conv_hash)
    if isinstance(result, dict):
        return result
    return None


async def cache_set_summarizer(conv_hash: str, results: dict) -> None:
    """Cache summarizer results (TTL 5 min)."""
    await cache_set("sum", conv_hash, results, ttl=DEFAULT_TTL)
