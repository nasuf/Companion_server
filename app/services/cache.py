"""Redis caching layer for expensive operations.

Caches: embeddings, retrieval results, prompt context, graph queries,
and summarizer results.
"""

import hashlib
import json
import logging

from app.redis_client import get_redis, DEFAULT_TTL

logger = logging.getLogger(__name__)

CACHE_PREFIX = "cache:"


def _make_key(namespace: str, identifier: str) -> str:
    """Create a namespaced cache key."""
    h = hashlib.md5(identifier.encode()).hexdigest()[:16]
    return f"{CACHE_PREFIX}{namespace}:{h}"


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


async def cache_retrieval(query: str, user_id: str) -> dict | None:
    """Get cached retrieval results."""
    result = await cache_get("ret", f"{user_id}:{query}")
    if isinstance(result, dict):
        return result
    return None


async def cache_set_retrieval(
    query: str, user_id: str, results: dict
) -> None:
    """Cache retrieval results (TTL 5 min)."""
    await cache_set("ret", f"{user_id}:{query}", results, ttl=DEFAULT_TTL)


async def cache_graph_context(user_id: str) -> dict | None:
    """Get cached graph context."""
    result = await cache_get("graph", user_id)
    if isinstance(result, dict):
        return result
    return None


async def cache_set_graph_context(user_id: str, context: dict) -> None:
    """Cache graph context (TTL 5 min)."""
    await cache_set("graph", user_id, context, ttl=DEFAULT_TTL)


async def cache_summarizer(conv_hash: str) -> dict | None:
    """Get cached summarizer results."""
    result = await cache_get("sum", conv_hash)
    if isinstance(result, dict):
        return result
    return None


async def cache_set_summarizer(conv_hash: str, results: dict) -> None:
    """Cache summarizer results (TTL 5 min)."""
    await cache_set("sum", conv_hash, results, ttl=DEFAULT_TTL)
