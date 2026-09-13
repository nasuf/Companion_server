"""Redis cache for proactive trending snippets (shared across agents)."""

from __future__ import annotations

import hashlib
import logging

from app.redis_client import get_redis
from app.services.runtime_config import resolve_config_sync

logger = logging.getLogger(__name__)

_GLOBAL_CACHE_KEY = "proactive:trending:global:v1"
_GENERIC_TOPIC_SEEDS = frozenset({
    "",
    "公共话题",
    "问候",
    "日常",
    "分享有趣见闻",
    "询问近况",
    "日常琐事",
})


def cache_key_for_topic(topic: str | None) -> str:
    seed = (topic or "").strip()
    if seed in _GENERIC_TOPIC_SEEDS:
        return _GLOBAL_CACHE_KEY
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"proactive:trending:topic:{digest}"


async def get_cached_trending(key: str) -> str | None:
    try:
        redis = await get_redis()
        raw = await redis.get(key)
    except Exception as exc:  # noqa: BLE001 — cache miss must not break send path
        logger.warning("[proactive-trending] cache read failed key=%s: %s", key, exc)
        return None
    if not raw:
        return None
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    cleaned = text.strip()
    return cleaned or None


async def set_cached_trending(key: str, text: str) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    ttl = resolve_config_sync(agent_id=None).proactive_trending_cache_ttl_s
    try:
        redis = await get_redis()
        await redis.set(key, cleaned, ex=ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proactive-trending] cache write failed key=%s: %s", key, exc)
