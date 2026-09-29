"""Redis cache for proactive trending snippets (shared across agents)."""

from __future__ import annotations

import hashlib
import json
import logging

from app.redis_client import get_redis
from app.services.runtime_config import resolve_config_sync

logger = logging.getLogger(__name__)

# v2: 连同结构化候选一起缓存 —— 热点筛选 (trending_pick) 按候选挑, 只有文本的
# v1 缓存命中时等于没有热点
_GLOBAL_CACHE_KEY = "proactive:trending:global:v2"
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
    return f"proactive:trending:topic:v2:{digest}"


async def get_cached_trending(key: str) -> tuple[str, tuple[dict, ...]] | None:
    """→ (text, candidates); 未命中 / 读失败 / 内容损坏返回 None."""
    try:
        redis = await get_redis()
        raw = await redis.get(key)
    except Exception as exc:  # noqa: BLE001 — cache miss must not break send path
        logger.warning("[proactive-trending] cache read failed key=%s: %s", key, exc)
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        text = str(payload.get("text") or "").strip()
        candidates = tuple(c for c in payload.get("candidates") or () if isinstance(c, dict))
    except (TypeError, ValueError, AttributeError):
        return None
    return (text, candidates) if text else None


async def set_cached_trending(key: str, text: str, candidates: tuple[dict, ...] = ()) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    ttl = resolve_config_sync(agent_id=None).proactive_trending_cache_ttl_s
    payload = json.dumps({"text": cleaned, "candidates": list(candidates)}, ensure_ascii=False)
    try:
        redis = await get_redis()
        await redis.set(key, payload, ex=ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[proactive-trending] cache write failed key=%s: %s", key, exc)
