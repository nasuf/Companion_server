"""Conversation-local crisis care state.

This is deliberately separate from intent detection.  Once a user has entered
the crisis path, follow-up turns stay in a safety-aware mode until the user
explicitly releases it or the state expires.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_CRISIS_CARE_TTL_SECONDS = 6 * 60 * 60


def _crisis_care_key(conversation_id: str, user_id: str) -> str:
    return f"chat:crisis_care:{conversation_id}:{user_id}"


async def load_crisis_care_state(conversation_id: str, user_id: str) -> dict[str, Any] | None:
    """Return active crisis-care state if present. Redis is best-effort."""
    try:
        redis = await get_redis()
        raw = await redis.get(_crisis_care_key(conversation_id, user_id))
        if not raw:
            return None
        data = json.loads(raw)
        context = str(data.get("context") or "").strip()
        release_count = data.get("release_count", 0)
        try:
            release_count = int(release_count)
        except (TypeError, ValueError):
            release_count = 0
        source = data.get("source")
        return {
            "context": context or "(recent crisis care active)",
            "source": str(source) if source else None,
            "release_count": max(0, release_count),
        }
    except Exception as e:
        logger.warning(f"load crisis care state failed: {e}")
        return None


async def load_crisis_care_context(conversation_id: str, user_id: str) -> str | None:
    """Return active crisis-care context if present. Redis is best-effort."""
    state = await load_crisis_care_state(conversation_id, user_id)
    if not state:
        return None
    return str(state.get("context") or "").strip() or "(recent crisis care active)"


async def get_crisis_care_status(conversation_id: str, user_id: str) -> dict[str, Any]:
    """Return UI-safe crisis-care status for a conversation."""
    try:
        redis = await get_redis()
        key = _crisis_care_key(conversation_id, user_id)
        raw = await redis.get(key)
        if not raw:
            return {
                "active": False,
                "unavailable": False,
                "source": None,
                "release_count": 0,
                "ttl_seconds": None,
                "context_preview": None,
            }
        data = json.loads(raw)
        context = str(data.get("context") or "").strip()
        ttl = await redis.ttl(key)
        source = data.get("source")
        return {
            "active": True,
            "unavailable": False,
            "source": str(source) if source else None,
            "release_count": int(data.get("release_count") or 0),
            "ttl_seconds": ttl if isinstance(ttl, int) and ttl >= 0 else None,
            "context_preview": context[-160:] if context else None,
        }
    except Exception as e:
        logger.warning(f"get crisis care status failed: {e}")
        return {
            "active": False,
            "unavailable": True,
            "source": None,
            "release_count": 0,
            "ttl_seconds": None,
            "context_preview": None,
        }


async def mark_crisis_care_active(
    conversation_id: str,
    user_id: str,
    *,
    context: str,
    source: str,
    release_count: int = 0,
) -> None:
    """Persist active crisis-care state with a bounded TTL."""
    try:
        redis = await get_redis()
        payload: dict[str, Any] = {
            "context": context[-1200:],
            "source": source,
            "release_count": max(0, int(release_count)),
        }
        await redis.set(
            _crisis_care_key(conversation_id, user_id),
            json.dumps(payload, ensure_ascii=False),
            ex=_CRISIS_CARE_TTL_SECONDS,
        )
    except Exception as e:
        logger.warning(f"mark crisis care state failed: {e}")


async def clear_crisis_care_state(conversation_id: str, user_id: str) -> None:
    """Clear crisis-care state when the user explicitly releases it."""
    try:
        redis = await get_redis()
        await redis.delete(_crisis_care_key(conversation_id, user_id))
    except Exception as e:
        logger.warning(f"clear crisis care state failed: {e}")
