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


def _scope(value: str | None) -> str:
    return str(value or "_none").replace(":", "_")


def _crisis_care_key(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
) -> str:
    return ":".join([
        "chat",
        "crisis_care",
        _scope(workspace_id),
        _scope(agent_id),
        _scope(conversation_id),
        _scope(user_id),
    ])


def _coerce_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


async def load_crisis_care_state(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
) -> dict[str, Any] | None:
    """Return active crisis-care state if present. Redis is best-effort."""
    try:
        redis = await get_redis()
        raw = await redis.get(_crisis_care_key(
            conversation_id,
            user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
        ))
        if not raw:
            return None
        data = json.loads(raw)
        context = str(data.get("context") or "").strip()
        source = data.get("source")
        return {
            "context": context or "(recent crisis care active)",
            "source": str(source) if source else None,
            "release_count": _coerce_nonnegative_int(data.get("release_count")),
            "aftercare_turn_count": _coerce_nonnegative_int(
                data.get("aftercare_turn_count"),
            ),
            "turns_since_safety_check": _coerce_nonnegative_int(
                data.get("turns_since_safety_check"),
            ),
            "workspace_id": str(data.get("workspace_id") or workspace_id or ""),
            "agent_id": str(data.get("agent_id") or agent_id or ""),
        }
    except Exception as e:
        logger.warning(f"load crisis care state failed: {e}")
        return None


async def load_crisis_care_context(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
) -> str | None:
    """Return active crisis-care context if present. Redis is best-effort."""
    state = await load_crisis_care_state(
        conversation_id,
        user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    if not state:
        return None
    return str(state.get("context") or "").strip() or "(recent crisis care active)"


async def get_crisis_care_status(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
) -> dict[str, Any]:
    """Return UI-safe crisis-care status for a conversation."""
    try:
        redis = await get_redis()
        key = _crisis_care_key(
            conversation_id,
            user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )
        raw = await redis.get(key)
        if not raw:
            return {
                "active": False,
                "unavailable": False,
                "source": None,
                "release_count": 0,
                "aftercare_turn_count": 0,
                "turns_since_safety_check": 0,
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
            "release_count": _coerce_nonnegative_int(data.get("release_count")),
            "aftercare_turn_count": _coerce_nonnegative_int(
                data.get("aftercare_turn_count"),
            ),
            "turns_since_safety_check": _coerce_nonnegative_int(
                data.get("turns_since_safety_check"),
            ),
            "workspace_id": str(data.get("workspace_id") or workspace_id or ""),
            "agent_id": str(data.get("agent_id") or agent_id or ""),
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
            "aftercare_turn_count": 0,
            "turns_since_safety_check": 0,
            "ttl_seconds": None,
            "context_preview": None,
        }


async def mark_crisis_care_active(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
    context: str,
    source: str,
    release_count: int = 0,
    aftercare_turn_count: int = 0,
    turns_since_safety_check: int = 0,
) -> None:
    """Persist active crisis-care state with a bounded TTL."""
    try:
        redis = await get_redis()
        payload: dict[str, Any] = {
            "context": context[-1200:],
            "source": source,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "release_count": max(0, int(release_count)),
            "aftercare_turn_count": max(0, int(aftercare_turn_count)),
            "turns_since_safety_check": max(0, int(turns_since_safety_check)),
        }
        await redis.set(
            _crisis_care_key(
                conversation_id,
                user_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
            ),
            json.dumps(payload, ensure_ascii=False),
            ex=_CRISIS_CARE_TTL_SECONDS,
        )
    except Exception as e:
        logger.warning(f"mark crisis care state failed: {e}")


async def clear_crisis_care_state(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
) -> None:
    """Clear crisis-care state when the user explicitly releases it."""
    try:
        redis = await get_redis()
        await redis.delete(_crisis_care_key(
            conversation_id,
            user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
        ))
    except Exception as e:
        logger.warning(f"clear crisis care state failed: {e}")
