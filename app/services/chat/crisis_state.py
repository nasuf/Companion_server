"""Conversation-local crisis care state.

This is deliberately separate from intent detection.  Once a user has entered
the crisis path, follow-up turns stay in a safety-aware mode for a bounded,
time-based aftercare window.  The window starts at the latest crisis signal and
is never extended by ordinary follow-up turns; a new crisis signal starts a new
episode.  An explicit release ends the episode but keeps a ``released`` marker
until the window would have ended, so history scans cannot re-open it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

# Aftercare lasts one day from the latest crisis signal. Clinical follow-up is
# scheduled in days, not chat turns; a new signal re-enters immediately.
CRISIS_CARE_WINDOW = timedelta(hours=24)
_CONTEXT_MAX_CHARS = 1200


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


def parse_state_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _normalize_state(data: dict, *, workspace_id: str | None, agent_id: str | None) -> dict[str, Any]:
    context = str(data.get("context") or "").strip()
    source = data.get("source")
    started_at = parse_state_time(data.get("started_at"))
    return {
        "status": "released" if data.get("status") == "released" else "active",
        "context": context or "(recent crisis care active)",
        "trigger": str(data.get("trigger") or "").strip(),
        "source": str(source) if source else None,
        "started_at": started_at,
        "expires_at": started_at + CRISIS_CARE_WINDOW if started_at else None,
        "aftercare_turn_count": _coerce_nonnegative_int(data.get("aftercare_turn_count")),
        "turns_since_safety_check": _coerce_nonnegative_int(
            data.get("turns_since_safety_check"),
        ),
        "safety_check_count": _coerce_nonnegative_int(data.get("safety_check_count")),
        "checks_suppressed": bool(data.get("checks_suppressed")),
        "workspace_id": str(data.get("workspace_id") or workspace_id or ""),
        "agent_id": str(data.get("agent_id") or agent_id or ""),
    }


async def load_crisis_care_state(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Return the crisis-care episode (active or released) if inside its window.

    Redis is best-effort: failures return None and the caller falls back to a
    time-bounded history scan.
    """
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
        state = _normalize_state(json.loads(raw), workspace_id=workspace_id, agent_id=agent_id)
    except Exception as e:
        logger.warning(f"load crisis care state failed: {e}")
        return None
    expires_at = state["expires_at"]
    if expires_at is not None and expires_at <= (now or datetime.now(UTC)):
        return None
    return state


async def get_crisis_care_status(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
) -> dict[str, Any]:
    """Return UI-safe crisis-care status for a conversation."""
    inactive = {
        "active": False,
        "unavailable": False,
        "status": None,
        "source": None,
        "aftercare_turn_count": 0,
        "turns_since_safety_check": 0,
        "safety_check_count": 0,
        "started_at": None,
        "expires_at": None,
        "ttl_seconds": None,
        "context_preview": None,
    }
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
            return inactive
        state = _normalize_state(json.loads(raw), workspace_id=workspace_id, agent_id=agent_id)
        ttl = await redis.ttl(key)
    except Exception as e:
        logger.warning(f"get crisis care status failed: {e}")
        return {**inactive, "unavailable": True}
    context = state["context"]
    return {
        "active": state["status"] == "active",
        "unavailable": False,
        "status": state["status"],
        "source": state["source"],
        "aftercare_turn_count": state["aftercare_turn_count"],
        "turns_since_safety_check": state["turns_since_safety_check"],
        "safety_check_count": state["safety_check_count"],
        "workspace_id": state["workspace_id"],
        "agent_id": state["agent_id"],
        "started_at": state["started_at"].isoformat() if state["started_at"] else None,
        "expires_at": state["expires_at"].isoformat() if state["expires_at"] else None,
        "ttl_seconds": ttl if isinstance(ttl, int) and ttl >= 0 else None,
        "context_preview": context[-160:] if context else None,
    }


async def _save_state(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
    payload: dict[str, Any],
    started_at: datetime,
    now: datetime | None,
) -> None:
    """Persist until the episode window ends; never extend it."""
    remaining = int((started_at + CRISIS_CARE_WINDOW - (now or datetime.now(UTC))).total_seconds())
    if remaining <= 0:
        return
    try:
        redis = await get_redis()
        await redis.set(
            _crisis_care_key(
                conversation_id,
                user_id,
                workspace_id=workspace_id,
                agent_id=agent_id,
            ),
            json.dumps(
                {
                    **payload,
                    "started_at": started_at.isoformat(),
                    "workspace_id": workspace_id,
                    "agent_id": agent_id,
                },
                ensure_ascii=False,
            ),
            ex=remaining,
        )
    except Exception as e:
        logger.warning(f"save crisis care state failed: {e}")


async def mark_crisis_care_active(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
    context: str,
    source: str,
    started_at: datetime,
    trigger: str = "",
    aftercare_turn_count: int = 0,
    turns_since_safety_check: int = 0,
    safety_check_count: int = 0,
    checks_suppressed: bool = False,
    now: datetime | None = None,
) -> None:
    """Persist an active crisis-care episode, expiring at the end of its window."""
    await _save_state(
        conversation_id,
        user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
        payload={
            "status": "active",
            "context": context[-_CONTEXT_MAX_CHARS:],
            "trigger": trigger[:_CONTEXT_MAX_CHARS],
            "source": source,
            "aftercare_turn_count": max(0, int(aftercare_turn_count)),
            "turns_since_safety_check": max(0, int(turns_since_safety_check)),
            "safety_check_count": max(0, int(safety_check_count)),
            "checks_suppressed": bool(checks_suppressed),
        },
        started_at=started_at,
        now=now,
    )


async def mark_crisis_care_released(
    conversation_id: str,
    user_id: str,
    *,
    workspace_id: str | None,
    agent_id: str | None,
    started_at: datetime,
    now: datetime | None = None,
) -> None:
    """End the episode; the marker stops history scans from re-opening it."""
    await _save_state(
        conversation_id,
        user_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
        payload={
            "status": "released",
            "source": "released",
            "released_at": (now or datetime.now(UTC)).isoformat(),
        },
        started_at=started_at,
        now=now,
    )
