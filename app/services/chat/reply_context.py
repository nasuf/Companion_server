"""Reply timing context for PRD-aligned delayed explanations.

Captures the AI's status at message receipt time and computes a
structured delay decision that can later be injected into the reply prompt.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from app.redis_client import get_redis

LAST_REPLY_TTL = 86400 * 14


def _last_reply_key(agent_id: str, user_id: str) -> str:
    return f"last_reply:{agent_id}:{user_id}"


async def get_last_reply_timestamp(agent_id: str, user_id: str) -> datetime | None:
    """Load the last assistant reply timestamp for an agent-user pair."""
    redis = await get_redis()
    raw = await redis.get(_last_reply_key(agent_id, user_id))
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


async def save_last_reply_timestamp(
    agent_id: str,
    user_id: str,
    when: datetime | None = None,
) -> None:
    """Persist the last assistant reply timestamp."""
    if not agent_id or not user_id:
        return
    ts = when or datetime.now(timezone.utc)
    redis = await get_redis()
    await redis.set(_last_reply_key(agent_id, user_id), ts.isoformat(), ex=LAST_REPLY_TTL)


def _normalize_received_at(received_at: datetime | None = None) -> datetime:
    ts = received_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _is_high_emotion(
    user_emotion: dict[str, Any] | None,
    ai_emotion: dict[str, Any] | None = None,
) -> bool:
    """spec §6.2: 高情绪判定。

    - 用户侧：唤醒度>0.6 且愉悦度<0.4，或唤醒度>0.7
    - AI 侧：AI PAD 唤醒度 > 0.7
    任一条件满足即进入高情绪状态。
    """
    if user_emotion:
        arousal = float(user_emotion.get("arousal", 0.5))
        pleasure = float(user_emotion.get("pleasure", 0.0))
        if (arousal > 0.6 and pleasure < 0.4) or arousal > 0.7:
            return True
    if ai_emotion:
        if float(ai_emotion.get("arousal", 0.0)) > 0.7:
            return True
    return False


def _schedule_delay_for_status(status: str) -> float:
    """委托给 timing.py 的 calculate_status_delay，避免重复实现。"""
    from app.services.schedule_domain.timing import calculate_status_delay
    return calculate_status_delay(status)


def compute_delay_profile(
    *,
    last_reply_at: datetime | None,
    received_at: datetime,
    received_status: dict[str, Any] | None,
    user_emotion: dict[str, Any] | None = None,
    ai_emotion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute PRD-aligned delay mode and duration."""
    age_seconds = None
    if last_reply_at is not None:
        base = last_reply_at if last_reply_at.tzinfo else last_reply_at.replace(tzinfo=timezone.utc)
        age_seconds = max(0.0, (received_at - base).total_seconds())

    if age_seconds is not None and age_seconds < 1800:
        return {
            "interaction_mode": "conversation_mode",
            "delay_reason": "conversation_mode",
            "delay_seconds": random.uniform(1, 5),
        }

    if _is_high_emotion(user_emotion, ai_emotion):
        # spec §6.2: 高情绪 0-5s(90%), 5-10s(10%)
        delay = random.uniform(0, 5) if random.random() < 0.9 else random.uniform(5, 10)
        return {
            "interaction_mode": "high_emotion",
            "delay_reason": "high_emotion",
            "delay_seconds": delay,
        }

    status = str((received_status or {}).get("status", "idle"))
    return {
        "interaction_mode": "schedule_state",
        "delay_reason": f"schedule_{status}",
        "delay_seconds": _schedule_delay_for_status(status),
    }


async def build_reply_timing_context(
    *,
    agent_id: str,
    user_id: str,
    received_status: dict[str, Any] | None,
    user_emotion: dict[str, Any] | None = None,
    ai_emotion: dict[str, Any] | None = None,
    received_at: datetime | None = None,
) -> dict[str, Any]:
    """Build structured timing context at message receipt time."""
    received_ts = _normalize_received_at(received_at)
    last_reply_at = await get_last_reply_timestamp(agent_id, user_id)
    profile = compute_delay_profile(
        last_reply_at=last_reply_at,
        received_at=received_ts,
        received_status=received_status,
        user_emotion=user_emotion,
        ai_emotion=ai_emotion,
    )
    return {
        "received_at": received_ts.isoformat(),
        "received_status": received_status or {},
        "user_emotion": user_emotion or {},
        "ai_emotion": ai_emotion or {},
        **profile,
    }


def merge_reply_contexts(
    base_context: dict[str, Any] | None,
    latest_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge aggregated fragment timing context with the latest message context.

    Keep the first receipt status/reason but refresh the latest receipt time.
    """
    if not base_context:
        return latest_context
    if not latest_context:
        return base_context

    merged = dict(base_context)
    merged["latest_received_at"] = latest_context.get("received_at", base_context.get("received_at"))
    merged["latest_user_emotion"] = latest_context.get("user_emotion", {})
    return merged


def actual_delay_seconds(
    reply_context: dict[str, Any] | None,
    now: datetime | None = None,
) -> float | None:
    """Compute actual elapsed seconds since the first received message."""
    if not reply_context:
        return None
    raw = reply_context.get("received_at")
    if not raw:
        return None
    try:
        received_at = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    current = _normalize_received_at(now)
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)
    return max(0.0, (current - received_at.astimezone(timezone.utc)).total_seconds())
