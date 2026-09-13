"""Hot-path readers for chat management runtime config (SystemConfig → env)."""

from __future__ import annotations

from app.services.runtime_config import resolve_config_sync


def reply_delay_enabled() -> bool:
    return bool(resolve_config_sync(agent_id=None).reply_delay_enabled)


def reply_delay_max_seconds() -> int:
    return int(resolve_config_sync(agent_id=None).reply_delay_max_seconds)


def user_message_aggregation_enabled() -> bool:
    return bool(resolve_config_sync(agent_id=None).user_message_aggregation_enabled)


def clamp_reply_delay_seconds(seconds: float) -> float:
    """Apply admin max cap; return 0 when delay is globally disabled."""
    if not reply_delay_enabled():
        return 0.0
    cap = reply_delay_max_seconds()
    return max(0.0, min(float(seconds), float(cap)))
