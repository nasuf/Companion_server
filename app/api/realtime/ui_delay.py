"""WS UI timing helpers for delayed reply read/typing indicators."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.interaction.user_turn_aggregation import UserMessageAggregationPlan
from app.services.schedule_domain.time_service import _now_corrected


def _parse_received_at(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        received_at = datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None
    if received_at.tzinfo is None:
        return received_at.replace(tzinfo=timezone.utc)
    return received_at.astimezone(timezone.utc)


def remaining_ui_delay_seconds(reply_context: dict | None) -> float:
    """Return seconds until read/typing UI should appear for this reply context."""
    ctx = reply_context or {}
    delay = float(ctx.get("delay_seconds", 0) or 0)
    if delay <= 0:
        return 0.0

    received_at = _parse_received_at(ctx.get("received_at"))
    if received_at is None:
        return delay

    now = _now_corrected()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    remaining = delay - (now - received_at).total_seconds()
    return max(0.0, remaining)


def compute_ack_ui_timing(
    *,
    plan: UserMessageAggregationPlan | None = None,
    reply_context: dict | None = None,
) -> dict[str, Any]:
    """Build ack payload fields controlling client read/typing timing."""
    from app.services.interaction.chat_management import reply_delay_enabled

    if plan and plan.should_wait:
        return {"ui_delay_seconds": 0.0, "defer_ui": True}

    if not reply_delay_enabled():
        return {"ui_delay_seconds": 0.0, "defer_ui": False}

    ctx = (plan.final_context if plan else None) or reply_context or {}
    remaining = remaining_ui_delay_seconds(ctx)
    if remaining <= 0:
        return {"ui_delay_seconds": 0.0, "defer_ui": False}
    return {"ui_delay_seconds": remaining, "defer_ui": False}


async def send_processing_event(
    send_fn,
    *,
    conversation_id: str,
    message_id: str | None = None,
    client_id: str | None = None,
    reply_context: dict | None = None,
    ui_delay_seconds: float | None = None,
) -> None:
    """Notify the client that reply generation is queued or starting."""
    delay = (
        float(ui_delay_seconds)
        if ui_delay_seconds is not None
        else remaining_ui_delay_seconds(reply_context)
    )
    payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "ui_delay_seconds": max(0.0, delay),
    }
    if message_id:
        payload["message_id"] = message_id
    if client_id:
        payload["client_id"] = client_id
    await send_fn({"type": "processing", "data": payload})
