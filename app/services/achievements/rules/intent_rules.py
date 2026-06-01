"""Achievement rules driven by resolved chat intents."""

from __future__ import annotations

from datetime import datetime

from app.services.achievements.events import IntentAchievementEvent
from app.services.achievements.repository import _event_count, record_event, unlock_achievement


async def evaluate_intent(event: IntentAchievementEvent) -> None:
    await _evaluate_intent(
        intent=event.intent,
        user_id=event.user_id,
        agent_id=event.agent_id,
        workspace_id=event.workspace_id,
        conversation_id=event.conversation_id,
        message_id=event.message_id,
        metadata=event.metadata,
        occurred_at=event.occurred_at,
    )


async def _evaluate_intent(
    *,
    intent: str,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    message_id: str | None,
    metadata: dict | None = None,
    occurred_at: datetime | None = None,
) -> None:
    """Route resolved chat intents to achievement-specific counters."""
    if intent == "schedule_adjust":
        await _process_schedule_adjust_intent(
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            message_id=message_id,
            metadata=metadata,
            occurred_at=occurred_at,
        )


async def _process_schedule_adjust_intent(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    message_id: str | None,
    metadata: dict | None,
    occurred_at: datetime | None,
) -> None:
    if not message_id:
        return
    await record_event(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        event_type="schedule_adjust_request",
        source_id=message_id,
        metadata=metadata,
        occurred_at=occurred_at,
    )
    await unlock_achievement(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        achievement_id=20,
    )
    if await _event_count(user_id, agent_id, "schedule_adjust_request") >= 50:
        await unlock_achievement(
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            achievement_id=87,
        )
