"""Achievement rules driven by completed fragment aggregation windows."""

from __future__ import annotations

from app.services.achievements.events import AggregationAchievementEvent
from app.services.achievements.repository import _event_count, record_event, unlock_achievement


async def evaluate_aggregation(event: AggregationAchievementEvent) -> None:
    if event.part_count < 2 or not event.source_id:
        return
    await record_event(
        user_id=event.user_id,
        agent_id=event.agent_id,
        workspace_id=event.workspace_id,
        conversation_id=event.conversation_id,
        event_type="aggregation_window_completed",
        source_id=event.source_id,
        value_int=event.part_count,
        occurred_at=event.occurred_at,
    )
    if await _event_count(
        event.user_id,
        event.agent_id,
        "aggregation_window_completed",
    ) >= 50:
        await unlock_achievement(
            user_id=event.user_id,
            agent_id=event.agent_id,
            workspace_id=event.workspace_id,
            conversation_id=event.conversation_id,
            achievement_id=86,
        )
