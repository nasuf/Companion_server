"""Public event router for achievement evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import overload

from app.services.achievements.events import (
    AssistantMessageAchievementEvent,
    IntentAchievementEvent,
    MemoryChangelogAchievementEvent,
    UserMessageAchievementEvent,
)
from app.services.achievements.rules.assistant_message_rules import (
    evaluate_assistant_message,
)
from app.services.achievements.rules.intent_rules import evaluate_intent
from app.services.achievements.rules.memory_rules import evaluate_memory_changelog
from app.services.achievements.rules.user_message_rules import evaluate_user_message

AchievementEvent = (
    UserMessageAchievementEvent
    | AssistantMessageAchievementEvent
    | MemoryChangelogAchievementEvent
    | IntentAchievementEvent
)


@overload
async def handle_achievement_event(event: UserMessageAchievementEvent) -> None: ...


@overload
async def handle_achievement_event(event: AssistantMessageAchievementEvent) -> None: ...


@overload
async def handle_achievement_event(event: MemoryChangelogAchievementEvent) -> None: ...


@overload
async def handle_achievement_event(event: IntentAchievementEvent) -> None: ...


async def handle_achievement_event(event: AchievementEvent) -> None:
    """Dispatch an application event to internal achievement rules."""
    if isinstance(event, UserMessageAchievementEvent):
        await evaluate_user_message(event)
        return
    if isinstance(event, AssistantMessageAchievementEvent):
        await evaluate_assistant_message(event)
        return
    if isinstance(event, MemoryChangelogAchievementEvent):
        await evaluate_memory_changelog(event)
        return
    if isinstance(event, IntentAchievementEvent):
        await evaluate_intent(event)
        return
    raise TypeError(f"Unsupported achievement event: {type(event)!r}")


async def handle_user_message_event(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    message_id: str,
    text: str,
    agent_name: str | None = None,
    reply_context: dict | None = None,
    aggregation_route: str | None = None,
    component_card: dict | None = None,
    occurred_at: datetime | None = None,
) -> None:
    await handle_achievement_event(
        UserMessageAchievementEvent(
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            message_id=message_id,
            text=text,
            agent_name=agent_name,
            reply_context=reply_context,
            aggregation_route=aggregation_route,
            component_card=component_card,
            occurred_at=occurred_at,
        )
    )


async def handle_assistant_message_event(
    *,
    conversation_id: str,
    message_id: str,
    text: str,
    metadata: dict | None = None,
    occurred_at: datetime | None = None,
) -> None:
    await handle_achievement_event(
        AssistantMessageAchievementEvent(
            conversation_id=conversation_id,
            message_id=message_id,
            text=text,
            metadata=metadata,
            occurred_at=occurred_at,
        )
    )


async def handle_memory_changelog_event(
    user_id: str,
    memory_id: str,
    operation: str,
    workspace_id: str | None = None,
) -> None:
    await handle_achievement_event(
        MemoryChangelogAchievementEvent(
            user_id=user_id,
            memory_id=memory_id,
            operation=operation,
            workspace_id=workspace_id,
        )
    )


async def handle_intent_event(
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
    await handle_achievement_event(
        IntentAchievementEvent(
            intent=intent,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            message_id=message_id,
            metadata=metadata,
            occurred_at=occurred_at,
        )
    )
