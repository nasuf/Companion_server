"""Achievement rules driven by assistant and proactive messages."""

from __future__ import annotations

from datetime import datetime

from app.db import db
from app.services.achievements.events import AssistantMessageAchievementEvent
from app.services.achievements.repository import _birthday_mmdd, _day_role_char_counts, _event_count, record_event, unlock_achievement
from app.services.achievements.utils import _aware, _field, _has_emoji, _local, _now, count_chars

_ASSISTANT_EMOJI_EVENT = "assistant_emoji"


async def evaluate_assistant_message(event: AssistantMessageAchievementEvent) -> None:
    await _evaluate_assistant_message(
        conversation_id=event.conversation_id,
        message_id=event.message_id,
        text=event.text,
        metadata=event.metadata,
        occurred_at=event.occurred_at,
    )


async def _evaluate_assistant_message(
    *,
    conversation_id: str,
    message_id: str,
    text: str,
    metadata: dict | None,
    occurred_at: datetime | None = None,
) -> None:
    conv = await db.conversation.find_unique(where={"id": conversation_id})
    if not conv:
        return
    user_id = conv.userId
    agent_id = conv.agentId
    workspace_id = getattr(conv, "workspaceId", None)
    occurred_at = _aware(occurred_at or _now())
    char_count = count_chars(text)
    await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="assistant_message", source_id=message_id, value_int=char_count, metadata=metadata, occurred_at=occurred_at)
    user_chars, ai_chars = await _day_role_char_counts(user_id, agent_id, occurred_at)
    if user_chars + ai_chars >= 10000:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=60)
    if metadata and metadata.get("delay_explanation"):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=24)
    if _has_emoji(text):
        await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type=_ASSISTANT_EMOJI_EVENT, source_id=message_id)
        if await _event_count(user_id, agent_id, _ASSISTANT_EMOJI_EVENT) >= 100:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=59)
    if char_count <= 3:
        await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="assistant_short_reply", source_id=message_id)
        if await _event_count(user_id, agent_id, "assistant_short_reply") >= 500:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=79)
    local = _local(occurred_at)
    if metadata and metadata.get("proactive"):
        if local.hour == 13 and local.minute == 14:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=84)
        user_birthday = await _birthday_mmdd(user_id, workspace_id, source="user")
        if user_birthday:
            month, day = user_birthday
            if local.hour == month and local.minute == day:
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=96)
        trigger_type = str(metadata.get("trigger_type") or "")
        if trigger_type == "memory_proactive" or trigger_type.startswith("memory_"):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=55)
    if not (metadata and metadata.get("proactive")):
        await _check_pair_100(user_id, agent_id, workspace_id, conversation_id, occurred_at, text)
        await _check_slow_assistant_reply(user_id, agent_id, workspace_id, conversation_id, occurred_at)


async def _check_slow_assistant_reply(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    at: datetime,
) -> None:
    rows = await db.query_raw(
        """
        SELECT m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'user'
          AND m.created_at <= $3::timestamp
        ORDER BY m.created_at DESC
        LIMIT 1
        """,
        user_id,
        agent_id,
        at,
    )
    if rows and _field(rows[0], "created_at"):
        user_at = _aware(_field(rows[0], "created_at"))
        if (at - user_at).total_seconds() >= 1800:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=30)


async def _check_pair_100(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    at: datetime,
    assistant_text: str,
) -> None:
    rows = await db.query_raw(
        """
        SELECT m.content
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'user'
          AND m.created_at <= $3::timestamp
        ORDER BY m.created_at DESC
        LIMIT 1
        """,
        user_id,
        agent_id,
        at,
    )
    if rows and count_chars(str(_field(rows[0], "content") or "")) + count_chars(assistant_text) == 100:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=81)
