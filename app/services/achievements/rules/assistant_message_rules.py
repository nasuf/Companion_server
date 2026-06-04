"""Achievement rules driven by assistant and proactive messages."""

from __future__ import annotations

from datetime import datetime

from app.db import db
from app.services.achievements.events import AssistantMessageAchievementEvent, AssistantTurnAchievementEvent
from app.services.achievements.repository import _birthday_mmdd, _event_count, record_event, unlock_achievement
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


async def evaluate_assistant_turn(event: AssistantTurnAchievementEvent) -> None:
    await _evaluate_assistant_turn(
        conversation_id=event.conversation_id,
        message_id=event.message_id,
        assistant_texts=event.assistant_texts,
        user_message_ids=event.user_message_ids,
        turn_id=event.turn_id,
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
        await _check_slow_assistant_reply(user_id, agent_id, workspace_id, conversation_id, occurred_at)


async def _evaluate_assistant_turn(
    *,
    conversation_id: str,
    message_id: str,
    assistant_texts: list[str],
    user_message_ids: list[str],
    turn_id: str | None,
    metadata: dict | None,
    occurred_at: datetime | None = None,
) -> None:
    if metadata and metadata.get("proactive"):
        return
    conv = await db.conversation.find_unique(where={"id": conversation_id})
    if not conv:
        return
    user_id = conv.userId
    agent_id = conv.agentId
    workspace_id = getattr(conv, "workspaceId", None)
    occurred_at = _aware(occurred_at or _now())
    await _check_turn_pair_100(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        at=occurred_at,
        assistant_texts=assistant_texts,
        user_message_ids=user_message_ids,
        turn_id=turn_id,
    )


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


async def _check_turn_pair_100(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    at: datetime,
    assistant_texts: list[str],
    user_message_ids: list[str],
    turn_id: str | None = None,
) -> None:
    assistant_rows = await _turn_assistant_message_rows(user_id, agent_id, turn_id)
    if assistant_rows:
        assistant_chars = sum(count_chars(str(_field(row, "content") or "")) for row in assistant_rows)
    else:
        assistant_chars = sum(count_chars(text) for text in assistant_texts)
    user_rows = await _turn_user_message_rows(user_id, agent_id, user_message_ids, at)
    user_chars = sum(count_chars(str(_field(row, "content") or "")) for row in user_rows)
    if user_rows and user_chars + assistant_chars == 100:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=81)


async def _turn_assistant_message_rows(
    user_id: str,
    agent_id: str,
    turn_id: str | None,
) -> list:
    if not turn_id:
        return []
    return await db.query_raw(
        """
        SELECT m.content
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'assistant'
          AND m.metadata->>'achievement_turn_id' = $3
        ORDER BY m.created_at ASC
        """,
        user_id,
        agent_id,
        turn_id,
    )


async def _turn_user_message_rows(
    user_id: str,
    agent_id: str,
    user_message_ids: list[str],
    at: datetime,
) -> list:
    ids = [message_id for message_id in dict.fromkeys(user_message_ids) if message_id]
    if ids:
        return await db.query_raw(
            """
            SELECT m.content
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.user_id = $1
              AND c.agent_id = $2
              AND c.is_deleted = FALSE
              AND m.role = 'user'
              AND m.id = ANY($3::text[])
            ORDER BY m.created_at ASC
            """,
            user_id,
            agent_id,
            ids,
        )
    return await db.query_raw(
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
