"""Shared persona and conversation inputs for activity-visible messages."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from app.db import db
from app.services.mbti import build_personality_brief
from app.services.proactive.dialogue import RECENT_MESSAGE_LIMIT, format_recent_turns

logger = logging.getLogger(__name__)


def persona_fields(ctx: dict[str, Any]) -> dict[str, str]:
    agent = SimpleNamespace(mbti=ctx.get("agent_mbti"), currentMbti=None)
    return {
        "agent_name": str(ctx.get("agent_name") or "（未提供）"),
        "personality_brief": build_personality_brief(agent),
    }


async def message_context(ctx: dict[str, Any]) -> dict[str, str]:
    fields = persona_fields(ctx)
    fields["dialogue_context"] = "（无）"
    if not ctx.get("conversation_id") or not ctx.get("user_id"):
        return fields
    try:
        rows = await db.query_raw(
            """
            SELECT m.role, m.content, m.created_at
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE c.id = $1 AND c.user_id = $2 AND c.workspace_id = $3
              AND c.is_deleted = FALSE AND m.role IN ('user', 'assistant')
            ORDER BY m.created_at DESC, m.id DESC LIMIT $4
            """,
            ctx["conversation_id"], ctx["user_id"], ctx.get("workspace_id"),
            RECENT_MESSAGE_LIMIT,
        )
        fields["dialogue_context"] = format_recent_turns(list(reversed(rows or []))) or "（无）"
    except Exception as exc:
        logger.warning("[offline] recent dialogue unavailable: %s", type(exc).__name__)
    return fields


async def recommendation_dialogue(
    *, user_id: str, workspace_id: str | None, conversation_id: str | None,
) -> list[str]:
    """Whole user utterances only: assistant statements are not user evidence."""
    if not conversation_id:
        return []
    rows = await db.query_raw(
        """
        SELECT m.content FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.id = $1 AND c.user_id = $2
          AND c.workspace_id IS NOT DISTINCT FROM $3::text
          AND c.is_deleted = FALSE AND m.role = 'user'
          AND COALESCE(m.metadata->>'trigger_type', '') NOT LIKE 'offline_%'
          AND NOT (COALESCE(m.metadata, '{}'::jsonb) ? 'component_card')
          AND length(m.content) BETWEEN 1 AND 1000
        ORDER BY m.created_at DESC, m.id DESC LIMIT 10
        """,
        conversation_id, user_id, workspace_id,
    )
    return [str(row['content']) for row in reversed(rows or [])]
