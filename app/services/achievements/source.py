"""Persisted message boundary for chat-derived achievement evaluations.

The event callback is a reference, not evidence of message content, ownership or
receipt time. A task-local source also binds derived counters to the same input.
Memory/daily events retain their own source contracts.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime

from app.db import db
from app.services.achievements.events import (
    AggregationAchievementEvent,
    AssistantMessageAchievementEvent,
    AssistantTurnAchievementEvent,
    IntentAchievementEvent,
    UserMessageAchievementEvent,
)
from app.services.achievements.utils import _aware

CONTRACT = "persisted_message_v1"
MAX_TURN_MESSAGES = 100


class AchievementSourceUnavailable(ValueError):
    """The input no longer belongs to this live conversation and role."""


@dataclass(frozen=True)
class MessageSource:
    message_id: str
    conversation_id: str
    user_id: str
    agent_id: str
    workspace_id: str | None
    role: str
    occurred_at: datetime


current_source: ContextVar[MessageSource | None] = ContextVar(
    "achievement_message_source", default=None
)


async def resolve_message_event(event):
    """Resolve only chat events, fail closed on missing/cross-scope inputs.

    DB errors intentionally propagate; a failed read is not a missing source.
    """
    if not isinstance(
        event,
        (
            UserMessageAchievementEvent,
            AssistantMessageAchievementEvent,
            AssistantTurnAchievementEvent,
            IntentAchievementEvent,
            AggregationAchievementEvent,
        ),
    ):
        return event, None
    message_id = (
        event.source_id
        if isinstance(event, AggregationAchievementEvent)
        else event.message_id
    )
    role = (
        "assistant"
        if isinstance(
            event, (AssistantMessageAchievementEvent, AssistantTurnAchievementEvent)
        )
        else "user"
    )
    if not message_id or not event.conversation_id:
        raise AchievementSourceUnavailable("missing_message_id")
    rows = await db.query_raw(
        """
        SELECT m.id, m.content, m.metadata, m.created_at,
               c.user_id, c.agent_id, c.workspace_id
        FROM messages m JOIN conversations c ON c.id=m.conversation_id
        JOIN ai_agents a ON a.id=c.agent_id AND a.user_id=c.user_id
        LEFT JOIN chat_workspaces w ON w.id=c.workspace_id
        WHERE m.id=$1 AND m.conversation_id=$2 AND m.role=$3
          AND c.is_deleted=FALSE
          AND COALESCE(m.metadata->'human_takeover', 'false'::jsonb) <> 'true'::jsonb
          AND (c.workspace_id IS NULL OR (w.user_id=c.user_id AND w.agent_id=c.agent_id AND w.status='active'))
        LIMIT 1
        """,
        message_id,
        event.conversation_id,
        role,
    )
    if not rows:
        raise AchievementSourceUnavailable("missing_or_inactive_message")
    row = rows[0]
    if hasattr(event, "user_id") and (
        event.user_id != row["user_id"]
        or event.agent_id != row["agent_id"]
        or (
            event.workspace_id is not None and event.workspace_id != row["workspace_id"]
        )
    ):
        raise AchievementSourceUnavailable("scope_mismatch")
    source = MessageSource(
        message_id,
        event.conversation_id,
        row["user_id"],
        row["agent_id"],
        row["workspace_id"],
        role,
        _aware(row["created_at"]),
    )
    values = {"occurred_at": source.occurred_at}
    if hasattr(event, "workspace_id"):
        values["workspace_id"] = source.workspace_id
    metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
    if isinstance(event, UserMessageAchievementEvent):
        values.update(
            text=row["content"], component_card=metadata.get("component_card")
        )
    elif isinstance(event, AssistantMessageAchievementEvent):
        values.update(text=row["content"], metadata=metadata)
    elif isinstance(event, AssistantTurnAchievementEvent):
        # Never combine rows from another conversation or a guessed latest user
        # message. Missing links cannot prove an exact 100-character turn.
        ids = list(dict.fromkeys(event.user_message_ids))
        if not ids or len(ids) > MAX_TURN_MESSAGES or not event.turn_id:
            raise AchievementSourceUnavailable("incomplete_turn_source")
        if metadata.get("achievement_turn_id") != event.turn_id:
            raise AchievementSourceUnavailable("turn_id_mismatch")
        user_rows = await db.query_raw(
            """SELECT id FROM messages WHERE conversation_id=$1 AND role='user'
               AND id=ANY($2::text[]) AND created_at <= $3::timestamp""",
            event.conversation_id,
            ids,
            source.occurred_at,
        )
        if {r["id"] for r in user_rows} != set(ids):
            raise AchievementSourceUnavailable("turn_user_scope_mismatch")
        values.update(user_message_ids=ids, metadata=metadata)
    return replace(event, **values), source


def assert_matching_scope(source, user_id, agent_id, workspace_id, conversation_id):
    if (user_id, agent_id, workspace_id, conversation_id) != (
        source.user_id,
        source.agent_id,
        source.workspace_id,
        source.conversation_id,
    ):
        raise AchievementSourceUnavailable("write_scope_mismatch")


# Evaluated in the INSERT statement, so a deleted source cannot create a new
# derived counter/unlock after an earlier asynchronous lookup succeeded.
LIVE_SOURCE_SQL = """
    EXISTS (SELECT 1 FROM messages sm
        JOIN conversations sc ON sc.id=sm.conversation_id
        LEFT JOIN chat_workspaces sw ON sw.id=sc.workspace_id
        WHERE sm.id={message} AND sm.role={role}
          AND sc.id=$4 AND sc.user_id=$1 AND sc.agent_id=$2
          AND sc.workspace_id IS NOT DISTINCT FROM $3::text
          AND sc.is_deleted=FALSE
          AND COALESCE(sm.metadata->'human_takeover', 'false'::jsonb) <> 'true'::jsonb
          AND (sc.workspace_id IS NULL OR (sw.user_id=sc.user_id AND sw.agent_id=sc.agent_id AND sw.status='active')))
"""


async def recent_message_sources(user_id: str, agent_id: str, limit: int = 20) -> dict:
    """Bounded admin projection; no content, callback metadata or private URLs.

    Four index-backed type slices cap candidate materialization at 4 * limit.
    A link identifies a source; it does not prove an unlock or notification.
    """
    from datetime import timezone

    if not 1 <= limit <= 50:
        raise ValueError("invalid_source_sample_limit")
    rows = await db.query_raw(
        """
        WITH sample AS (
            SELECT e.* FROM (VALUES ('user_message'), ('assistant_message'),
                ('schedule_adjust_request'), ('aggregation_window_completed')) kinds(kind)
            CROSS JOIN LATERAL (
                SELECT id,event_type,source_id,conversation_id,occurred_at,metadata
                FROM achievement_events
                WHERE user_id=$1 AND agent_id=$2 AND event_type=kinds.kind
                ORDER BY occurred_at DESC LIMIT $3
            ) e
            ORDER BY occurred_at DESC,id DESC LIMIT $3
        )
        SELECT e.id,e.event_type,e.occurred_at,
            CASE WHEN e.metadata->>'source_contract' IS DISTINCT FROM 'persisted_message_v1'
                THEN 'legacy_unverified'
                WHEN m.id IS NULL OR c.id IS NULL THEN 'source_missing'
                ELSE 'persisted_message' END AS source_state
        FROM sample e
        LEFT JOIN conversations c ON c.id=e.conversation_id AND c.user_id=$1
            AND c.agent_id=$2 AND c.is_deleted=FALSE
        LEFT JOIN messages m ON m.id=e.metadata->>'source_message_id'
            AND m.conversation_id=c.id AND m.role=e.metadata->>'source_role'
        ORDER BY e.occurred_at DESC,e.id DESC
        """,
        user_id,
        agent_id,
        limit,
    )
    return {
        "state": "available",
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "count": len(rows),
        "denominator": "recent_message_events_only",
        "items": [
            {**r, "occurred_at": _aware(r["occurred_at"]).isoformat()} for r in rows
        ],
    }
