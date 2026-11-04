"""Load persona, core memories, dynamic context and recent dialogue for one call."""

from __future__ import annotations

import asyncio
import logging

from app.db import db
from app.services.memory.core_memory import load_core_memory_strings
from app.services.realtime_voice.instructions import (
    DynamicContext,
    PersonaFacts,
    build_instructions,
    history_turns,
    persona_from_agent,
)

logger = logging.getLogger(__name__)


async def load_call_context(
    *,
    agent,
    user_id: str,
    workspace_id: str | None,
    conversation_id: str,
) -> tuple[PersonaFacts, str, list[dict[str, str]]]:
    """Load everything needed to configure a voice session.

    Returns (persona, instructions_text, history_turns_list).
    """
    (
        user_memories, ai_memories, recent,
        dynamic,
    ) = await _gather_all(
        agent=agent,
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )
    persona = persona_from_agent(agent)
    history = history_turns(recent)
    instructions = build_instructions(
        persona,
        user_memories,
        ai_memories,
        has_history=bool(history),
        dynamic=dynamic,
    )
    return persona, instructions, history


async def load_dynamic_context(
    *,
    agent_id: str | None,
    user_id: str,
    conversation_id: str,
) -> DynamicContext:
    """Load dynamic context (time, intimacy, mood, patience, relation meta).

    Each source is best-effort: failure returns a default rather than
    crashing the call.
    """
    time_ctx = ""
    intimacy_stage = ""
    ai_mood_text = ""
    patience_zone = "normal"
    relation_meta_line = ""

    try:
        from app.services.time_domain.clock import build_time_context
        time_ctx = build_time_context()
    except Exception:
        logger.debug("[realtime-voice] time context unavailable")

    try:
        from app.services.relationship.intimacy import get_relationship_stage, get_topic_intimacy
        if agent_id and user_id:
            score = await get_topic_intimacy(agent_id, user_id)
            intimacy_stage = get_relationship_stage(score)
    except Exception:
        logger.debug("[realtime-voice] intimacy unavailable")

    try:
        from app.services.relationship.ai_mood import format_ai_mood_text, load_ai_mood
        mood = await load_ai_mood(conversation_id)
        ai_mood_text = format_ai_mood_text(mood)
    except Exception:
        logger.debug("[realtime-voice] ai mood unavailable")

    try:
        from app.services.interaction.boundary import get_patience, get_patience_zone
        if agent_id:
            patience = await get_patience(agent_id, user_id)
            patience_zone = get_patience_zone(patience)
    except Exception:
        logger.debug("[realtime-voice] patience unavailable")

    try:
        from app.services.relationship.relation_meta import (
            format_relation_meta_line,
            get_relation_meta,
        )
        meta = await get_relation_meta(conversation_id)
        relation_meta_line = format_relation_meta_line(meta)
    except Exception:
        logger.debug("[realtime-voice] relation meta unavailable")

    return DynamicContext(
        time_context=time_ctx,
        intimacy_stage=intimacy_stage,
        ai_mood_text=ai_mood_text,
        patience_zone=patience_zone,
        relation_meta_line=relation_meta_line,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _gather_all(
    *,
    agent,
    user_id: str,
    workspace_id: str | None,
    conversation_id: str,
) -> tuple[list, list, list, DynamicContext]:
    """Parallel load of all call setup data."""
    agent_id = getattr(agent, "id", None)
    results = await asyncio.gather(
        load_core_memory_strings(user_id=user_id, workspace_id=workspace_id, source="user"),
        load_core_memory_strings(user_id=user_id, workspace_id=workspace_id, source="ai"),
        _recent_messages(conversation_id),
        load_dynamic_context(
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
        ),
        return_exceptions=True,
    )
    user_memories = _or_empty(results[0], "user core memories")
    ai_memories = _or_empty(results[1], "ai core memories")
    recent = _or_empty(results[2], "recent messages")
    dynamic = results[3] if isinstance(results[3], DynamicContext) else DynamicContext()
    if isinstance(results[3], BaseException):
        logger.warning("[realtime-voice] dynamic context failed: %s", results[3])
    return user_memories, ai_memories, recent, dynamic


def _or_empty(value, label: str):
    if isinstance(value, BaseException):
        logger.warning("[realtime-voice] %s unavailable: %s", label, value)
        return []
    return value or []


async def _recent_messages(conversation_id: str) -> list[dict]:
    rows = await db.message.find_many(
        where={"conversationId": conversation_id},
        order={"createdAt": "desc"},
        take=12,
    )
    ordered = list(reversed(rows))
    return [
        {"role": row.role, "content": row.content or ""}
        for row in ordered
        if row.role in {"user", "assistant"} and (row.content or "").strip()
    ]
