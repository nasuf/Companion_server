"""Repair entry point for retrieval-driven memory mistakes.

When the assistant used a retrieved memory and the next user turn sounds like a
correction ("you remembered that wrong", "I never said that"), this module
routes the case into the existing contradiction flow. It deliberately asks for
confirmation first instead of editing memory immediately.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from prisma import Json

from app.db import db
from app.observability.events import EVT_PREFLIGHT_RESOLVED
from app.services.memory.interaction.contradiction import (
    generate_contradiction_inquiry,
    save_pending_contradiction,
)
from app.services.memory.retrieval.trace import build_memory_retrieval_feedback
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.repo import MemoryRecord

logger = logging.getLogger(__name__)

_MIN_FEEDBACK_CONFIDENCE = 0.82


async def build_retrieval_feedback_conflict(
    *,
    user_message: str,
    previous_assistant: Any,
    user_id: str,
    workspace_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return a contradiction-compatible conflict from the previous reply trace."""
    metadata = getattr(previous_assistant, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    feedback = build_memory_retrieval_feedback(
        user_message=user_message,
        previous_assistant_reply=getattr(previous_assistant, "content", "") or "",
        previous_metadata=metadata,
    )
    if not feedback:
        return None
    if float(feedback.get("confidence") or 0.0) < _MIN_FEEDBACK_CONFIDENCE:
        return None

    memory = await _first_repairable_memory(
        feedback.get("memory_ids") or [],
        user_id=user_id,
        workspace_id=workspace_id,
    )
    if memory is None:
        return None

    old_text = memory.summary or memory.content
    conflict = {
        "has_conflict": True,
        "conflicting_memory_id": memory.id,
        "old_content": old_text,
        "new_info": user_message,
        "conflict_description": (
            "用户在上一轮回复后疑似纠正被使用的记忆。"
            f"用户现在说：{user_message}"
        ),
        "source": "retrieval_feedback",
        "feedback_signal": feedback.get("signal"),
        "feedback_confidence": feedback.get("confidence"),
        "assistant_message_id": getattr(previous_assistant, "id", None),
    }
    feedback["repair_action"] = {
        "type": "confirmation_requested",
        "memory_id": memory.id,
        "memory_source": memory.source,
    }
    return conflict, feedback


async def resolve_retrieval_feedback_correction(
    *,
    user_message: str,
    previous_assistant: Any | None,
    ctx: Any,
    workspace_id: str | None,
) -> AsyncGenerator[dict, None]:
    """Ask for confirmation when the user appears to correct a retrieved memory."""
    if previous_assistant is None:
        return

    result = await build_retrieval_feedback_conflict(
        user_message=user_message,
        previous_assistant=previous_assistant,
        user_id=ctx.user_id,
        workspace_id=workspace_id,
    )
    if not result:
        return
    conflict, feedback = result

    await _patch_previous_feedback(previous_assistant, feedback)
    await save_pending_contradiction(ctx.conversation_id, conflict)
    personality_brief = ctx.agent.name if ctx.agent else "AI"
    reply = await generate_contradiction_inquiry(
        conflict,
        agent_name=personality_brief,
    )
    ctx.last_short_circuit_reply = reply
    logger.info(
        "[PREFLIGHT] retrieval feedback routed to contradiction confirmation",
        extra={
            "event": EVT_PREFLIGHT_RESOLVED,
            "kind": "retrieval_feedback",
            "memory_id": conflict.get("conflicting_memory_id"),
            "confidence": feedback.get("confidence"),
        },
    )
    for evt in await ctx.short_circuit_fn(
        reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
        trace_id=ctx.tracer.safe_trace_id,
    ):
        yield evt
    ctx.tracer.close()
    ctx.stopped = True


async def _first_repairable_memory(
    memory_ids: list[Any],
    *,
    user_id: str,
    workspace_id: str | None,
) -> MemoryRecord | None:
    for raw_id in memory_ids:
        memory_id = str(raw_id or "")
        if not memory_id:
            continue
        memory = await memory_repo.find_unique(memory_id)
        if not memory or memory.isArchived:
            continue
        if memory.userId != user_id:
            continue
        if workspace_id is not None and memory.workspaceId != workspace_id:
            continue
        return memory
    return None


async def _patch_previous_feedback(previous_assistant: Any, feedback: dict[str, Any]) -> None:
    message_id = getattr(previous_assistant, "id", None)
    if not message_id:
        return
    latest = await db.message.find_unique(where={"id": message_id})
    if not latest:
        return
    metadata = latest.metadata if isinstance(latest.metadata, dict) else {}
    existing = metadata.get("memory_retrieval_feedback")
    if isinstance(existing, dict):
        merged_feedback = {**existing, **feedback}
    else:
        merged_feedback = feedback
    await db.message.update(
        where={"id": message_id},
        data={"metadata": Json({**metadata, "memory_retrieval_feedback": merged_feedback})},
    )
