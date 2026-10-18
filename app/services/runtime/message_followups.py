"""Message statistics outbox; Redis is transport, SQL owns durable completion.

Only idempotent persisted-message rules run here. This is not a retry wrapper
for memory extraction, personality, patience, complete turns or notifications.
"""
from __future__ import annotations

import logging

from app.db import db
from app.services.runtime.job_queue import (
    enqueue_runtime_job, inspect_runtime_job,
)

from app.services.runtime.followup_queue import (
    MESSAGE, FollowupRetryConflict, dispatch_followups, inspect_followups,
    prune_followups, retry_followup,
)

logger = logging.getLogger(__name__)
JOB_TYPE = "message_statistics"
_TERMINAL = {"completed", "cancelled"}


class MessageFollowupFailed(RuntimeError):
    """Public failures contain a class only, never SQL, messages or credentials."""


async def dispatch_message_followups(*, limit: int = 10, message_id: str | None = None) -> int:
    return await dispatch_followups(db, MESSAGE, limit=limit, identity=message_id,
        enqueue=enqueue_runtime_job, inspect=inspect_runtime_job)


async def _load_event(message_id: str):
    rows = await db.query_raw("""SELECT e.*,m.role,m.content,m.metadata,m.created_at AS happened,
        message_followup_version(m.content,m.role,m.metadata,m.created_at) AS current_version,
        jsonb_build_object('user_id',c.user_id,'agent_id',c.agent_id,
            'workspace_id',c.workspace_id,'conversation_id',c.id,'role',m.role) AS current_scope,
        (NOT c.is_deleted AND a.user_id=c.user_id AND
          (c.workspace_id IS NULL OR (w.status='active' AND w.user_id=c.user_id AND w.agent_id=c.agent_id))) AS live
        FROM message_followup_events e JOIN messages m ON m.id=e.message_id
        JOIN conversations c ON c.id=m.conversation_id JOIN ai_agents a ON a.id=c.agent_id
        LEFT JOIN chat_workspaces w ON w.id=c.workspace_id WHERE e.message_id=$1 FOR SHARE OF e""", message_id)
    return rows[0] if rows else None


async def _finish(mid: str, generation: int, status: str, outcome: str):
    await db.execute_raw("""UPDATE message_followup_events SET status=$3,outcome=$4,
        last_error=NULL,updated_at=NOW() WHERE message_id=$1 AND generation=$2
        AND status NOT IN ('completed','cancelled')""", mid, generation, status, outcome)


async def run_message_followup(payload: dict) -> None:
    mid, generation = payload.get("message_id"), payload.get("generation")
    if not isinstance(mid, str) or not mid or len(mid) > 200 or type(generation) is not int or generation < 0:
        raise ValueError("invalid_message_followup_payload")
    try:
        await _run_bound_followup(mid, generation)
    except Exception as exc:
        try:
            await db.execute_raw("""UPDATE message_followup_events SET last_error=$3,updated_at=NOW()
                WHERE message_id=$1 AND generation=$2 AND status NOT IN ('completed','cancelled')""",
                mid, generation, type(exc).__name__[:80])
        except Exception:
            logger.warning("Message followup error receipt unavailable")
        raise MessageFollowupFailed(type(exc).__name__) from None


async def _run_bound_followup(mid: str, generation: int) -> None:
    row = await _load_event(mid)
    if row is None or row["status"] in _TERMINAL or row["generation"] > generation:
        return
    if row["generation"] < generation:
        raise RuntimeError("dispatch_not_committed")
    if not row["live"] or row["scope"] != row["current_scope"] or row["source_version"] != row["current_version"]:
        await _finish(mid, generation, "cancelled", "source_changed_or_unavailable")
        return
    scope = row["scope"]
    from app.services.interaction_streak import record_user_message_day_for_conversation
    from app.services.achievements.engine import handle_achievement_event
    from app.services.achievements.events import UserMessageAchievementEvent, AssistantMessageAchievementEvent
    from app.services.achievements.mode import achievement_evaluation_enabled
    from app.services.achievements.source import AchievementSourceUnavailable, resolve_message_event

    if row["role"] == "user":
        # Human takeover still counts as a user interaction day, as before.
        await record_user_message_day_for_conversation(scope["conversation_id"],
            message_id=mid, workspace_id=scope["workspace_id"], user_id=scope["user_id"])
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
        card = metadata.get("component_card")
        card_type = card.get("type") if isinstance(card, dict) else None
        if metadata.get("red_packet") or metadata.get("gift") or card_type in {"red_packet", "gift"}:
            # Preserve the offering route's separate achievement contract.
            await _finish(mid, generation, "completed", "offering_managed_separately")
            return
        event = UserMessageAchievementEvent(scope["user_id"], scope["agent_id"],
            scope["workspace_id"], scope["conversation_id"], mid, "")
    else:
        event = AssistantMessageAchievementEvent(scope["conversation_id"], mid, "")
    try:
        await resolve_message_event(event)
    except AchievementSourceUnavailable:
        outcome = "achievement_source_unavailable"
    else:
        if await achievement_evaluation_enabled():
            await handle_achievement_event(event)
            outcome = "statistics_evaluated"
        else:
            outcome = "achievement_disabled"
    await _finish(mid, generation, "completed", outcome)


async def prune_message_followups() -> int:
    return await prune_followups(db, MESSAGE)


async def inspect_message_followups(*, limit: int = 20, status: str = "all") -> dict:
    return await inspect_followups(db, MESSAGE, limit=limit, status=status)


async def retry_message_followup(message_id: str) -> bool:
    return await retry_followup(db, MESSAGE, message_id, inspect=inspect_runtime_job)
