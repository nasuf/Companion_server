"""Message statistics outbox; Redis is transport, SQL owns durable completion.

Only idempotent persisted-message rules run here. This is not a retry wrapper
for memory extraction, personality, patience, complete turns or notifications.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging

from app.db import db
from app.services.runtime.job_queue import (
    RuntimeJobOrphaned, enqueue_runtime_job, inspect_runtime_job,
)

logger = logging.getLogger(__name__)
JOB_TYPE = "message_statistics"
_TERMINAL = {"completed", "cancelled"}


class MessageFollowupFailed(RuntimeError):
    """Public failures contain a class only, never SQL, messages or credentials."""


async def dispatch_message_followups(*, limit: int = 10, message_id: str | None = None) -> int:
    """Bounded admission/reconciliation. Failed jobs require explicit retry.

    Lost Redis records can be re-admitted under a new generation because all
    admitted effects use durable uniqueness. A stale generation cannot mark a
    newer dispatch complete. Uncertain admission retries the same key.
    """
    dispatched = 0
    for _ in range(max(1, min(int(limit), 10))):
        async with db.tx(timeout=timedelta(seconds=8)) as tx:
            rows = await tx.query_raw("""
                SELECT * FROM message_followup_events
                WHERE status IN ('pending_dispatch','queued') AND next_dispatch_at<=NOW()
                  AND ($1::text IS NULL OR message_id=$1)
                ORDER BY next_dispatch_at,message_id LIMIT 1 FOR UPDATE SKIP LOCKED
            """, message_id)
            if not rows:
                break
            row = rows[0]
            mid, generation = row["message_id"], row["generation"]
            try:
                job = None
                if row["job_id"]:
                    job = await asyncio.wait_for(inspect_runtime_job(row["job_id"]), 3)
                if job is not None:
                    if (job["type"] != JOB_TYPE or job["payload"] !=
                            {"message_id": mid, "generation": generation}):
                        state, reason = "failed", "queue_identity_mismatch"
                    elif job["status"] in {"dead_letter", "failed", "resolved", "succeeded"}:
                        state, reason = "failed", "queue_" + job["status"]
                    elif job["status"] in {"queued", "running"}:
                        state, reason = "queued", row["last_error"]
                    else:
                        state, reason = "failed", "queue_state_unknown"
                    await tx.execute_raw("""UPDATE message_followup_events SET status=$2,
                        last_error=$3,next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW()
                        WHERE message_id=$1""", mid, state, reason)
                    continue
                if row["job_id"]:
                    generation += 1
                job_id = await asyncio.wait_for(enqueue_runtime_job(
                    JOB_TYPE, {"message_id": mid, "generation": generation},
                    idempotency_key=f"message-statistics:{mid}:{generation}",
                ), 3)
            except RuntimeJobOrphaned:
                # Admission's receipt survived but its job disappeared. No local
                # execution: next scan uses another generation of the same event.
                await tx.execute_raw("""UPDATE message_followup_events SET generation=generation+1,
                    status='pending_dispatch',job_id=NULL,last_error='queue_record_missing',
                    next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW(),
                    dispatch_attempts=dispatch_attempts+1 WHERE message_id=$1""", mid)
                break
            except Exception as exc:
                await tx.execute_raw("""UPDATE message_followup_events SET
                    dispatch_attempts=dispatch_attempts+1,last_error=$2,
                    next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW()
                    WHERE message_id=$1""", mid, type(exc).__name__[:80])
                logger.warning("Message followup dispatch deferred (%s)", type(exc).__name__)
                break
            else:
                await tx.execute_raw("""UPDATE message_followup_events SET status='queued',
                    job_id=$2,generation=$3,dispatch_attempts=dispatch_attempts+1,last_error=NULL,
                    next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW()
                    WHERE message_id=$1""", mid, job_id, generation)
                dispatched += 1
    return dispatched


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
    """Terminal diagnostics only, never messages, achievements or failed work."""
    return await db.execute_raw("""DELETE FROM message_followup_events WHERE message_id IN (
        SELECT message_id FROM message_followup_events WHERE status IN ('completed','cancelled')
        AND updated_at<NOW()-INTERVAL '30 days' ORDER BY updated_at,message_id LIMIT 100
        FOR UPDATE SKIP LOCKED)""")


async def inspect_message_followups(*, limit: int = 20, status: str = "all") -> dict:
    limit = max(1, min(int(limit), 50))
    if status not in {"all", "failed", "pending_dispatch", "queued", "completed", "cancelled"}:
        raise ValueError("invalid_message_followup_status")
    where, params = ("", []) if status == "all" else ("WHERE status=$2", [status])
    rows = await db.query_raw(f"""SELECT message_id,status,job_id,generation,dispatch_attempts,
        last_error,outcome,created_at,updated_at FROM message_followup_events
        {where} ORDER BY created_at DESC,message_id DESC LIMIT $1""", limit + 1, *params)
    return {"sampled_at": datetime.now(timezone.utc).isoformat(), "limit": limit,
        "has_more": len(rows) > limit, "denominator": "recent_sample_only",
        "retention_days": 30, "status_filter": status, "items": rows[:limit]}


class FollowupRetryConflict(ValueError):
    pass


async def retry_message_followup(message_id: str) -> bool:
    """Explicit operator recovery. Never reset running or completed work."""
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        rows = await tx.query_raw('SELECT * FROM message_followup_events WHERE message_id=$1 FOR UPDATE', message_id)
        if not rows:
            return False
        row = rows[0]
        if row['status'] != 'failed':
            raise FollowupRetryConflict('只允许恢复失败记录，请刷新状态。')
        if row['job_id']:
            job = await asyncio.wait_for(inspect_runtime_job(row['job_id']), 3)
            if job is not None:
                if job['type'] != JOB_TYPE or job['payload'] != {'message_id': message_id, 'generation': row['generation']}:
                    raise FollowupRetryConflict('队列关联不匹配，需要先核对任务来源。')
                if job['status'] not in {'dead_letter', 'failed', 'resolved', 'succeeded'}:
                    raise FollowupRetryConflict('任务仍在处理，请等待并刷新状态。')
        await tx.execute_raw("""UPDATE message_followup_events SET status='pending_dispatch',
            generation=generation+1,job_id=NULL,last_error=NULL,outcome=NULL,
            next_dispatch_at=NOW(),updated_at=NOW() WHERE message_id=$1""", message_id)
        return True
