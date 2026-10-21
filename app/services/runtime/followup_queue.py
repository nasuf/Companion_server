"""Bounded SQL receipt / Redis transport operations for durable follow-ups.

Business qualification and effects belong to each family. Identifiers below
are fixed application constants, never request or job payload values.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging

from app.services.runtime.job_queue import RuntimeJobOrphaned

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FollowupFamily:
    table: str
    identity_column: str
    job_type: str
    namespace: str


MESSAGE = FollowupFamily("message_followup_events", "message_id", "message_statistics", "message-statistics")
TURN = FollowupFamily("turn_followup_events", "turn_id", "turn_statistics", "turn-statistics")
ARRIVAL = FollowupFamily("offline_arrival_followups", "activity_id", "offline_arrival", "offline-arrival")


def _family(value: FollowupFamily) -> FollowupFamily:
    if value not in (MESSAGE, TURN, ARRIVAL):
        raise ValueError("invalid_followup_family")
    return value


class FollowupRetryConflict(ValueError):
    pass


async def dispatch_followups(db, family, *, enqueue, inspect, limit=10, identity=None):
    family = _family(family)
    table, column = family.table, family.identity_column
    dispatched = 0
    for _ in range(max(1, min(int(limit), 10))):
        async with db.tx(timeout=timedelta(seconds=8)) as tx:
            rows = await tx.query_raw(f"""
                SELECT * FROM {table}
                WHERE status IN ('pending_dispatch','queued') AND next_dispatch_at<=NOW()
                  AND ($1::text IS NULL OR {column}=$1)
                ORDER BY next_dispatch_at,{column} LIMIT 1 FOR UPDATE SKIP LOCKED
            """, identity)
            if not rows:
                break
            row = rows[0]
            key, generation = row[column], row["generation"]
            try:
                job = None
                if row["job_id"]:
                    job = await asyncio.wait_for(inspect(row["job_id"]), 3)
                if job is not None:
                    if job["type"] != family.job_type or job["payload"] != {column: key, "generation": generation}:
                        state, reason = "failed", "queue_identity_mismatch"
                    elif job["status"] in {"dead_letter", "failed", "resolved", "succeeded"}:
                        state, reason = "failed", "queue_" + job["status"]
                    elif job["status"] in {"queued", "running"}:
                        state, reason = "queued", row["last_error"]
                    else:
                        state, reason = "failed", "queue_state_unknown"
                    await tx.execute_raw(f"""UPDATE {table} SET status=$2,
                        last_error=$3,next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW()
                        WHERE {column}=$1""", key, state, reason)
                    continue
                if row["job_id"]:
                    generation += 1
                job_id = await asyncio.wait_for(enqueue(
                    family.job_type, {column: key, "generation": generation},
                    idempotency_key=f"{family.namespace}:{key}:{generation}",
                ), 3)
            except RuntimeJobOrphaned:
                await tx.execute_raw(f"""UPDATE {table} SET generation=generation+1,
                    status='pending_dispatch',job_id=NULL,last_error='queue_record_missing',
                    next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW(),
                    dispatch_attempts=dispatch_attempts+1 WHERE {column}=$1""", key)
                break
            except Exception as exc:
                await tx.execute_raw(f"""UPDATE {table} SET
                    dispatch_attempts=dispatch_attempts+1,last_error=$2,
                    next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW()
                    WHERE {column}=$1""", key, type(exc).__name__[:80])
                logger.warning("%s followup dispatch deferred (%s)", family.job_type, type(exc).__name__)
                break
            else:
                await tx.execute_raw(f"""UPDATE {table} SET status='queued',
                    job_id=$2,generation=$3,dispatch_attempts=dispatch_attempts+1,last_error=NULL,
                    next_dispatch_at=NOW()+INTERVAL '30 seconds',updated_at=NOW()
                    WHERE {column}=$1""", key, job_id, generation)
                dispatched += 1
    return dispatched


async def prune_followups(db, family):
    family = _family(family)
    return await db.execute_raw(f"""WITH expired AS MATERIALIZED (
        SELECT {family.identity_column} FROM {family.table} WHERE status IN ('completed','cancelled')
        AND updated_at<NOW()-INTERVAL '30 days' ORDER BY updated_at,{family.identity_column} LIMIT 100
        FOR UPDATE SKIP LOCKED)
        DELETE FROM {family.table} target USING expired
        WHERE target.{family.identity_column}=expired.{family.identity_column}""")


async def inspect_followups(db, family, *, limit=20, status="all"):
    family = _family(family)
    limit = max(1, min(int(limit), 50))
    if status not in {"all", "failed", "pending_dispatch", "queued", "completed", "cancelled"}:
        raise ValueError("invalid_message_followup_status")
    where, params = ("", []) if status == "all" else ("WHERE status=$2", [status])
    extra = ",message_id" if family == TURN else ""
    rows = await db.query_raw(f"""SELECT {family.identity_column}{extra},status,job_id,generation,dispatch_attempts,
        last_error,outcome,created_at,updated_at FROM {family.table}
        {where} ORDER BY created_at DESC,{family.identity_column} DESC LIMIT $1""", limit + 1, *params)
    return {"sampled_at": datetime.now(timezone.utc).isoformat(), "limit": limit,
        "has_more": len(rows) > limit, "denominator": "recent_sample_only",
        "retention_days": 30, "status_filter": status, "items": rows[:limit]}


async def retry_followup(db, family, identity, *, inspect):
    family = _family(family)
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        rows = await tx.query_raw(f'SELECT * FROM {family.table} WHERE {family.identity_column}=$1 FOR UPDATE', identity)
        if not rows:
            return False
        row = rows[0]
        if row['status'] != 'failed':
            raise FollowupRetryConflict('只允许恢复失败记录，请刷新状态。')
        if row['job_id']:
            job = await asyncio.wait_for(inspect(row['job_id']), 3)
            if job is not None:
                if job['type'] != family.job_type or job['payload'] != {family.identity_column: identity, 'generation': row['generation']}:
                    raise FollowupRetryConflict('队列关联不匹配，需要先核对任务来源。')
                if job['status'] not in {'dead_letter', 'failed', 'resolved', 'succeeded'}:
                    raise FollowupRetryConflict('任务仍在处理，请等待并刷新状态。')
        await tx.execute_raw(f"""UPDATE {family.table} SET status='pending_dispatch',
            generation=generation+1,job_id=NULL,last_error=NULL,outcome=NULL,
            next_dispatch_at=NOW(),updated_at=NOW() WHERE {family.identity_column}=$1""", identity)
        return True
