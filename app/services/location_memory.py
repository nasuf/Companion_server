"""Recoverable, scoped location-share events; a shared point is not a residence.

The message INSERT trigger creates the outbox record in the same transaction.
Redis carries only its ID. A SQL receipt fences concurrent/replayed consumers,
including after a user deletes either derived memory.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from typing import Any

from app.db import db
from app.services.memory.evidence import (
    EvidenceSource,
    bind_memory_evidence,
    content_version,
)
from app.services.memory.input_evidence import (
    prepare_input_origin,
    persist_input_origin,
)
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.storage import repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.taxonomy import resolve_taxonomy
from app.services.runtime.job_queue import enqueue_runtime_job

logger = logging.getLogger(__name__)
EXTRACTOR = "location-share-v1"
JOB_TYPE = "location_share_memory"


def location_place(card: dict[str, Any]) -> str:
    payload = card.get("payload")
    if card.get("type") != "location" or not isinstance(payload, dict):
        raise ValueError("invalid_location_card")
    for key, maximum in (("latitude", 90), ("longitude", 180)):
        value = payload.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or abs(value) > maximum
        ):
            raise ValueError("invalid_location_coordinates")
    # Do not embed precise coordinates in permanent recall text. The private
    # immutable card snapshot retains the accepted point for source auditing.
    parts = [payload.get(k) for k in ("address", "city", "region")]
    place = "，".join(
        dict.fromkeys(
            v.strip()[:200] for v in parts if isinstance(v, str) and v.strip()
        )
    )
    return place or "一个位置点（未提供地址）"


def build_location_memory_texts(
    component_card: dict[str, Any], *, accepted_at: datetime
) -> tuple[str, str]:
    if accepted_at.tzinfo is None:
        accepted_at = accepted_at.replace(tzinfo=timezone.utc)
    when = accepted_at.astimezone(timezone.utc).isoformat(timespec="seconds")
    event = f"在{when}的消息中，用户分享了位置点：{location_place(component_card)}。这仅是当时的位置分享，不代表常住地或Agent所在地。"
    return event, f"我收到过一次用户的位置分享。{event}"


async def dispatch_location_events(
    *, message_id: str | None = None, limit: int = 10
) -> int:
    """Bounded transactional outbox dispatch; retry uncertain Redis admission.

    No local execution fallback: the same idempotency key is retried after a
    lost response. Unknown Redis receipts remain visible as dispatch failures.
    """
    count = 0
    for _ in range(max(1, min(limit, 10))):
        async with db.tx(timeout=timedelta(seconds=8)) as tx:
            rows = await tx.query_raw(
                """SELECT message_id FROM location_memory_events
                WHERE status='pending_dispatch' AND next_dispatch_at<=NOW()
                AND ($1::text IS NULL OR message_id=$1)
                ORDER BY next_dispatch_at,message_id LIMIT 1 FOR UPDATE SKIP LOCKED""",
                message_id,
            )
            if not rows:
                break
            mid = rows[0]["message_id"]
            try:
                job_id = await asyncio.wait_for(
                    enqueue_runtime_job(
                        JOB_TYPE,
                        {"message_id": mid},
                        idempotency_key=f"location-memory:{mid}",
                    ),
                    timeout=3,
                )
            except Exception as exc:
                # Exception classes are diagnostic; never store coordinates,
                # database errors or Redis payloads in a public job failure.
                await tx.execute_raw(
                    """UPDATE location_memory_events SET dispatch_attempts=dispatch_attempts+1,
                    next_dispatch_at=NOW()+INTERVAL '30 seconds',last_error=$2,updated_at=NOW()
                    WHERE message_id=$1""",
                    mid,
                    type(exc).__name__[:80],
                )
                logger.warning(
                    "Location memory dispatch pending",
                    extra={"message_id": mid, "reason": type(exc).__name__},
                )
                break  # A Redis outage must not occupy ten consecutive timeouts.
            else:
                await tx.execute_raw(
                    """UPDATE location_memory_events SET status='queued',job_id=$2,
                    dispatch_attempts=dispatch_attempts+1,last_error=NULL,updated_at=NOW()
                    WHERE message_id=$1""",
                    mid,
                    job_id,
                )
                count += 1
    return count


async def _load_source(database, message_id: str, *, lock: bool = False):
    rows = await database.query_raw(
        """SELECT m.id,m.content,m.metadata,m.created_at,c.id AS conversation_id,
        c.user_id,c.workspace_id,c.agent_id,c.is_deleted,
        location_message_version(m.content,m.metadata,m.created_at) AS version,
        EXISTS(SELECT 1 FROM human_takeovers h WHERE h.user_id=c.user_id AND h.status='active') AS taken_over
        FROM messages m JOIN conversations c ON c.id=m.conversation_id
        JOIN chat_workspaces w ON w.id=c.workspace_id AND w.user_id=c.user_id AND w.agent_id=c.agent_id
        WHERE m.id=$1 AND m.role='user'""" + (" FOR SHARE OF m,c,w" if lock else ""),
        message_id,
    )
    if not rows:
        return None
    source = rows[0]
    if isinstance(source["created_at"], str):
        source["created_at"] = datetime.fromisoformat(
            source["created_at"].replace("Z", "+00:00")
        )
    return source


def _usable(source, version, scope):
    return bool(
        source
        and not source["is_deleted"]
        and not source["taken_over"]
        and source["version"] == version
        and all(
            source.get(k) == scope.get(k)
            for k in ("user_id", "workspace_id", "agent_id", "conversation_id")
        )
        and str((source["metadata"] or {}).get("human_takeover", False)).lower()
        != "true"
    )


async def write_location_share_memories(payload: dict[str, Any]) -> None:
    """Runtime handler. Exceptions propagate to existing retries/dead letters."""
    mid = payload.get("message_id")
    if not isinstance(mid, str) or not 1 <= len(mid) <= 200:
        raise ValueError("invalid_location_event_id")
    events = await db.query_raw(
        "SELECT status,source_version,scope FROM location_memory_events WHERE message_id=$1",
        mid,
    )
    if not events or events[0]["status"] in {"completed", "cancelled"}:
        return
    version = events[0]["source_version"]
    scope = events[0]["scope"]
    source = await _load_source(db, mid)
    prepared = None
    if _usable(source, version, scope):
        card = source["metadata"].get("component_card")
        texts = build_location_memory_texts(card, accepted_at=source["created_at"])
        # Network/embedding work never holds DB locks. Both sides are fixed,
        # bounded structured events; no model extraction or persona inference.
        prepared = [
            (side, text, await generate_embedding(text))
            for side, text in zip(("user", "ai"), texts)
        ]
    stored = []
    async with db.tx(timeout=timedelta(seconds=20)) as tx:
        if source:
            for side in ("user", "ai"):
                await tx.execute_raw(
                    "SELECT pg_advisory_xact_lock($1::bigint)",
                    scope_lock_key(side, source["user_id"], source["workspace_id"]),
                )
        if source:
            await tx.query_raw(
                "SELECT id FROM human_takeovers WHERE user_id=$1 FOR SHARE",
                source["user_id"],
            )
        current = await _load_source(tx, mid, lock=True)
        claims = await tx.query_raw(
            "SELECT status,source_version,scope FROM location_memory_events WHERE message_id=$1 FOR UPDATE",
            mid,
        )
        if not claims or claims[0]["status"] in {"completed", "cancelled"}:
            return
        if (
            not prepared
            or not _usable(current, version, scope)
            or claims[0]["source_version"] != version
        ):
            await tx.execute_raw(
                "UPDATE location_memory_events SET status='cancelled',last_error='source_unavailable_or_changed',updated_at=NOW() WHERE message_id=$1",
                mid,
            )
            return
        origin = prepare_input_origin(
            kind="location_share",
            payload={
                "format": EXTRACTOR,
                "message_id": mid,
                "message_version": version,
                "conversation_id": current["conversation_id"],
                "accepted_at": current["created_at"]
                .replace(tzinfo=timezone.utc)
                .isoformat(),
                "component_card": current["metadata"]["component_card"],
                "content": current["content"],
            },
        )
        input_source = await persist_input_origin(
            tx,
            user_id=current["user_id"],
            workspace_id=current["workspace_id"],
            agent_id=current["agent_id"],
            origin=origin,
        )
        message_source = EvidenceSource(
            "message",
            mid,
            relation="recorded_from",
            expected_role="user",
            expected_version=content_version(current["content"]),
        )
        for side, text, vector in prepared:
            taxonomy = resolve_taxonomy(
                main_category="生活",
                sub_category="交互",
                source=side,
                level=3,
                provenance="user_stated",
            )
            if not taxonomy.allowed:
                raise ValueError("location_event_taxonomy_invalid")
            row = await repo.create(
                source=side,
                database=tx,
                userId=current["user_id"],
                workspaceId=current["workspace_id"],
                content=text,
                level=3,
                importance=0.4,
                type=taxonomy.legacy_type,
                mainCategory=taxonomy.main_category,
                subCategory=taxonomy.sub_category,
                provenance="user_stated",
                occurTime=current["created_at"],
                statementTime=current["created_at"],
                valueUpdatedAt=datetime.now(timezone.utc),
            )
            await store_embedding(row.id, vector, database=tx)
            await bind_memory_evidence(
                database=tx,
                memory_id=row.id,
                side=side,
                user_id=current["user_id"],
                workspace_id=current["workspace_id"],
                sources=(input_source, message_source),
                extractor_version=EXTRACTOR,
            )
            await tx.memorychangelog.create(
                data={
                    "userId": current["user_id"],
                    "workspaceId": current["workspace_id"],
                    "memoryId": row.id,
                    "operation": "insert",
                    "newValue": text,
                }
            )
            stored.append(row.id)
        await tx.execute_raw(
            """UPDATE location_memory_events SET status='completed',
            memory_ids=$2::jsonb,last_error=NULL,updated_at=NOW() WHERE message_id=$1""",
            mid,
            json.dumps(stored),
        )
    await repo.invalidate_scope(source["user_id"], source["workspace_id"])


async def inspect_location_events(*, limit: int = 20) -> dict:
    """Recent bounded receipt sample. Queue disappearance is never success."""
    from app.services.runtime.job_queue import inspect_runtime_job

    if not 1 <= limit <= 50:
        raise ValueError("invalid_location_event_limit")
    rows = await db.query_raw(
        """SELECT message_id,status,job_id,dispatch_attempts,last_error,created_at,updated_at
        FROM location_memory_events ORDER BY created_at DESC,message_id DESC LIMIT $1""",
        limit + 1,
    )
    items = []
    queue_unavailable = False
    for row in rows[:limit]:
        state = "not_dispatched"
        if row["status"] in {"completed", "cancelled"}:
            state = "not_needed"
        elif row["job_id"] and queue_unavailable:
            state = "unavailable"
        elif row["job_id"]:
            try:
                job = await asyncio.wait_for(
                    inspect_runtime_job(row["job_id"]), timeout=1
                )
                state = job["status"] if job else "missing_or_expired"
            except Exception:
                queue_unavailable = True
                state = "unavailable"
        items.append({**row, "queue_state": state})
    return {
        "items": items,
        "limit": limit,
        "has_more": len(rows) > limit,
        "denominator": "recent_sample_only",
        "sampled_at": datetime.now(timezone.utc).isoformat(),
    }
