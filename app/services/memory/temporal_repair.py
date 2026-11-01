"""Explicit, bounded repair of missing projections from already pinned evidence.

No text-similarity matching and no date guessing. Old records without a stored,
verifiable interpretation remain unknown; the administrator cannot invent a
user confirmation or copy an event date from a different source.
"""

from datetime import datetime, timedelta, timezone
import json
import logging

from app.db import db
from app.services.memory.evidence import (
    EvidenceSource,
    _resolve_source,
    content_version,
    message_statement_time,
)
from app.services.memory.input_evidence import (
    prepare_input_origin,
    persist_input_origin,
)
from app.services.memory.lifecycle.lazy_update import scope_lock_key
from app.services.memory.manual_write import (
    MemoryWriteError,
    memory_scope,
    lock_actor,
    lock_memories,
)
from app.services.memory.storage import repo
from app.services.memory.temporal_assertions import (
    persist_assertion,
    validate_assertion_source,
)
from app.services.memory.recording.validity import public_validity

logger = logging.getLogger(__name__)


async def _target(database, *, user_id, workspace_id, side, memory_id):
    if side not in {"user", "ai"}:
        raise MemoryWriteError(422, "invalid_memory_side")
    model = database.usermemory if side == "user" else database.aimemory
    row = await model.find_unique(where={"id": memory_id})
    if not row or row.userId != user_id or row.workspaceId != workspace_id:
        raise MemoryWriteError(404, "memory_not_found_in_scope")
    record = repo._to_record(row, side)
    scope = await memory_scope(database, record)
    return record, scope


async def _rows(database, record, evidence_id=None):
    return await database.query_raw(
        """SELECT e.id,e.temporal_version_id,e.source_message_id,e.source_version,
          e.source_role,e.source_statement_time,e.source_card_version,e.temporal_basis,
          m.conversation_id,m.content,m.created_at,m.metadata,a.id AS existing_assertion
        FROM memory_evidence_links e
        JOIN memory_temporal_versions v ON v.id=e.temporal_version_id
          AND v.side=e.memory_source AND v.memory_id=e.memory_id AND v.user_id=e.user_id AND v.workspace_id=e.workspace_id
        LEFT JOIN messages m ON m.id=e.source_message_id
        LEFT JOIN memory_temporal_assertions a ON a.id=e.id
        WHERE e.memory_id=$1 AND e.memory_source=$2 AND e.user_id=$3 AND e.workspace_id=$4
          AND e.source_kind='message' AND e.content_version=$5
          AND e.temporal_version_id=(SELECT id FROM memory_temporal_versions
            WHERE memory_id=$1 AND side=$2 AND user_id=$3 AND workspace_id=$4 ORDER BY id DESC LIMIT 1)
          AND ($6::text IS NULL OR e.id=$6)
        ORDER BY e.created_at DESC,e.id LIMIT 21""",
        record.id,
        record.source,
        record.userId,
        record.workspaceId,
        content_version(record.content),
        evidence_id,
    )


def _fingerprint(row, record):
    return content_version(
        json.dumps(
            {
                "row": {k: v for k, v in row.items() if k != "existing_assertion"},
                "memory_content_version": content_version(record.content),
                "memory_updated_at": record.updatedAt,
            },
            sort_keys=True,
            default=str,
        )
    )


async def _validate(database, row, record, scope):
    if (
        not row["source_statement_time"]
        or not row["content"]
        or not row["conversation_id"]
        or row["source_card_version"] is not None
        or "component_card" in (row.get("metadata") or {})
    ):
        raise MemoryWriteError(409, "temporal_repair_source_unavailable")
    source = EvidenceSource(
        "message",
        row["source_message_id"],
        relation="extracted_from",
        expected_role=row["source_role"],
        expected_version=row["source_version"],
        expected_statement_time=message_statement_time(row["source_statement_time"]),
        expected_card_version=row["source_card_version"],
        expected_conversation_id=row["conversation_id"],
        input_text=row["content"],
        temporal_basis=row["temporal_basis"],
    )
    origin = await _resolve_source(
        database,
        source,
        {
            "user_id": record.userId,
            "workspace_id": record.workspaceId,
            "agent_id": scope["agent_id"],
            "existing_memory_id": record.id,
        },
    )
    validated = validate_assertion_source(
        source=source,
        origin={**origin, "role": row["source_role"]},
        side=record.source,
        revision_id=row["temporal_version_id"],
    )
    if validated is None:
        raise MemoryWriteError(409, "temporal_repair_no_explicit_basis")
    return source, origin, validated


async def preview(*, user_id, workspace_id, side, memory_id):
    record, scope = await _target(
        db, user_id=user_id, workspace_id=workspace_id, side=side, memory_id=memory_id
    )
    rows = await _rows(db, record)
    items = []
    for row in rows[:20]:
        item = {
            "evidence_id": row["id"],
            "expected_version": _fingerprint(row, record),
            "classification": "unknown",
            "reason": "no_explicit_basis",
            "proposed": None,
        }
        if row["existing_assertion"]:
            item.update(classification="already_recorded", reason="projection_exists")
        else:
            try:
                _, _, validated = await _validate(db, row, record, scope)
                item.update(
                    classification="provable",
                    reason="pinned_explicit_basis",
                    proposed=public_validity(validated[0]),
                )
            except (ValueError, MemoryWriteError):
                item.update(
                    classification=(
                        "needs_review" if row["temporal_basis"] else "unknown"
                    ),
                    reason="source_or_interpretation_unavailable",
                )
        items.append(item)
    return {
        "items": items,
        "limit": 20,
        "truncated": len(rows) > 20,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "state": "recorded" if rows else "no_pinned_candidates",
        "complete_history": False,
        "writes_performed": 0,
    }


async def apply(
    *, user_id, workspace_id, side, memory_id, evidence_id, expected_version, actor
):
    if actor.get("role") != "admin":
        raise MemoryWriteError(403, "admin_required")
    record, scope = await _target(
        db, user_id=user_id, workspace_id=workspace_id, side=side, memory_id=memory_id
    )
    async with db.tx(timeout=timedelta(seconds=15)) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock($1::bigint)",
            scope_lock_key(side, user_id, workspace_id),
        )
        if await memory_scope(tx, record, lock=True) != scope:
            raise MemoryWriteError(409, "memory_scope_changed")
        await lock_actor(tx, actor["sub"], "admin")
        await lock_memories(tx, [record], allow_archived=True)
        rows = await _rows(tx, record, evidence_id)
        if len(rows) != 1 or _fingerprint(rows[0], record) != expected_version:
            raise MemoryWriteError(409, "temporal_repair_preview_stale")
        row = rows[0]
        source, origin, _ = await _validate(tx, row, record, scope)
        if row["existing_assertion"]:
            return {"status": "already_applied", "evidence_id": evidence_id}
        await persist_assertion(
            database=tx,
            evidence_id=evidence_id,
            revision_id=row["temporal_version_id"],
            source=source,
            origin={**origin, "role": row["source_role"]},
            user_id=user_id,
            workspace_id=workspace_id,
            side=side,
            memory_id=memory_id,
        )
        origin_input = prepare_input_origin(
            kind="admin_repair",
            payload={
                "operation": "temporal_projection_backfill",
                "actor_id": actor["sub"],
                "memory_id": memory_id,
                "memory_side": side,
                "evidence_id": evidence_id,
                "preview_version": expected_version,
                "source_version": source.expected_version,
                "user_confirmed": False,
            },
        )
        receipt = await persist_input_origin(
            tx,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=scope["agent_id"],
            origin=origin_input,
        )
        await tx.memorychangelog.create(
            data={
                "userId": user_id,
                "workspaceId": workspace_id,
                "memoryId": memory_id,
                "operation": "temporal_projection_backfill",
                "newValue": json.dumps(
                    {
                        "evidence_id": evidence_id,
                        "admin_id": actor["sub"],
                        "input_origin": receipt.ref,
                        "user_confirmed": False,
                    }
                ),
            }
        )
    try:
        await repo.invalidate_scope(user_id, workspace_id)
    except Exception:
        logger.warning(
            "temporal repair committed; cache invalidation pending", exc_info=True
        )
    return {"status": "applied", "evidence_id": evidence_id}
