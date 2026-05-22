"""Materialized memory quality state.

`quality.py` remains the derivation engine. This module persists the derived
state so admin dashboards and consolidation jobs do not need to recompute a
memory's trust state from the full changelog on every read.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.services.memory.lifecycle.quality import derive_memory_quality
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.repo import MemoryRecord, Source

logger = logging.getLogger(__name__)

_ADMIN_REPAIR_OPS = {
    "admin_verified",
    "repair_archive",
    "repair_downgrade",
    "repair_edit",
    "repair_insert_replacement",
    "repair_merge",
    "repair_merge_archived",
}


async def refresh_memory_quality_state(
    memory_id: str,
    *,
    source: Source | None = None,
) -> dict[str, Any] | None:
    """Recompute and upsert one memory's materialized quality state."""
    record = await memory_repo.find_unique(memory_id)
    if record is None:
        return None
    if source and record.source != source:
        return None
    states = await refresh_memory_quality_states([record])
    return states[0] if states else None


async def refresh_memory_quality_states(records: list[MemoryRecord]) -> list[dict[str, Any]]:
    if not records:
        return []
    derived = await derive_memory_quality(records)
    results: list[dict[str, Any]] = []
    for record in records:
        q = derived.get(record.id)
        if q is None:
            continue
        admin_state = await _admin_repair_state(record.id)
        evidence_ids = list(dict.fromkeys(q.evidence_message_ids))
        last_verified_at = _max_dt(q.last_verified_at, admin_state.get("last_verified_at"))
        payload = {
            "signals": q.signals,
            "quality_source": "materialized_from_changelog_v1",
        }
        row = await _upsert_state(
            memory_id=record.id,
            memory_source=record.source,
            user_id=record.userId,
            workspace_id=record.workspaceId,
            confidence=q.confidence,
            evidence_message_ids=evidence_ids,
            last_verified_at=last_verified_at,
            verified_by=admin_state.get("verified_by"),
            contradiction_state=q.contradiction_state,
            user_corrected_count=q.user_corrected_count,
            admin_repaired_count=int(admin_state.get("admin_repaired_count") or 0),
            access_count=q.access_count,
            last_repair_item_id=admin_state.get("last_repair_item_id"),
            superseded_by_memory_id=admin_state.get("superseded_by_memory_id"),
            signals=payload,
            source_updated_at=record.updatedAt,
        )
        if row:
            results.append(row)
    return results


async def refresh_quality_state_for_changelog(memory_id: str) -> None:
    """Best-effort hook for changelog writers."""
    try:
        await refresh_memory_quality_state(memory_id)
    except Exception as e:
        logger.debug("memory quality state refresh failed for %s: %s", memory_id, e)


async def backfill_memory_quality_states(
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    limit: int = 500,
) -> dict[str, int]:
    """Backfill active and archived memory quality states in bounded batches."""
    where: dict[str, Any] = {}
    if user_id:
        where["userId"] = user_id
    if workspace_id:
        where["workspaceId"] = workspace_id
    records = await memory_repo.find_many(
        source=None,
        where=where,
        order={"updatedAt": "desc"},
        take=max(1, min(limit, 2000)),
        allow_cross_user=True,
    )
    states = await refresh_memory_quality_states(records)
    return {"checked": len(records), "updated": len(states)}


async def mark_memory_superseded(
    *,
    memory_id: str,
    source: Source,
    superseded_by_memory_id: str,
    repair_item_id: str | None = None,
) -> None:
    record = await memory_repo.find_unique(memory_id)
    if record is None:
        return
    await refresh_memory_quality_state(memory_id, source=source)
    await db.query_raw(
        """
        UPDATE memory_quality_states
        SET superseded_by_memory_id = $3,
            last_repair_item_id = COALESCE($4, last_repair_item_id),
            updated_at = CURRENT_TIMESTAMP
        WHERE memory_id = $1 AND memory_source = $2
        """,
        memory_id,
        source,
        superseded_by_memory_id,
        repair_item_id,
    )


async def get_memory_quality_state(
    memory_id: str,
    *,
    source: Source | None = None,
) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM memory_quality_states
        WHERE memory_id = $1
          AND ($2::text IS NULL OR memory_source = $2)
        LIMIT 1
        """,
        memory_id,
        source,
    )
    return _serialize_state(rows[0]) if rows else None


async def list_low_quality_memory_states(
    *,
    limit: int = 100,
    max_confidence: float = 0.55,
) -> list[dict[str, Any]]:
    rows = await db.query_raw(
        """
        SELECT *
        FROM memory_quality_states
        WHERE confidence <= $1
           OR contradiction_state <> 'none'
           OR superseded_by_memory_id IS NOT NULL
        ORDER BY confidence ASC, updated_at DESC
        LIMIT $2
        """,
        max_confidence,
        max(1, min(limit, 500)),
    )
    return [_serialize_state(row) for row in rows]


async def _admin_repair_state(memory_id: str) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        SELECT operation, new_value, created_at
        FROM memory_changelogs
        WHERE memory_id = $1
          AND operation = ANY($2::text[])
        ORDER BY created_at ASC
        """,
        memory_id,
        list(_ADMIN_REPAIR_OPS),
    )
    count = 0
    last_verified_at: datetime | None = None
    verified_by: str | None = None
    last_repair_item_id: str | None = None
    superseded_by_memory_id: str | None = None
    for row in rows:
        count += 1
        created_at = _coerce_dt(_row_value(row, "created_at", "createdAt"))
        if created_at:
            last_verified_at = _max_dt(last_verified_at, created_at)
        payload = _json_dict(_row_value(row, "new_value", "newValue"))
        if not payload:
            continue
        verified_by = str(payload.get("admin_id") or verified_by or "") or None
        last_repair_item_id = str(payload.get("repair_item_id") or last_repair_item_id or "") or None
        after = payload.get("after")
        if isinstance(after, str):
            after_dict = _json_dict(after)
        elif isinstance(after, dict):
            after_dict = after
        else:
            after_dict = {}
        superseded_by_memory_id = str(after_dict.get("merged_into") or superseded_by_memory_id or "") or None
    return {
        "admin_repaired_count": count,
        "last_verified_at": last_verified_at,
        "verified_by": verified_by,
        "last_repair_item_id": last_repair_item_id,
        "superseded_by_memory_id": superseded_by_memory_id,
    }


async def _upsert_state(**values: Any) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        INSERT INTO memory_quality_states (
            memory_id, memory_source, user_id, workspace_id, confidence,
            evidence_message_ids, last_verified_at, verified_by,
            contradiction_state, user_corrected_count, admin_repaired_count,
            access_count, last_repair_item_id, superseded_by_memory_id,
            signals, source_updated_at, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5,
            $6::text[], $7::timestamp, $8,
            $9, $10, $11,
            $12, $13, $14,
            $15::jsonb, $16::timestamp, CURRENT_TIMESTAMP
        )
        ON CONFLICT (memory_id, memory_source) DO UPDATE SET
            user_id = EXCLUDED.user_id,
            workspace_id = EXCLUDED.workspace_id,
            confidence = EXCLUDED.confidence,
            evidence_message_ids = EXCLUDED.evidence_message_ids,
            last_verified_at = EXCLUDED.last_verified_at,
            verified_by = COALESCE(EXCLUDED.verified_by, memory_quality_states.verified_by),
            contradiction_state = EXCLUDED.contradiction_state,
            user_corrected_count = EXCLUDED.user_corrected_count,
            admin_repaired_count = EXCLUDED.admin_repaired_count,
            access_count = EXCLUDED.access_count,
            last_repair_item_id = COALESCE(EXCLUDED.last_repair_item_id, memory_quality_states.last_repair_item_id),
            superseded_by_memory_id = COALESCE(EXCLUDED.superseded_by_memory_id, memory_quality_states.superseded_by_memory_id),
            signals = EXCLUDED.signals,
            source_updated_at = EXCLUDED.source_updated_at,
            updated_at = CURRENT_TIMESTAMP
        RETURNING *
        """,
        values["memory_id"],
        values["memory_source"],
        values["user_id"],
        values.get("workspace_id"),
        float(values["confidence"]),
        values.get("evidence_message_ids") or [],
        _naive(values.get("last_verified_at")),
        values.get("verified_by"),
        values.get("contradiction_state") or "none",
        int(values.get("user_corrected_count") or 0),
        int(values.get("admin_repaired_count") or 0),
        int(values.get("access_count") or 0),
        values.get("last_repair_item_id"),
        values.get("superseded_by_memory_id"),
        json.dumps(values.get("signals") or {}, ensure_ascii=False),
        _naive(values.get("source_updated_at")),
    )
    return _serialize_state(rows[0]) if rows else None


def _serialize_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": _row_value(row, "memory_id", "memoryId"),
        "memory_source": _row_value(row, "memory_source", "memorySource"),
        "user_id": _row_value(row, "user_id", "userId"),
        "workspace_id": _row_value(row, "workspace_id", "workspaceId"),
        "confidence": float(_row_value(row, "confidence") or 0.0),
        "evidence_message_ids": list(_row_value(row, "evidence_message_ids", "evidenceMessageIds") or []),
        "last_verified_at": _iso(_row_value(row, "last_verified_at", "lastVerifiedAt")),
        "verified_by": _row_value(row, "verified_by", "verifiedBy"),
        "contradiction_state": _row_value(row, "contradiction_state", "contradictionState") or "none",
        "user_corrected_count": int(_row_value(row, "user_corrected_count", "userCorrectedCount") or 0),
        "admin_repaired_count": int(_row_value(row, "admin_repaired_count", "adminRepairedCount") or 0),
        "access_count": int(_row_value(row, "access_count", "accessCount") or 0),
        "last_repair_item_id": _row_value(row, "last_repair_item_id", "lastRepairItemId"),
        "superseded_by_memory_id": _row_value(row, "superseded_by_memory_id", "supersededByMemoryId"),
        "signals": _json_value(_row_value(row, "signals")) or {},
        "source_updated_at": _iso(_row_value(row, "source_updated_at", "sourceUpdatedAt")),
        "updated_at": _iso(_row_value(row, "updated_at", "updatedAt")),
    }


def _row_value(row: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(row, dict) and key in row:
            return row[key]
        if hasattr(row, key):
            return getattr(row, key)
    return None


def _json_dict(raw: Any) -> dict[str, Any]:
    data = _json_value(raw)
    return data if isinstance(data, dict) else {}


def _json_value(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return None


def _coerce_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _naive(value: Any) -> datetime | None:
    dt = _coerce_dt(value)
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _iso(value: Any) -> str | None:
    dt = _coerce_dt(value)
    return dt.isoformat() if dt else None


def _max_dt(left: datetime | None, right: datetime | None) -> datetime | None:
    if right is None:
        return left
    if left is None:
        return right
    return right if right > left else left
