"""Derived memory quality signals.

The memory tables currently stay schema-stable. P1 quality fields are derived
from `memory_changelogs` so API callers can inspect evidence and correction
state without a migration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.services.memory.storage.repo import MemoryRecord

_CORRECTION_OPS = {
    "user_edit",
    "contradiction_archived",
    "contradiction_new",
    "retrieval_feedback_confirmed",
}
_VERIFICATION_OPS = {
    "evidence_linked",
    "user_emphasized",
    "contradiction_new",
    "user_edit",
    "admin_verified",
    "repair_edit",
    "repair_merge",
    "repair_insert_replacement",
}
_NEGATIVE_OPS = {
    "contradiction_archived",
    "user_bulk_delete",
    "workspace_wipe",
}


@dataclass
class MemoryQuality:
    confidence: float
    evidence_message_ids: list[str] = field(default_factory=list)
    last_verified_at: datetime | None = None
    contradiction_state: str = "none"
    user_corrected_count: int = 0
    access_count: int = 0
    signals: list[str] = field(default_factory=list)


def serialize_quality(q: MemoryQuality | None) -> dict[str, Any] | None:
    if q is None:
        return None
    return {
        "confidence": q.confidence,
        "evidence_message_ids": q.evidence_message_ids,
        "last_verified_at": q.last_verified_at.isoformat() if q.last_verified_at else None,
        "contradiction_state": q.contradiction_state,
        "user_corrected_count": q.user_corrected_count,
        "access_count": q.access_count,
        "signals": q.signals,
    }


async def derive_memory_quality(records: list[MemoryRecord]) -> dict[str, MemoryQuality]:
    if not records:
        return {}
    record_by_id = {record.id: record for record in records}
    rows = await db.query_raw(
        """
        SELECT memory_id, operation, old_value, new_value, created_at
        FROM memory_changelogs
        WHERE memory_id = ANY($1::text[])
        ORDER BY created_at ASC
        """,
        list(record_by_id),
    )
    grouped: dict[str, list[dict[str, Any]]] = {mid: [] for mid in record_by_id}
    for row in rows:
        mid = _row_value(row, "memory_id", "memoryId")
        if mid in grouped:
            grouped[mid].append(row)

    return {
        mid: _derive_one(record_by_id[mid], grouped.get(mid, []))
        for mid in record_by_id
    }


async def log_memory_evidence(
    *,
    user_id: str,
    memory_id: str,
    message_ids: list[str],
    workspace_id: str | None = None,
) -> None:
    if not message_ids:
        return
    from app.services.memory.storage.persistence import log_memory_changelog

    unique_ids = list(dict.fromkeys(str(mid) for mid in message_ids if mid))
    if not unique_ids:
        return
    await log_memory_changelog(
        user_id,
        memory_id,
        "evidence_linked",
        new_value=json.dumps({"message_ids": unique_ids}, ensure_ascii=False),
        workspace_id=workspace_id,
    )


def derive_memory_quality_from_changelog_rows(
    *,
    memory_id: str,
    importance: float,
    rows: list[dict[str, Any]],
    source: str = "user",
    is_archived: bool = False,
) -> MemoryQuality:
    """Derive quality when callers already fetched changelog rows.

    Used by debug/trace enrichment paths to avoid another DB round trip and to
    keep the public async `derive_memory_quality()` contract unchanged.
    """
    record = MemoryRecord(
        id=memory_id,
        userId="",
        type=None,
        source="ai" if source == "ai" else "user",
        level=3,
        content="",
        summary=None,
        importance=importance,
        mentionCount=0,
        isArchived=is_archived,
        occurTime=None,
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc),
    )
    return _derive_one(record, rows)


def _derive_one(record: MemoryRecord, rows: list[dict[str, Any]]) -> MemoryQuality:
    evidence_ids: list[str] = []
    access_count = 0
    correction_count = 0
    negative_count = 0
    emphasized = False
    last_verified_at: datetime | None = None
    last_correction_op: str | None = None

    for row in rows:
        operation = str(_row_value(row, "operation") or "")
        created_at = _coerce_dt(_row_value(row, "created_at", "createdAt"))
        if operation == "access":
            access_count += 1
        if operation == "user_emphasized":
            emphasized = True
        if operation in _CORRECTION_OPS:
            correction_count += 1
            last_correction_op = operation
        if operation in _NEGATIVE_OPS:
            negative_count += 1
            last_correction_op = operation
        if operation in _VERIFICATION_OPS and created_at is not None:
            last_verified_at = _max_dt(last_verified_at, created_at)
        if operation == "evidence_linked":
            for message_id in _extract_message_ids(_row_value(row, "new_value", "newValue")):
                if message_id not in evidence_ids:
                    evidence_ids.append(message_id)

    confidence = float(record.importance or 0.0)
    confidence += min(0.12, 0.02 * len(evidence_ids))
    confidence += min(0.10, 0.01 * access_count)
    if emphasized:
        confidence += 0.08
    confidence -= min(0.25, 0.08 * correction_count)
    confidence -= min(0.20, 0.10 * negative_count)
    if record.isArchived:
        confidence = min(confidence, 0.2)
    confidence = round(max(0.0, min(0.99, confidence)), 2)

    contradiction_state = "none"
    if last_correction_op == "contradiction_archived":
        contradiction_state = "archived_by_contradiction"
    elif last_correction_op in {"contradiction_new", "user_edit", "retrieval_feedback_confirmed"}:
        contradiction_state = "corrected"
    elif negative_count:
        contradiction_state = "archived"

    signals: list[str] = []
    if evidence_ids:
        signals.append("has_evidence_messages")
    if access_count:
        signals.append("used_in_retrieval")
    if emphasized:
        signals.append("user_emphasized")
    if correction_count:
        signals.append("user_corrected")
    if record.isArchived:
        signals.append("archived")

    return MemoryQuality(
        confidence=confidence,
        evidence_message_ids=evidence_ids[:20],
        last_verified_at=last_verified_at,
        contradiction_state=contradiction_state,
        user_corrected_count=correction_count,
        access_count=access_count,
        signals=signals,
    )


def _extract_message_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    try:
        data = json.loads(str(raw))
    except Exception:
        return []
    ids = data.get("message_ids") if isinstance(data, dict) else None
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids if item]


def _row_value(row: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(row, dict) and key in row:
            return row[key]
        if hasattr(row, key):
            return getattr(row, key)
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


def _max_dt(left: datetime | None, right: datetime) -> datetime:
    if left is None:
        return right
    return right if right > left else left
