"""Operational repair queue for memory quality issues.

This is deliberately separate from the chat hot-path contradiction flow. The
hot path still auto-confirms and applies clear user corrections; this queue is
for failures, ambiguous evidence, and admin-reviewed memory safety reports.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.db import db

logger = logging.getLogger(__name__)

OPEN = "open"
RESOLVED = "resolved"
DISMISSED = "dismissed"
VALID_STATUSES = {OPEN, RESOLVED, DISMISSED}
VALID_SEVERITIES = {"low", "medium", "high", "critical"}


def serialize_repair_item(row: dict[str, Any]) -> dict[str, Any]:
    def _iso(value: Any) -> str | None:
        return value.isoformat() if isinstance(value, datetime) else None

    evidence = row.get("evidence")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            evidence = {"raw": evidence}

    return {
        "id": row.get("id"),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "status": row.get("status"),
        "severity": row.get("severity"),
        "user_id": row.get("user_id"),
        "agent_id": row.get("agent_id"),
        "workspace_id": row.get("workspace_id"),
        "conversation_id": row.get("conversation_id"),
        "message_id": row.get("message_id"),
        "memory_id": row.get("memory_id"),
        "memory_source": row.get("memory_source"),
        "reason": row.get("reason"),
        "suggested_action": row.get("suggested_action"),
        "evidence": evidence if isinstance(evidence, dict) else {},
        "resolution_note": row.get("resolution_note"),
        "resolved_by_id": row.get("resolved_by_id"),
        "resolved_at": _iso(row.get("resolved_at")),
        "created_at": _iso(row.get("created_at")),
        "updated_at": _iso(row.get("updated_at")),
    }


async def create_memory_repair_item(
    *,
    source_type: str,
    source_id: str | None = None,
    severity: str = "medium",
    user_id: str | None = None,
    agent_id: str | None = None,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    memory_id: str | None = None,
    memory_source: str | None = None,
    reason: str | None = None,
    suggested_action: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an open repair item, deduping by source id or memory/source pair."""
    source_type = source_type.strip()
    if not source_type:
        raise ValueError("source_type is required")
    if severity not in VALID_SEVERITIES:
        severity = "medium"

    existing = await _find_existing_open_item(
        source_type=source_type,
        source_id=source_id,
        memory_id=memory_id,
    )
    if existing:
        return serialize_repair_item(existing)

    rows = await db.query_raw(
        """
        INSERT INTO memory_repair_items (
            id, source_type, source_id, status, severity,
            user_id, agent_id, workspace_id, conversation_id, message_id,
            memory_id, memory_source, reason, suggested_action, evidence
        )
        VALUES (
            $1, $2, $3, 'open', $4,
            $5, $6, $7, $8, $9,
            $10, $11, $12, $13, $14::jsonb
        )
        ON CONFLICT DO NOTHING
        RETURNING *
        """,
        str(uuid.uuid4()),
        source_type,
        source_id,
        severity,
        user_id,
        agent_id,
        workspace_id,
        conversation_id,
        message_id,
        memory_id,
        memory_source,
        reason,
        suggested_action,
        json.dumps(evidence or {}, ensure_ascii=False),
    )
    if rows:
        return serialize_repair_item(rows[0])

    # Extremely unlikely id collision fallback.
    existing = await _find_existing_open_item(
        source_type=source_type,
        source_id=source_id,
        memory_id=memory_id,
    )
    return serialize_repair_item(existing or {})


async def best_effort_create_memory_repair_item(**kwargs: Any) -> dict[str, Any] | None:
    try:
        return await create_memory_repair_item(**kwargs)
    except Exception as e:
        logger.warning(
            "memory repair item creation failed: %s",
            e,
            extra={
                "source_type": kwargs.get("source_type"),
                "source_id": kwargs.get("source_id"),
                "memory_id": kwargs.get("memory_id"),
                "error_type": type(e).__name__,
            },
        )
        return None


async def list_memory_repair_items(
    *,
    status: str | None = OPEN,
    source_type: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    workspace_id: str | None = None,
    memory_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    rows = await db.query_raw(
        """
        SELECT *
        FROM memory_repair_items
        WHERE ($1::text IS NULL OR status = $1)
          AND ($2::text IS NULL OR source_type = $2)
          AND ($3::text IS NULL OR user_id = $3)
          AND ($4::text IS NULL OR agent_id = $4)
          AND ($5::text IS NULL OR workspace_id = $5)
          AND ($6::text IS NULL OR memory_id = $6)
        ORDER BY created_at DESC
        LIMIT $7 OFFSET $8
        """,
        None if status in (None, "all") else status,
        source_type,
        user_id,
        agent_id,
        workspace_id,
        memory_id,
        limit,
        offset,
    )
    return [serialize_repair_item(row) for row in rows]


async def get_memory_repair_item(item_id: str) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM memory_repair_items
        WHERE id = $1
        LIMIT 1
        """,
        item_id,
    )
    return serialize_repair_item(rows[0]) if rows else None


async def update_memory_repair_item_status(
    item_id: str,
    *,
    status: str,
    resolution_note: str | None = None,
    resolved_by_id: str | None = None,
) -> dict[str, Any] | None:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    is_closed = status in {RESOLVED, DISMISSED}
    rows = await db.query_raw(
        """
        UPDATE memory_repair_items
        SET status = $2,
            resolution_note = $3,
            resolved_by_id = CASE WHEN $4::boolean THEN $5 ELSE NULL END,
            resolved_at = CASE WHEN $4::boolean THEN $6::timestamp ELSE NULL END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        RETURNING *
        """,
        item_id,
        status,
        resolution_note,
        is_closed,
        resolved_by_id,
        datetime.now(timezone.utc).replace(tzinfo=None),
    )
    return serialize_repair_item(rows[0]) if rows else None


async def _find_existing_open_item(
    *,
    source_type: str,
    source_id: str | None,
    memory_id: str | None,
) -> dict[str, Any] | None:
    if source_id:
        rows = await db.query_raw(
            """
            SELECT *
            FROM memory_repair_items
            WHERE status = 'open'
              AND source_type = $1
              AND source_id = $2
            ORDER BY created_at DESC
            LIMIT 1
            """,
            source_type,
            source_id,
        )
        return rows[0] if rows else None
    if memory_id:
        rows = await db.query_raw(
            """
            SELECT *
            FROM memory_repair_items
            WHERE status = 'open'
              AND source_type = $1
              AND memory_id = $2
            ORDER BY created_at DESC
            LIMIT 1
            """,
            source_type,
            memory_id,
        )
        return rows[0] if rows else None
    return None
