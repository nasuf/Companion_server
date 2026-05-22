"""Structured operational metrics for P3 dashboards."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.db import db

logger = logging.getLogger(__name__)


async def record_reply_operational_metrics(
    *,
    message_id: str,
    conversation_id: str,
    metadata: dict[str, Any],
) -> None:
    """Persist visible memory use and crisis counters from reply metadata.

    This is best-effort and intentionally derived from existing metadata so it
    does not add extra LLM calls to the chat path.
    """
    try:
        conv = await db.conversation.find_unique(where={"id": conversation_id})
        agent_id = getattr(conv, "agentId", None) if conv else None
        user_id = getattr(conv, "userId", None) if conv else None
        workspace_id = getattr(conv, "workspaceId", None) if conv else None
        await record_visible_use_event(
            message_id=message_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_id=user_id,
            workspace_id=workspace_id,
            metadata=metadata,
        )
        await record_crisis_event(
            message_id=message_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_id=user_id,
            workspace_id=workspace_id,
            metadata=metadata,
        )
    except Exception as e:
        logger.debug("reply operational metric persistence skipped: %s", e)


async def record_visible_use_event(
    *,
    message_id: str,
    conversation_id: str,
    agent_id: str | None,
    user_id: str | None,
    workspace_id: str | None,
    metadata: dict[str, Any],
) -> None:
    analysis = metadata.get("memory_retrieval_analysis")
    if not isinstance(analysis, dict):
        return
    quality_metrics = analysis.get("quality_metrics")
    if not isinstance(quality_metrics, dict):
        quality_metrics = {}
    selected_count = _safe_int(analysis.get("selected_count"))
    likely_used_count = _safe_int(analysis.get("likely_used_count"))
    warning_count = _safe_int(quality_metrics.get("warning_count"))
    warnings = analysis.get("warnings") if isinstance(analysis.get("warnings"), list) else []
    unsupported_count = sum(1 for item in warnings if isinstance(item, dict) and item.get("code") == "no_visible_memory_use")
    memory_ids = [
        str(item.get("id"))
        for item in analysis.get("items", [])
        if isinstance(item, dict) and item.get("id")
    ][:50]
    await db.query_raw(
        """
        INSERT INTO memory_visible_use_events (
            id, message_id, conversation_id, agent_id, user_id, workspace_id,
            trace_id, method, selected_count, likely_used_count,
            likely_unused_count, visible_use_rate, unsupported_reference_count,
            warning_count, has_prompt_dilution, has_final_gate_drop,
            memory_ids, payload
        )
        VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13,
            $14, $15, $16,
            $17::text[], $18::jsonb
        )
        """,
        str(uuid.uuid4()),
        message_id,
        conversation_id,
        agent_id,
        user_id,
        workspace_id,
        metadata.get("trace_id"),
        str(analysis.get("method") or "lexical_overlap_v1"),
        selected_count,
        likely_used_count,
        _safe_int(analysis.get("likely_unused_count")),
        _safe_float(quality_metrics.get("visible_use_rate")),
        unsupported_count,
        warning_count,
        bool(quality_metrics.get("has_prompt_dilution")),
        bool(quality_metrics.get("has_final_gate_drop")),
        memory_ids,
        json.dumps(analysis, ensure_ascii=False),
    )


async def record_crisis_event(
    *,
    message_id: str,
    conversation_id: str,
    agent_id: str | None,
    user_id: str | None,
    workspace_id: str | None,
    metadata: dict[str, Any],
) -> None:
    diagnostics = metadata.get("response_diagnostics")
    if not isinstance(diagnostics, dict):
        return
    status = str(diagnostics.get("crisis_guard_status") or "none")
    if status == "none":
        return
    await db.query_raw(
        """
        INSERT INTO crisis_events (
            id, message_id, conversation_id, agent_id, user_id, workspace_id,
            trace_id, status, category, severity, handler_path,
            aftercare_triggered, safety_check_mode, semantic_checked,
            semantic_detected, payload
        )
        VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10, $11,
            $12, $13, $14,
            $15, $16::jsonb
        )
        """,
        str(uuid.uuid4()),
        message_id,
        conversation_id,
        agent_id,
        user_id,
        workspace_id,
        metadata.get("trace_id"),
        status,
        _crisis_category(status),
        _crisis_severity(status, diagnostics),
        "crisis_followup" if diagnostics.get("crisis_followup_check_mode") else "crisis",
        bool(status in {"direct_crisis", "semantic_crisis", "crisis_followup"}),
        diagnostics.get("crisis_followup_check_mode"),
        bool(diagnostics.get("crisis_semantic_checked")),
        bool(diagnostics.get("crisis_semantic_detected")),
        json.dumps(diagnostics, ensure_ascii=False),
    )


async def summarize_visible_use(
    *,
    start: datetime | None,
    agent_id: str | None = None,
    user_id: str | None = None,
    db_client: Any | None = None,
) -> dict[str, Any]:
    client = db_client or db
    where, params = _where(start=start, agent_id=agent_id, user_id=user_id, created_expr="created_at")
    try:
        rows = await client.query_raw(
            f"""
            SELECT
                COUNT(*)::int AS event_count,
                COALESCE(SUM(selected_count), 0)::int AS injected_count,
                COALESCE(SUM(likely_used_count), 0)::int AS visibly_used_count,
                COALESCE(SUM(unsupported_reference_count), 0)::int AS unsupported_reference_count,
                COALESCE(AVG(visible_use_rate), 0)::float AS avg_visible_use_rate,
                COALESCE(SUM(warning_count), 0)::int AS warning_count
            FROM memory_visible_use_events
            WHERE {where}
            """,
            *params,
        )
    except Exception:
        rows = []
    row = rows[0] if rows else {}
    return {
        "event_count": _safe_int(row.get("event_count")),
        "injected_count": _safe_int(row.get("injected_count")),
        "visibly_used_count": _safe_int(row.get("visibly_used_count")),
        "unsupported_reference_count": _safe_int(row.get("unsupported_reference_count")),
        "avg_visible_use_rate": round(_safe_float(row.get("avg_visible_use_rate")), 4),
        "warning_count": _safe_int(row.get("warning_count")),
    }


async def summarize_crisis_events(
    *,
    start: datetime | None,
    agent_id: str | None = None,
    user_id: str | None = None,
    db_client: Any | None = None,
) -> dict[str, Any]:
    client = db_client or db
    where, params = _where(start=start, agent_id=agent_id, user_id=user_id, created_expr="created_at")
    try:
        rows = await client.query_raw(
            f"""
            SELECT status, COUNT(*)::int AS count
            FROM crisis_events
            WHERE {where}
            GROUP BY status
            ORDER BY count DESC
            """,
            *params,
        )
        severity_rows = await client.query_raw(
            f"""
            SELECT COALESCE(severity, 'unknown') AS severity, COUNT(*)::int AS count
            FROM crisis_events
            WHERE {where}
            GROUP BY COALESCE(severity, 'unknown')
            ORDER BY count DESC
            """,
            *params,
        )
    except Exception:
        rows = []
        severity_rows = []
    by_status = {str(row.get("status") or "unknown"): _safe_int(row.get("count")) for row in rows}
    by_severity = {str(row.get("severity") or "unknown"): _safe_int(row.get("count")) for row in severity_rows}
    return {
        "created_count": sum(by_status.values()),
        "by_status": by_status,
        "by_severity": by_severity,
    }


def _where(
    *,
    start: datetime | None,
    agent_id: str | None,
    user_id: str | None,
    created_expr: str,
) -> tuple[str, list[Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if start is not None:
        params.append(start.replace(tzinfo=None).isoformat())
        clauses.append(f"{created_expr} >= ${len(params)}::timestamp")
    if agent_id:
        params.append(agent_id)
        clauses.append(f"agent_id = ${len(params)}")
    if user_id:
        params.append(user_id)
        clauses.append(f"user_id = ${len(params)}")
    return " AND ".join(clauses), params


def _crisis_category(status: str) -> str:
    if status in {"direct_crisis", "semantic_crisis"}:
        return "detected_crisis"
    if status == "crisis_followup":
        return "aftercare"
    if status in {"release_pending", "released"}:
        return "release"
    return "other"


def _crisis_severity(status: str, diagnostics: dict[str, Any]) -> str:
    if status == "direct_crisis":
        return "high"
    if status == "semantic_crisis":
        return "medium"
    if diagnostics.get("crisis_followup_check_mode") in {"soft", "annoyed"}:
        return "medium"
    return "low"


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0
