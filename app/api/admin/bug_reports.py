"""Admin-only bug report endpoints.

Admin 测试聊天时给单条 AI 回复打 bug 标签 (e.g. "AI 与用户记忆混淆").
和 trace 系统正交: trace 看"系统跑了什么", bug report 是"人对结果的不满判定".
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from prisma.errors import RecordNotFoundError
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.db import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-api/bug-reports", tags=["admin-bug-reports"])

_INCLUDE_USERS = {"reporter": True, "resolvedBy": True}


class BugReportStatus(str, Enum):
    open = "open"
    resolved = "resolved"


class CreateBugReportRequest(BaseModel):
    message_id: str
    error_types: list[str] = Field(default_factory=list)
    reason: str | None = None


class UpdateBugReportRequest(BaseModel):
    status: BugReportStatus


def _serialize(report) -> dict:
    return {
        "id": report.id,
        "message_id": report.messageId,
        "error_types": list(report.errorTypes or []),
        "reason": report.reason,
        "status": report.status,
        "reporter_id": report.reporterId,
        "reporter_email": report.reporter.email if getattr(report, "reporter", None) else None,
        "resolved_at": report.resolvedAt.isoformat() if report.resolvedAt else None,
        "resolved_by_id": report.resolvedById,
        "resolver_email": report.resolvedBy.email if getattr(report, "resolvedBy", None) else None,
        "created_at": report.createdAt.isoformat(),
    }


@router.post("")
async def create_bug_report(
    payload: CreateBugReportRequest,
    user: dict = Depends(require_admin_jwt),
):
    if not payload.error_types and not (payload.reason or "").strip():
        raise HTTPException(
            status_code=400,
            detail="error_types or reason required",
        )

    msg = await db.message.find_unique(where={"id": payload.message_id})
    if not msg:
        raise HTTPException(status_code=404, detail="message_not_found")

    report = await db.bugreport.create(
        data={
            "messageId": payload.message_id,
            "reporterId": user["sub"],
            "errorTypes": payload.error_types,
            "reason": (payload.reason or "").strip() or None,
        },
        include=_INCLUDE_USERS,
    )
    return _serialize(report)


@router.get("/counts-by-agent")
async def bug_report_counts_by_agent(_: dict = Depends(require_admin_jwt)):
    """Return per-agent bug report counts for the admin agent list badge.

    左外连 conversations 而非按 user — 一条 bug 通过 message → conversation
    锚到 agent. 用户可能跨多个 agent 各自有 bug, 必须按 agent 维度切.
    """
    rows = await db.query_raw(
        """
        SELECT c.agent_id AS agent_id,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE br.status = 'open') AS open_count
        FROM bug_reports br
        JOIN messages m ON m.id = br.message_id
        JOIN conversations c ON c.id = m.conversation_id
        GROUP BY c.agent_id
        """
    )
    return [
        {
            "agent_id": r["agent_id"],
            "total": int(r["total"]),
            "open": int(r["open_count"]),
        }
        for r in rows
    ]


@router.get("/by-conversation/{conversation_id}")
async def list_bug_reports_by_conversation(
    conversation_id: str,
    _: dict = Depends(require_admin_jwt),
):
    """List bug reports for all messages in a conversation, newest first."""
    reports = await db.bugreport.find_many(
        where={"message": {"is": {"conversationId": conversation_id}}},
        include=_INCLUDE_USERS,
        order={"createdAt": "desc"},
    )
    return [_serialize(r) for r in reports]


@router.patch("/{report_id}")
async def update_bug_report(
    report_id: str,
    payload: UpdateBugReportRequest,
    user: dict = Depends(require_admin_jwt),
):
    is_resolved = payload.status == BugReportStatus.resolved
    try:
        report = await db.bugreport.update(
            where={"id": report_id},
            data={
                "status": payload.status.value,
                "resolvedAt": datetime.now(timezone.utc) if is_resolved else None,
                "resolvedById": user["sub"] if is_resolved else None,
            },
            include=_INCLUDE_USERS,
        )
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail="bug_report_not_found")
    return _serialize(report)


@router.delete("/{report_id}")
async def delete_bug_report(
    report_id: str,
    _: dict = Depends(require_admin_jwt),
):
    try:
        await db.bugreport.delete(where={"id": report_id})
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail="bug_report_not_found")
    return {"ok": True}
