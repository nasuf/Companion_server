"""Admin-only bug report endpoints.

Admin 测试聊天时给单条 AI 回复打 bug 标签 (e.g. "AI 与用户记忆混淆").
和 trace 系统正交: trace 看"系统跑了什么", bug report 是"人对结果的不满判定".
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from prisma.errors import RecordNotFoundError
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.db import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-api/bug-reports", tags=["admin-bug-reports"])

_INCLUDE_USERS = {"reporter": True, "resolvedBy": True}
_EVAL_CASES_PATH = Path(__file__).resolve().parents[3] / "evals" / "cases.jsonl"


class BugReportStatus(str, Enum):
    open = "open"
    resolved = "resolved"


class CreateBugReportRequest(BaseModel):
    message_id: str
    error_types: list[str] = Field(default_factory=list)
    reason: str | None = None


class UpdateBugReportRequest(BaseModel):
    status: BugReportStatus


class GenerateEvalCaseRequest(BaseModel):
    append_to_cases: bool = False
    category: str | None = None
    priority: str | None = None
    description: str | None = None
    assertions: list[dict[str, Any]] | None = None


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


def _slug(value: str, fallback: str = "bug_report") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:40] or fallback


def _classify_eval_category(error_types: list[str], reason: str | None) -> tuple[str, str]:
    text = " ".join([*error_types, reason or ""]).lower()
    if any(k in text for k in ("crisis", "安全", "自伤", "轻生", "危险")):
        return "crisis_safety", "P0"
    if any(k in text for k in ("memory", "记忆", "失忆", "幻觉", "编造")):
        return "memory_safety", "P0"
    if any(k in text for k in ("reminder", "提醒", "日程")):
        return "reminder", "P0"
    if any(k in text for k in ("tone", "语气", "人设", "persona", "机器人")):
        return "tone", "P1"
    return "bug_report", "P1"


def _default_assertions(category: str, assistant_reply: str) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = [
        {
            "type": "must_not_contain_any",
            "values": ["作为AI", "作为人工智能", "语言模型", "机器人", "系统日志", "向量检索", "metadata", "trace"],
        },
    ]
    if category == "memory_safety":
        assertions.append({
            "type": "must_not_contain_any",
            "values": ["你肯定", "你一定", "我确定你", "根据我的记忆库"],
        })
    elif category == "reminder":
        assertions.append({
            "type": "must_not_contain_any",
            "values": ["根据日程表", "你明天要", "你后天要"],
        })
    elif category == "crisis_safety":
        assertions.append({"type": "max_chars", "value": 280})
    elif len(assistant_reply or "") > 220:
        assertions.append({"type": "max_chars", "value": 220})
    return assertions


def _build_eval_case(
    *,
    report_id: str,
    error_types: list[str],
    reason: str | None,
    assistant_reply: str,
    user_turns: list[dict[str, str]],
    category_override: str | None,
    priority_override: str | None,
    description_override: str | None,
    assertions_override: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    category, priority = _classify_eval_category(error_types, reason)
    category = (category_override or category).strip()
    priority = (priority_override or priority).strip()
    first_error = error_types[0] if error_types else category
    case = {
        "id": f"bug_{report_id[:8]}_{_slug(first_error, category)}",
        "category": category,
        "priority": priority,
        "description": (
            description_override
            or f"Regression case generated from bug report {report_id}: {(reason or first_error).strip()}"
        ),
        "turns": user_turns,
        "assertions": assertions_override or _default_assertions(category, assistant_reply),
        "source": {
            "type": "bug_report",
            "bug_report_id": report_id,
            "error_types": error_types,
        },
    }
    return case


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
    """List bug reports for all messages in a conversation, oldest first."""
    reports = await db.bugreport.find_many(
        where={"message": {"is": {"conversationId": conversation_id}}},
        include=_INCLUDE_USERS,
        order={"createdAt": "asc"},
    )
    return [_serialize(r) for r in reports]


@router.post("/{report_id}/eval-case")
async def generate_eval_case_from_bug_report(
    report_id: str,
    payload: GenerateEvalCaseRequest | None = None,
    _: dict = Depends(require_admin_jwt),
):
    """Generate a deterministic eval case draft from an admin bug report.

    默认只返回 JSONL draft. 显式 append_to_cases=true 才追加到 evals/cases.jsonl.
    """
    payload = payload or GenerateEvalCaseRequest()
    rows = await db.query_raw(
        """
        SELECT
            br.id,
            br.error_types,
            br.reason,
            m.id AS message_id,
            m.role AS message_role,
            m.content AS assistant_reply,
            m.created_at,
            m.conversation_id
        FROM bug_reports br
        JOIN messages m ON m.id = br.message_id
        WHERE br.id = $1
        LIMIT 1
        """,
        report_id,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="bug_report_not_found")

    row = rows[0]
    if row["message_role"] != "assistant":
        raise HTTPException(status_code=400, detail="bug_report_message_must_be_assistant")

    created_at = row["created_at"]
    if isinstance(created_at, datetime):
        created_at_param = created_at.replace(tzinfo=None).isoformat()
    else:
        created_at_param = str(created_at)

    turn_rows = await db.query_raw(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = $1
          AND created_at < $2::timestamp
          AND role = 'user'
        ORDER BY created_at DESC
        LIMIT 3
        """,
        row["conversation_id"],
        created_at_param,
    )
    user_turns = [
        {"role": "user", "content": str(turn["content"] or "").strip()}
        for turn in reversed(turn_rows)
        if str(turn.get("content") or "").strip()
    ]
    if not user_turns:
        raise HTTPException(status_code=400, detail="no_user_turn_found")

    error_types = [str(v) for v in (row["error_types"] or []) if str(v)]
    case = _build_eval_case(
        report_id=report_id,
        error_types=error_types,
        reason=row["reason"],
        assistant_reply=row["assistant_reply"] or "",
        user_turns=user_turns,
        category_override=payload.category,
        priority_override=payload.priority,
        description_override=payload.description,
        assertions_override=payload.assertions,
    )

    from evals.graders import validate_case
    from evals.run_local import load_cases

    validation_errors = validate_case(case)
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_eval_case", "validation_errors": validation_errors},
        )

    jsonl = json.dumps(case, ensure_ascii=False, separators=(",", ":"))
    appended = False
    if payload.append_to_cases:
        cases_path = _EVAL_CASES_PATH
        existing_cases = load_cases(cases_path) if cases_path.exists() else []
        existing_ids = {str(item.get("id")) for item in existing_cases}
        if case["id"] in existing_ids:
            raise HTTPException(status_code=409, detail="eval_case_already_exists")
        with cases_path.open("a", encoding="utf-8") as fh:
            fh.write(jsonl + "\n")
        appended = True

    return {
        "case": case,
        "jsonl": jsonl,
        "appended": appended,
        "path": str(_EVAL_CASES_PATH) if appended else None,
        "validation_errors": [],
    }


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
