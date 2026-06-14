"""Business-facing notification enqueue helpers."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import Any

from app.db import db
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)


def _field(row: Any, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _trim(text: str, limit: int) -> str:
    value = " ".join((text or "").strip().split())
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 1)]}…"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _raw_timestamp(value: datetime) -> str:
    return _as_utc(value).replace(tzinfo=None).isoformat()


async def enqueue_notification(
    *,
    user_id: str,
    type: str,
    title: str,
    body: str,
    payload: dict[str, Any] | None = None,
    dedupe_key: str,
    agent_id: str | None = None,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    scheduled_for: datetime | None = None,
    dispatch_now: bool = True,
) -> str | None:
    clean_title = _trim(title, 80)
    clean_body = _trim(body, 160)
    if not user_id or not type or not clean_title or not clean_body or not dedupe_key:
        return None
    scheduled = _as_utc(scheduled_for or datetime.now(UTC))
    rows = await db.query_raw(
        """
        INSERT INTO notification_events (
            user_id, agent_id, workspace_id, conversation_id, message_id,
            type, title, body, payload, dedupe_key, scheduled_for, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9::jsonb, $10, $11::timestamp, CURRENT_TIMESTAMP
        )
        ON CONFLICT (user_id, type, dedupe_key) DO NOTHING
        RETURNING id
        """,
        user_id,
        agent_id,
        workspace_id,
        conversation_id,
        message_id,
        type,
        clean_title,
        clean_body,
        json.dumps(payload or {}, ensure_ascii=False),
        dedupe_key,
        _raw_timestamp(scheduled),
    )
    if not rows:
        return None
    event_id = str(_field(rows[0], "id"))
    if dispatch_now and scheduled <= datetime.now(UTC):
        try:
            from app.services.notifications.dispatcher import dispatch_due_notifications

            fire_background(dispatch_due_notifications(limit=10))
        except Exception as e:
            logger.debug(f"[PUSH] dispatch kick skipped: {e}")
    return event_id


async def _conversation_context(conversation_id: str) -> dict[str, Any] | None:
    rows = await db.query_raw(
        """
        SELECT
            c.user_id AS "userId",
            c.agent_id AS "agentId",
            c.workspace_id AS "workspaceId",
            a.name AS "agentName"
        FROM conversations c
        JOIN ai_agents a ON a.id = c.agent_id
        WHERE c.id = $1 AND c.is_deleted = FALSE
        LIMIT 1
        """,
        conversation_id,
    )
    return dict(rows[0]) if rows else None


async def notify_agent_message_created(
    *,
    conversation_id: str,
    message_id: str,
    text: str,
    metadata: dict | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    workspace_id: str | None = None,
    agent_name: str | None = None,
) -> None:
    ctx = None
    if not user_id or not agent_id:
        ctx = await _conversation_context(conversation_id)
        if not ctx:
            return
    user_id = user_id or str(ctx["userId"])
    agent_id = agent_id or str(ctx["agentId"])
    workspace_id = workspace_id if workspace_id is not None else (str(ctx["workspaceId"]) if ctx and ctx.get("workspaceId") else None)
    agent_name = agent_name or (str(ctx["agentName"]) if ctx else "伴生")
    origin = "proactive" if (metadata or {}).get("proactive") else "reply"
    payload = {
        "type": "agent_message",
        "route": "chat",
        "conversation_id": conversation_id,
        "workspace_id": workspace_id,
        "agent_id": agent_id,
        "message_id": message_id,
        "origin": origin,
        "trigger_type": (metadata or {}).get("trigger_type"),
    }
    await enqueue_notification(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        message_id=message_id,
        type="agent_message",
        title=agent_name or "伴生",
        body=_trim(text, 120) or "你收到了一条新消息",
        payload=payload,
        dedupe_key=message_id,
    )


async def notify_achievement_unlocked(
    *,
    user_id: str,
    agent_id: str,
    achievement_id: int,
    title: str,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
) -> None:
    await enqueue_notification(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        type="achievement_unlocked",
        title="成就达成",
        body=f"你解锁了「{_trim(title, 40)}」",
        payload={
            "type": "achievement_unlocked",
            "route": "achievement",
            "achievement_id": achievement_id,
            "agent_id": agent_id,
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
        },
        dedupe_key=f"{agent_id}:{achievement_id}",
    )


async def notify_checkin_reminder(
    *,
    user_id: str,
    agent_id: str,
    trigger_id: str,
    summary: str,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    memory_id: str | None = None,
) -> None:
    await enqueue_notification(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        type="checkin_reminder",
        title="伴生打卡提醒",
        body=f"该完成「{_trim(summary, 60)}」啦",
        payload={
            "type": "checkin_reminder",
            "route": "checkin",
            "trigger_id": trigger_id,
            "memory_id": memory_id,
            "agent_id": agent_id,
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
        },
        dedupe_key=trigger_id,
    )


async def notify_capsules_ready(
    *,
    user_id: str,
    ready_count: int,
    local_date: date,
    workspace_id: str | None = None,
) -> None:
    if ready_count <= 0:
        return
    body = "今天有一个时间胶囊可以开启了" if ready_count == 1 else f"今天有 {ready_count} 个时间胶囊可以开启了"
    await enqueue_notification(
        user_id=user_id,
        workspace_id=workspace_id,
        type="capsule_ready",
        title="时间胶囊可以开启了",
        body=body,
        payload={
            "type": "capsule_ready",
            "route": "capsules",
            "state": "ready",
            "workspace_id": workspace_id,
            "local_date": local_date.isoformat(),
            "ready_count": ready_count,
        },
        dedupe_key=f"{local_date.isoformat()}:{workspace_id or 'all'}",
    )
