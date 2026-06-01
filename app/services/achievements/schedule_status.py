"""Achievement helpers for chat coverage across AI schedule states."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from app.db import db
from app.services.achievements.repository import record_event
from app.services.achievements.utils import _field, _local
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status

logger = logging.getLogger(__name__)

_STATUS_BUCKETS = {"idle", "busy", "sleep"}


def _status_bucket(status: str | None) -> str | None:
    if status == "very_busy":
        return "busy"
    if status in _STATUS_BUCKETS:
        return status
    return None


async def record_schedule_status_chat(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    message_id: str,
    occurred_at: datetime,
) -> str | None:
    """Record which AI schedule state the user chatted in, scoped by local day."""
    local_at = _local(occurred_at)
    try:
        schedule = await get_cached_schedule(agent_id, local_at)
        if not schedule:
            return None

        status = get_current_status(schedule, local_at)
        bucket = _status_bucket(str(status.get("status") or ""))
        if not bucket:
            return None
    except Exception as e:
        logger.debug(f"[ACH] schedule status skipped agent_id={agent_id}: {e}")
        return None

    local_date = local_at.date().isoformat()
    await record_event(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        event_type="schedule_status_chat_day",
        source_id=f"schedule_status_chat_day:{local_date}:{bucket}",
        value_text=bucket,
        metadata={
            "message_id": message_id,
            "activity": status.get("activity") or status.get("event") or "",
            "status": status.get("status") or "",
        },
        occurred_at=occurred_at,
    )
    return bucket


async def has_schedule_status_streak(
    *,
    user_id: str,
    agent_id: str,
    local_day: datetime,
    days: int,
) -> bool:
    """Return true if each day in the window has idle, busy, and sleep chat."""
    for offset in range(days):
        local_date = (local_day.date() - timedelta(days=offset)).isoformat()
        rows = await db.query_raw(
            """
            SELECT COUNT(DISTINCT value_text) AS count
            FROM achievement_events
            WHERE user_id = $1
              AND agent_id = $2
              AND event_type = 'schedule_status_chat_day'
              AND source_id LIKE $3
              AND value_text IN ('idle', 'busy', 'sleep')
            """,
            user_id,
            agent_id,
            f"schedule_status_chat_day:{local_date}:%",
        )
        if int(_field(rows[0], "count", 0)) < 3:
            return False
    return True
