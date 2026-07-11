"""Achievement helpers for chat coverage across AI schedule states."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.db import db
from app.services.achievements.repository import record_event
from app.services.achievements.utils import _field, _local
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status

logger = logging.getLogger(__name__)

_STATUS_BUCKETS = {"idle", "busy", "sleep"}


@dataclass(frozen=True, slots=True)
class ScheduleStatusObservation:
    bucket: str
    period_key: str


def _status_bucket(status: str | None) -> str | None:
    if status == "very_busy":
        return "busy"
    if status in _STATUS_BUCKETS:
        return status
    return None


def _matching_schedule_slot(schedule: list[dict], local_at: datetime) -> dict | None:
    current_time = local_at.strftime("%H:%M")
    for slot in schedule:
        start = str(slot.get("start") or "00:00")
        end = str(slot.get("end") or "23:59")
        if start > end:
            if current_time >= start or current_time < end:
                return slot
        elif start <= current_time < end:
            return slot
    return None


def _schedule_period_key(slot: dict | None, local_at: datetime, bucket: str) -> str:
    if not slot:
        return f"{local_at.date().isoformat()}:default:{bucket}"
    start = str(slot.get("start") or "00:00")
    end = str(slot.get("end") or "23:59")
    period_date = local_at.date()
    if start > end and local_at.strftime("%H:%M") < end:
        period_date -= timedelta(days=1)
    return f"{period_date.isoformat()}:{start}-{end}:{bucket}"


async def record_schedule_status_chat(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    message_id: str,
    occurred_at: datetime,
) -> ScheduleStatusObservation | None:
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
        slot = _matching_schedule_slot(schedule, local_at)
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
    return ScheduleStatusObservation(
        bucket=bucket,
        period_key=_schedule_period_key(slot, local_at, bucket),
    )


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
            SELECT COUNT(DISTINCT e.value_text) AS count
            FROM achievement_events e
            LEFT JOIN conversations c ON c.id = e.conversation_id
            WHERE e.user_id = $1
              AND e.agent_id = $2
              AND e.event_type = 'schedule_status_chat_day'
              AND e.source_id LIKE $3
              AND e.value_text IN ('idle', 'busy', 'sleep')
              AND (e.conversation_id IS NULL OR c.is_deleted = FALSE)
            """,
            user_id,
            agent_id,
            f"schedule_status_chat_day:{local_date}:%",
        )
        if int(_field(rows[0], "count", 0)) < 3:
            return False
    return True
