"""Time capsule notification scans."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import db
from app.services.notifications.service import notify_capsules_ready

_LOCAL_TZ = timezone(timedelta(hours=8))


def _field(row: Any, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


async def scan_ready_capsule_notifications(now: datetime | None = None) -> int:
    local_date = (now or datetime.now(_LOCAL_TZ)).astimezone(_LOCAL_TZ).date()
    rows = await db.query_raw(
        """
        SELECT
            user_id AS "userId",
            workspace_id AS "workspaceId",
            COUNT(*) AS count
        FROM time_capsules
        WHERE status = 'sealed'
          AND opened_at IS NULL
          AND open_date IS NOT NULL
          AND open_date::date <= $1::date
        GROUP BY user_id, workspace_id
        """,
        local_date,
    )
    count = 0
    for row in rows:
        ready_count = int(_field(row, "count", 0) or 0)
        if ready_count <= 0:
            continue
        await notify_capsules_ready(
            user_id=str(_field(row, "userId")),
            workspace_id=_field(row, "workspaceId"),
            ready_count=ready_count,
            local_date=local_date,
        )
        count += 1
    return count
