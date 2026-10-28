"""Admin view: is the activity copy users see reviewed model text or a fallback?

A fallback is silent by design: the page still reads fine, so nobody notices
that 14 of 20 recommendations showed templates until someone queries the
database by hand. Every status here is the one generation recorded at the time.
Rows created before a status existed are reported as ``not_recorded``, never as
success or as zero.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from app.db import db

MAX_WINDOW_DAYS = 30
ROW_LIMIT = 500
RECENT_LIMIT = 20
NOT_RECORDED = "not_recorded"

# field → (status column, issues column) in the query below.
FIELDS = {
    "card_copy": ("card_copy", "card_copy_issues"),
    "detail": ("detail", "detail_issues"),
    "message": ("message", "message_issues"),
    "invite": ("invite", "invite_issues"),
}

_QUERY = """
SELECT a.id, a.created_at, a.title, a.source, a.status,
       a.discovery_metadata->>'card_copy_status' AS card_copy,
       a.discovery_metadata->'card_copy_issues' AS card_copy_issues,
       a.discovery_metadata->>'copy_status' AS detail,
       a.discovery_metadata->'copy_issues' AS detail_issues,
       a.discovery_metadata->>'copy_section_count' AS section_count,
       a.discovery_metadata->'copy_kept_sections' AS kept_sections,
       a.discovery_metadata->'copy_dropped_sections' AS dropped_sections,
       a.discovery_metadata->'visit_timing' AS visit_timing,
       a.discovery_metadata->>'recommendation_message_status' AS message,
       a.discovery_metadata->'recommendation_message_issues' AS message_issues,
       a.discovery_metadata->>'recommendation_relevance_status' AS relevance,
       a.discovery_metadata->>'invite_message_status' AS invite,
       a.discovery_metadata->'invite_message_issues' AS invite_issues
FROM offline_activity_recommendations a
WHERE a.created_at >= now() - make_interval(days => $1::int)
ORDER BY a.created_at DESC, a.id DESC
LIMIT $2::int
"""


def _issues(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, list):
        return []
    return [item[:120] for item in value if isinstance(item, str) and item.strip()][:3]


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def _section_counts(row: dict) -> dict | None:
    try:
        count = int(row.get("section_count"))
        kept, dropped = row.get("kept_sections"), row.get("dropped_sections")
        kept = json.loads(kept) if isinstance(kept, str) else kept
        dropped = json.loads(dropped) if isinstance(dropped, str) else dropped
        if not 1 <= count <= 8 or not isinstance(kept, list) or not isinstance(dropped, list):
            return None
        values = kept + dropped
        if any(type(i) is not int for i in values) or len(values) != count or set(values) != set(range(count)):
            return None
        return {"total": count, "kept": len(kept), "dropped": len(dropped)}
    except (TypeError, ValueError):
        return None


def _visit_status(value: Any) -> dict | None:
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return None
    if not isinstance(value, dict) or value.get("status") not in {"open", "closed", "unknown", "not_applicable"}:
        return None
    return {key: str(value.get(key) or "")[:60] for key in ("status", "reason", "assessed_at")}


async def collect_copy_quality(days: int) -> dict[str, Any]:
    window = max(1, min(MAX_WINDOW_DAYS, int(days or 7)))
    rows = list(await db.query_raw(_QUERY, window, ROW_LIMIT + 1) or [])
    return summarize_copy_rows(rows, window=window, now=datetime.now(UTC))


def summarize_copy_rows(rows, *, window, now, row_limit=ROW_LIMIT):
    truncated = len(rows) > row_limit
    rows = rows[:row_limit]
    counts = {field: Counter() for field in FIELDS}
    for row in rows:
        for field, (status_column, _) in FIELDS.items():
            counts[field][row.get(status_column) or NOT_RECORDED] += 1
    recent = [
        {
            "id": row.get("id"),
            "created_at": _iso(row.get("created_at")),
            "title": str(row.get("title") or "")[:60],
            "source": row.get("source"),
            "status": row.get("status"),
            "relevance": row.get("relevance") or NOT_RECORDED,
            "detail_sections": _section_counts(row),
            "visit_timing": _visit_status(row.get("visit_timing")),
            **{field: row.get(status_column) or NOT_RECORDED for field, (status_column, _) in FIELDS.items()},
            "issues": {field: _issues(row.get(issues_column)) for field, (_, issues_column) in FIELDS.items()},
        }
        for row in rows[:RECENT_LIMIT]
    ]
    return {
        "sampled_at": now.isoformat(),
        "window_days": window,
        "row_limit": row_limit,
        "truncated": truncated,
        "total": len(rows),
        "fields": {field: dict(counter) for field, counter in counts.items()},
        "recent": recent,
    }
