"""Recheck recalled memories right before the final prompt is assembled.

Recall can be served from the 5-minute retrieval cache or finish while the
user forgets a memory in another request. Candidate-time filters cannot see
those changes, so the last step keeps only rows that still exist, are live
and belong to the caller's scope. A failed check drops every checkable row:
one turn without memories is better than replaying a forgotten fact.
"""

from __future__ import annotations

import logging

from app.db import db

logger = logging.getLogger(__name__)

_TABLES = {"user": "memories_user", "ai": "memories_ai"}


async def keep_live_memories(memories, *, user_id: str, workspace_id: str | None,
                             diagnostics: dict | None = None):
    """Return `memories` without rows deleted, archived or moved since recall.

    Items without an id predate structured caching and have no row to check;
    they are kept as is.
    """
    if not memories:
        return memories
    def _key(m) -> tuple[str, str]:
        return ("ai" if getattr(m, "source", "user") == "ai" else "user", getattr(m, "id", "") or "")

    wanted: dict[str, set[str]] = {}
    for m in memories:
        side, mid = _key(m)
        if mid:
            wanted.setdefault(side, set()).add(mid)
    if not wanted:
        return memories
    live: set[tuple[str, str]] = set()
    try:
        for side, ids in wanted.items():
            rows = await db.query_raw(
                f"""SELECT id FROM {_TABLES[side]} WHERE id=ANY($1::text[]) AND user_id=$2
                AND workspace_id IS NOT DISTINCT FROM $3 AND NOT is_archived""",
                sorted(ids), user_id, workspace_id,
            )
            live.update((side, r["id"]) for r in rows)
        status = "checked"
    except Exception as exc:
        logger.warning("Memory live check failed: %s", type(exc).__name__)
        status = "failed_closed"
    kept = [m for m in memories if not _key(m)[1] or _key(m) in live]
    if diagnostics is not None:
        diagnostics["memory_live_check"] = {
            "status": status,
            "checked": sum(len(ids) for ids in wanted.values()),
            "dropped": len(memories) - len(kept),
        }
    return kept
