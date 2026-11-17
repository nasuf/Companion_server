"""User forgetting by archive: hide the rows and fence their sources atomically.

App bulk delete and workspace wipe keep the text archived for manual recovery,
but they are user forgetting, not housekeeping. The same transaction archives
the rows, writes the audit and calls `fence_forgotten_memories`, so a late or
replayed extraction of the same source messages cannot write the facts back.
Internal archives (consolidation, reminder cancel) do not use this path.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import timedelta

from app.db import db
from app.services.memory.storage.persistence import after_memory_changelog
from app.services.memory.storage.repo import MemoryRecord, invalidate_scope

_TABLES = {"user": "memories_user", "ai": "memories_ai"}


async def archive_forgotten(user_id: str, records: list[MemoryRecord], *, operation: str) -> int:
    """Archive the caller's live records, fence their sources and return the count."""
    groups: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for rec in records:
        if rec.userId == user_id and not rec.isArchived and rec.source in _TABLES:
            groups[(rec.source, rec.workspaceId)].append(rec.id)
    archived: list[tuple[str, str | None]] = []
    for (side, workspace_id), ids in groups.items():
        # A wipe redacts every frozen batch of the scope in one transaction.
        async with db.tx(timeout=timedelta(seconds=15)) as tx:
            rows = await tx.query_raw(
                f"""UPDATE {_TABLES[side]} SET is_archived=true,updated_at=CURRENT_TIMESTAMP
                WHERE id=ANY($1::text[]) AND user_id=$2 AND workspace_id IS NOT DISTINCT FROM $3
                  AND NOT is_archived RETURNING id,content""",
                ids, user_id, workspace_id,
            )
            if not rows:
                continue
            done = [r["id"] for r in rows]
            await tx.execute_raw(
                """INSERT INTO memory_changelogs(id,user_id,workspace_id,memory_id,operation,old_value)
                SELECT unnest($1::text[]),$2,$3,unnest($4::text[]),$5,unnest($6::text[])""",
                [str(uuid.uuid4()) for _ in done], user_id, workspace_id, done, operation,
                [r["content"] for r in rows],
            )
            await tx.query_raw(
                "SELECT 1 AS ok FROM (SELECT fence_forgotten_memories($1,$2,$3::text[])) f",
                side, workspace_id, done,
            )
        archived += [(mid, workspace_id) for mid in done]
        await invalidate_scope(user_id, workspace_id)
    for memory_id, workspace_id in archived:
        await after_memory_changelog(user_id=user_id, memory_id=memory_id,
                                     operation=operation, workspace_id=workspace_id)
    return len(archived)
