"""Memory access logging.

Records which memories were retrieved and injected into the chat prompt.
Used by L2 dynamics (§1.5.2) to compute the frequency_factor:
  "最近1年内被调用的次数" = count of 'access' operations in changelog.

Also touches updatedAt so admin views reflect real usage (the L2 time factor
itself reads the changelog, not updatedAt).

Writes are batched: one multi-VALUES INSERT for the changelog rows and one
UPDATE per table for the timestamp touch — a heavy chat turn injects 10-20
memories, and the previous per-row inserts made the hot path chatty.
"""

from __future__ import annotations

import logging
import uuid

from app.db import db
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)


async def log_memory_access(
    user_id: str,
    memory_ids: list[str],
    workspace_id: str | None = None,
) -> None:
    """Record that these memories were retrieved into a prompt.

    Single multi-VALUES changelog insert + one updatedAt touch per table.
    Runs in background (fire-and-forget from orchestrator). Access rows skip
    the quality-state / achievement hooks by design (same as before — those
    hooks always ignored operation='access').
    """
    if not memory_ids:
        return

    try:
        workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    except Exception as e:
        logger.debug(f"access log workspace resolve failed: {e}")

    try:
        placeholders = ",".join(
            f"(${i * 4 + 1}, ${i * 4 + 2}, ${i * 4 + 3}, ${i * 4 + 4}, 'access')"
            for i in range(len(memory_ids))
        )
        args: list = []
        for mid in memory_ids:
            args.extend((str(uuid.uuid4()), user_id, workspace_id, mid))
        await db.execute_raw(
            f"INSERT INTO memory_changelogs (id, user_id, workspace_id, memory_id, operation) "
            f"VALUES {placeholders}",
            *args,
        )
    except Exception as e:
        logger.debug(f"batch access log write failed: {e}")

    # Touch updatedAt in both user and ai tables for admin display freshness.
    # IDs are unique across tables so the WHERE misses gracefully.
    for table in ("memories_user", "memories_ai"):
        try:
            await db.execute_raw(
                f"UPDATE {table} SET updated_at = CURRENT_TIMESTAMP WHERE id = ANY($1::text[])",
                memory_ids,
            )
        except Exception as e:
            logger.debug(f"Memory access touch on {table} failed: {e}")
