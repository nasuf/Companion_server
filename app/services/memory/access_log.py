"""Memory access logging.

Records which memories were retrieved and injected into the chat prompt.
Used by L2 dynamics (§1.5.2) to compute the frequency_factor:
  "最近1年内被调用的次数" = count of 'access' operations in changelog.

Also touches updatedAt so the time_factor tracks real usage, not just edits.
"""

from __future__ import annotations

import logging

from app.db import db
from app.services.memory.storage import log_memory_changelog

logger = logging.getLogger(__name__)


async def log_memory_access(
    user_id: str,
    memory_ids: list[str],
    workspace_id: str | None = None,
) -> None:
    """Record that these memories were retrieved into a prompt.

    Bulk operation — one changelog entry + one updatedAt touch per memory.
    Runs in background (fire-and-forget from orchestrator).
    """
    if not memory_ids:
        return

    for mid in memory_ids:
        try:
            await log_memory_changelog(
                user_id=user_id,
                memory_id=mid,
                operation="access",
                workspace_id=workspace_id,
            )
        except Exception as e:
            logger.debug(f"access log write failed for {mid}: {e}")

    # Touch updatedAt in both user and ai tables so L2 time_factor reflects
    # last access. IDs are unique across tables so the WHERE misses gracefully.
    for table in ("memories_user", "memories_ai"):
        try:
            await db.execute_raw(
                f'UPDATE "{table}" SET "updatedAt" = CURRENT_TIMESTAMP WHERE "id" = ANY($1::text[])',
                memory_ids,
            )
        except Exception as e:
            logger.debug(f"Memory access touch on {table} failed: {e}")
