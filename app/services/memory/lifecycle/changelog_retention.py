"""Changelog retention: purge stale `access` rows.

`access` is written on every retrieval injection (10-20 rows per chat turn),
so the table grows unbounded. Only two consumers read access rows:

- L2 frequency factor: 1-year sliding window count
- L2 time factor: all-time MAX(created_at)

Deleting access rows older than 13 months keeps both correct in practice —
a memory whose newest access is older than that falls back to createdAt for
the time factor (tf 0.5-0.6 band either way). All non-access operations
(insert / promote / demote / contradiction / hygiene…) are audit history and
are kept forever.
"""

from __future__ import annotations

import logging

from app.db import db

logger = logging.getLogger(__name__)

RETENTION_MONTHS = 13
_BATCH_SIZE = 5000
_MAX_BATCHES_PER_RUN = 20  # cap one run at 100k rows; the weekly cadence catches up


async def purge_stale_access_changelog(
    *,
    retention_months: int = RETENTION_MONTHS,
    batch_size: int = _BATCH_SIZE,
    max_batches: int = _MAX_BATCHES_PER_RUN,
) -> int:
    """Delete old `access` changelog rows in bounded batches.

    Batched via `id IN (SELECT ... LIMIT n)` so a huge backlog never holds a
    long row lock or bloats one transaction. Returns rows deleted.
    """
    total = 0
    for _ in range(max_batches):
        try:
            deleted = await db.execute_raw(
                f"""
                DELETE FROM memory_changelogs
                WHERE id IN (
                    SELECT id FROM memory_changelogs
                    WHERE operation = 'access'
                      AND created_at < NOW() - INTERVAL '{int(retention_months)} months'
                    LIMIT {int(batch_size)}
                )
                """,
            )
        except Exception as e:
            logger.warning(f"access changelog purge failed after {total} rows: {e}")
            return total
        total += int(deleted or 0)
        if not deleted or int(deleted) < batch_size:
            break
    if total:
        logger.info(f"access changelog purge removed {total} rows (> {retention_months} months)")
    return total
