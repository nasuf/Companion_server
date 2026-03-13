"""Memory lifecycle manager.

Handles importance decay and archiving.
importance(t) = importance0 * e^(-0.002t)  (t in days)
Archive if importance < 0.1
"""

import logging

from app.db import db

logger = logging.getLogger(__name__)

DECAY_LAMBDA = 0.002
ARCHIVE_THRESHOLD = 0.1


async def decay_importance():
    """Apply importance decay to all active memories using bulk SQL."""
    # Single SQL to update importance and archive in one pass
    result = await db.execute_raw(
        f"""
        UPDATE memories
        SET importance = importance * exp(-{DECAY_LAMBDA} * EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400)
        WHERE is_archived = false
        """
    )
    logger.info(f"Decay applied to {result} memories")

    # Archive memories below threshold
    archived = await db.execute_raw(
        f"""
        UPDATE memories
        SET is_archived = true
        WHERE is_archived = false AND importance < {ARCHIVE_THRESHOLD}
        """
    )
    logger.info(f"Archived {archived} memories below threshold {ARCHIVE_THRESHOLD}")
