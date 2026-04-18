"""Reflection service.

Periodic self-reflection: memory merge, preference update,
relationship update, personality observation.
"""

import logging

from app.services.memory.lifecycle.consolidation import consolidate_daily, consolidate_weekly
from app.services.memory.storage.entity_repo import consolidate_entities_globally
from app.services.memory.lifecycle.decay import decay_importance

logger = logging.getLogger(__name__)


async def run_daily_reflection():
    """Run daily reflection tasks."""
    logger.info("Starting daily reflection")

    # 1. Decay importance
    await decay_importance()

    # 2. Consolidate old L3 memories
    await consolidate_daily()

    logger.info("Daily reflection complete")


async def run_weekly_reflection():
    """Run weekly reflection tasks."""
    logger.info("Starting weekly reflection")

    # 1. Consolidate L2 patterns to L1
    await consolidate_weekly()

    # 2. Entity knowledge-layer hygiene: archive stale entities, merge
    # near-duplicate names (handles "妈妈/我妈/妈咪" drift that accumulates
    # over months of chat).
    try:
        stats = await consolidate_entities_globally()
        logger.info(
            f"Entity consolidation: archived={stats['archived']}, merged={stats['merged']}"
        )
    except Exception as e:
        logger.warning(f"Entity consolidation failed: {e}")

    logger.info("Weekly reflection complete")
