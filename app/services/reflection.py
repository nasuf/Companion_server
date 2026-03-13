"""Reflection service.

Periodic self-reflection: memory merge, preference update,
relationship update, personality observation.
"""

import logging

from app.services.memory.consolidation import consolidate_daily, consolidate_weekly
from app.services.memory.lifecycle import decay_importance

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

    logger.info("Weekly reflection complete")
