"""Weekly reflection service.

Spec §1.5 的记忆衰减/升降级由 [l2_dynamics.py:run_l2_adjustment] 独立 cron 完成。
本模块只保留每周一次的 entity graph 清理（工程扩展，非 spec）。
"""

import logging

from app.services.memory.storage.entity_repo import consolidate_entities_globally

logger = logging.getLogger(__name__)


async def run_weekly_reflection():
    """Weekly entity knowledge-layer hygiene.

    Archive stale entities, merge near-duplicate names (handles "妈妈/我妈/妈咪"
    drift that accumulates over months of chat).
    """
    logger.info("Starting weekly reflection")
    try:
        stats = await consolidate_entities_globally()
        logger.info(
            f"Entity consolidation: archived={stats['archived']}, merged={stats['merged']}"
        )
    except Exception as e:
        logger.warning(f"Entity consolidation failed: {e}")
    logger.info("Weekly reflection complete")
