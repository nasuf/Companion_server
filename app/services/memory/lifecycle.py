"""Memory lifecycle manager.

Handles importance decay (tiered time table), dynamic weight, mention tracking, and archiving.
"""

import logging

from app.db import db

logger = logging.getLogger(__name__)

ARCHIVE_THRESHOLD = 0.1

# 分层时间衰减表
# (months_start, months_end, factor)
TIME_DECAY_TABLE = [
    (0, 1, 1.0),      # <1月
    (1, 3, 0.9),      # 1-3月
    (3, 6, 0.8),      # 3-6月
    (6, 12, 0.7),     # 6-12月
    (12, 24, 0.6),    # 1-2年
    (24, 9999, 0.5),  # >2年
]

# 提及频次加权表
FREQ_FACTOR_TABLE = [
    (1, 2, 1.0),
    (3, 5, 1.1),
    (6, 10, 1.2),
    (11, 9999, 1.3),
]


def _get_time_factor(months: float) -> float:
    """根据分层时间表获取衰减因子。"""
    for lo, hi, factor in TIME_DECAY_TABLE:
        if lo <= months < hi:
            return factor
    return 0.5


def _get_freq_factor(mention_count: int) -> float:
    """根据提及频次表获取加权因子。"""
    for lo, hi, factor in FREQ_FACTOR_TABLE:
        if lo <= mention_count <= hi:
            return factor
    return 1.0


def compute_dynamic_weight(initial_importance: float, months_age: float, mention_count: int) -> float:
    """计算L2记忆的动态权重。

    current = initial × time_factor × freq_factor
    """
    time_factor = _get_time_factor(months_age)
    freq_factor = _get_freq_factor(mention_count)
    return initial_importance * time_factor * freq_factor


async def increment_mention_count(memory_id: str) -> None:
    """在检索命中时增加记忆的提及次数。"""
    try:
        await db.execute_raw(
            """
            UPDATE memories
            SET mention_count = mention_count + 1, updated_at = NOW()
            WHERE id = $1
            """,
            memory_id,
        )
    except Exception as e:
        logger.warning(f"Failed to increment mention count for {memory_id}: {e}")


async def decay_importance():
    """Apply tiered importance decay to all active L2/L3 memories.

    Uses tiered time table instead of exponential decay.
    L1 memories are never decayed.
    """
    # Apply tiered decay using SQL CASE statement
    result = await db.execute_raw(
        """
        UPDATE memories
        SET importance = importance * (
            CASE
                WHEN EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 / 30 < 1 THEN 1.0
                WHEN EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 / 30 < 3 THEN 0.9
                WHEN EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 / 30 < 6 THEN 0.8
                WHEN EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 / 30 < 12 THEN 0.7
                WHEN EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 / 30 < 24 THEN 0.6
                ELSE 0.5
            END
        ) * (
            CASE
                WHEN mention_count <= 2 THEN 1.0
                WHEN mention_count <= 5 THEN 1.1
                WHEN mention_count <= 10 THEN 1.2
                ELSE 1.3
            END
        )
        WHERE is_archived = false AND level != 1
        """
    )
    logger.info(f"Tiered decay applied to {result} memories")

    # Archive memories below threshold
    archived = await db.execute_raw(
        f"""
        UPDATE memories
        SET is_archived = true
        WHERE is_archived = false AND importance < {ARCHIVE_THRESHOLD} AND level != 1
        """
    )
    logger.info(f"Archived {archived} memories below threshold {ARCHIVE_THRESHOLD}")
