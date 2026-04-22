"""Memory mention-count maintenance.

Spec §1.5 的记忆衰减/升降级统一由 [l2_dynamics.py:run_l2_adjustment] 实现，
本模块只负责维护 `mention_count` 列（denormalized counter, 供 admin 显示用）。
"""

import logging

from app.db import db

logger = logging.getLogger(__name__)


async def increment_mention_count(memory_id: str) -> None:
    """检索命中时增加记忆的累计提及次数（admin UI 展示用）。

    仅对未归档的记忆递增。Spec §1.5.2 的频率因子由 l2_dynamics 从
    memory_changelog (operation='access') 1 年滑动窗口计算，不依赖此列。
    """
    try:
        updated = await db.execute_raw(
            "UPDATE memories_user SET mention_count = mention_count + 1, updated_at = NOW() "
            "WHERE id = $1 AND is_archived = false",
            memory_id,
        )
        if not updated:
            await db.execute_raw(
                "UPDATE memories_ai SET mention_count = mention_count + 1, updated_at = NOW() "
                "WHERE id = $1 AND is_archived = false",
                memory_id,
            )
    except Exception as e:
        logger.warning(f"Failed to increment mention count for {memory_id}: {e}")
