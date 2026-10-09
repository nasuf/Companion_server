"""Compatibility API for bounded cumulative maintenance.

Legacy pure factor functions remain for historical offline comparisons. The
production entrypoints delegate exclusively to lazy_update's pure decay.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.redis_client import get_redis
from app.services.memory.taxonomy import is_singleton

logger = logging.getLogger(__name__)

# Redis key for tracking when a memory first dropped below 0.50 current_score.
# Cleared when the score recovers; demote once `now - since >= 30 days`.
_LOW_SCORE_TTL = 60 * 60 * 24 * 45  # 45 days auto-cleanup


def _low_score_key(side: str, mem_id: str) -> str:
    return f"l2:below_threshold_since:{side}:{mem_id}"


def _time_factor(days_since_access: int) -> float:
    if days_since_access < 30:
        return 1.0
    if days_since_access < 90:
        return 0.9
    if days_since_access < 180:
        return 0.8
    if days_since_access < 365:
        return 0.7
    if days_since_access < 730:
        return 0.6
    return 0.5


def _frequency_factor(mentions_1y: int) -> float:
    if mentions_1y <= 2:
        return 1.0
    if mentions_1y <= 5:
        return 1.1
    if mentions_1y <= 10:
        return 1.2
    return 1.3


def _quality_factor(corrections_1y: int, evidence_links: int) -> float:
    """Bounded quality modifier for P1 memory governance.

    Confirmed/user corrections reduce confidence in a memory's current form;
    source evidence links give a small stability boost. The factor stays close
    to 1.0 so spec time/frequency dynamics remain the dominant signal.
    """
    penalty = min(0.30, max(0, corrections_1y) * 0.10)
    boost = min(0.10, max(0, evidence_links) * 0.03)
    return round(max(0.70, min(1.10, 1.0 - penalty + boost)), 3)


async def _check_promotion_conditions(mem, side: str) -> bool:
    """L2→L1 晋升的**结构性**闸门 (值本身够不够由调用方判定)。

    spec §1.5.2 字面还要求"用户曾表达过重要" (changelog 里有 user_emphasized)。
    那一条已删除: 它和分数、频率是 AND 关系, 而 user_emphasized 只有在用户说出
    "一定要记住"这类话时才写入 —— 生产上历史晋升次数为 0, 等于根本没有晋升路径。
    一条被反复调用、始终有用的记忆升不上 L1, 分层就只剩下降通道。

    改为纯值驱动后, "用户强调过"仍然有用, 只是改在录入期抬高 importance, 而不是
    在晋升期当一票否决。

    这里保留的是真正的结构性约束: 同一 singleton 子类不能出现第二条 L1。
    """
    # Side-aware L1 conflict check (B5 fix): query the same table the memory
    # belongs to. A user-side L2 should only check user L1 conflicts; same for ai.
    # workspaceId 过滤确保同一 user 的不同 agent (workspace) L1 不会被误判冲突,
    # 每个 workspace 的 L1 是独立空间.
    #
    # SINGLETON 闸门: 该子类已有任何 L1 (姓名/年龄/生日 等硬唯一字段) → 一律
    # 拒绝晋升. 旧实现用字符 overlap>0.5 "相似即放行" — 相似恰恰意味着同一
    # 事实, 晋升近重复会造成第二条 singleton L1 (双"姓名"), 且 model.update
    # 直写不经过 store_memory 的 singleton 闸门, 无人兜底.
    if is_singleton(mem.mainCategory, mem.subCategory):
        model = db.usermemory if side == "user" else db.aimemory
        existing_l1 = await model.find_many(
            where={
                "userId": mem.userId,
                "workspaceId": mem.workspaceId,
                "level": 1,
                "isArchived": False,
                "mainCategory": mem.mainCategory,
                "subCategory": mem.subCategory,
            },
            take=1,
        )
        if any(l1.id != mem.id for l1 in existing_l1):
            logger.info(
                f"L2→L1 blocked: {side}/{mem.id} singleton "
                f"{mem.mainCategory}/{mem.subCategory} already has an L1"
            )
            return False

    return True


async def _track_low_score_streak(side: str, mem_id: str, below_threshold: bool) -> bool:
    """Track continuous-below-threshold streak in Redis.

    Returns True iff the memory has been continuously below 0.50 for ≥ 30 days
    (i.e. spec §1.5.2 L3 demote condition).
    """
    redis = await get_redis()
    key = _low_score_key(side, mem_id)
    if not below_threshold:
        # Score recovered — clear the streak marker
        await redis.delete(key)
        return False

    raw = await redis.get(key)
    now = datetime.now(UTC)
    if raw is None:
        # First time dropping below — mark now
        await redis.set(key, now.isoformat(), ex=_LOW_SCORE_TTL)
        return False

    # redis_client is configured with decode_responses=True so raw is str.
    try:
        since = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        # Corrupted marker — reset
        await redis.set(key, now.isoformat(), ex=_LOW_SCORE_TTL)
        return False

    return (now - since).days >= 30


async def _adjust_side(side: str, user_id: str | None) -> dict:
    """Compatibility entry; maintenance uses the same reward-free lifecycle."""
    from app.services.memory.lifecycle.lazy_update import sweep_stale_values

    result = await sweep_stale_values(user_id=user_id, sources=(side,))
    return result[side]


async def run_l2_adjustment(user_id: str | None = None) -> dict:
    """Bounded pure decay; never overwrite cumulative scores from importance."""
    from app.services.memory.lifecycle.lazy_update import sweep_stale_values

    result = await sweep_stale_values(user_id=user_id)
    return {
        "user": result["user"], "ai": result["ai"],
        "total": result["scanned"], "promoted": result["promoted"],
        "demoted": result["demoted"], "adjusted": result["adjusted"],
        "engine": "cumulative_decay_v2", "batch_limit_per_side": 1000,
    }
