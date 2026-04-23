"""L2 dynamic scoring and level adjustment.

Product spec §1.5.2: L2 memories have a "current score" that decays with
time and grows with mention frequency. Periodically (daily cron) we
recalculate current scores and promote/demote as needed:

  current_score = initial_importance × time_factor × frequency_factor

Time factor (days since last accessed/mentioned):
  <30d → 1.0 | 30-90d → 0.9 | 90-180d → 0.8 | 180-365d → 0.7
  365-730d → 0.6 | >730d → 0.5

Frequency factor (mentions in rolling 1-year window):
  1-2 → 1.0 | 3-5 → 1.1 | 6-10 → 1.2 | >10 → 1.3

Level transitions:
  current_score ≥ 0.85 AND mentions≥10 AND user expressed importance
    AND no L1 conflict → promote to L1
  0.50 ≤ current_score < 0.85 → stay L2
  current_score < 0.50 *持续 30 天* → demote to L3
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.redis_client import get_redis
from app.services.memory.taxonomy import L1_SINGLETON_SUBS

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


async def _check_promotion_conditions(mem, side: str) -> bool:
    """Spec §1.5.2 extra conditions for L2→L1 promotion:
    - User expressed importance (changelog contains 'user_emphasized')
    - No conflict with existing L1 memories on same side (user vs ai)
    """
    emphasis_count = await db.memorychangelog.count(
        where={"memoryId": mem.id, "operation": "user_emphasized"},
    )
    if emphasis_count == 0:
        return False

    # Side-aware L1 conflict check (B5 fix): query the same table the memory
    # belongs to. A user-side L2 should only check user L1 conflicts; same for ai.
    model = db.usermemory if side == "user" else db.aimemory
    existing_l1 = await model.find_many(
        where={
            "userId": mem.userId,
            "level": 1,
            "isArchived": False,
            "mainCategory": mem.mainCategory,
            "subCategory": mem.subCategory,
        },
        take=5,
    )
    mem_content = (mem.summary or mem.content or "").lower()
    for l1 in existing_l1:
        l1_content = (l1.summary or l1.content or "").lower()
        if l1_content and mem_content and l1.id != mem.id:
            overlap = sum(1 for c in mem_content if c in l1_content)
            if overlap > len(mem_content) * 0.5:
                continue
            if (mem.mainCategory, mem.subCategory) in L1_SINGLETON_SUBS:
                logger.info(
                    f"L2→L1 blocked: {side}/{mem.id} conflicts with L1 "
                    f"{l1.id} in singleton {mem.mainCategory}/{mem.subCategory}"
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
    """Process L2 adjustments for one side (user or ai)."""
    now = datetime.now(UTC)
    one_year_ago = now - timedelta(days=365)

    model = db.usermemory if side == "user" else db.aimemory

    where: dict = {"level": 2, "isArchived": False}
    if user_id:
        where["userId"] = user_id

    l2_memories = await model.find_many(where=where)
    if not l2_memories:
        return {"side": side, "total": 0, "promoted": 0, "demoted": 0, "adjusted": 0}

    mem_ids = [m.id for m in l2_memories]
    mention_counts: dict[str, int] = {}
    if mem_ids:
        rows = await db.query_raw(
            """
            SELECT "memoryId", COUNT(*)::int AS cnt
            FROM "memory_changelogs"
            WHERE "memoryId" = ANY($1::text[])
              AND "createdAt" >= $2
              AND "operation" = 'access'
            GROUP BY "memoryId"
            """,
            mem_ids,
            one_year_ago,
        )
        for r in rows:
            mention_counts[r.get("memoryId", "")] = r.get("cnt", 0)

    promoted = 0
    demoted = 0
    adjusted = 0
    updates: list[tuple[str, dict]] = []

    for mem in l2_memories:
        initial_importance = float(mem.importance or 0.5)

        last_access = mem.updatedAt or mem.createdAt
        if last_access and last_access.tzinfo:
            days = (now - last_access).days
        else:
            days = 90

        tf = _time_factor(days)
        mc = mention_counts.get(mem.id, 0)
        ff = _frequency_factor(mc)
        current_score = initial_importance * tf * ff

        # Track the continuous-below-threshold streak regardless of outcome
        sustained_low = await _track_low_score_streak(
            side, mem.id, below_threshold=current_score < 0.50,
        )

        if current_score >= 0.85 and mc >= 10:
            if not await _check_promotion_conditions(mem, side):
                if abs(current_score - initial_importance) > 0.01:
                    updates.append((mem.id, {"importance": current_score}))
                    adjusted += 1
            else:
                updates.append((mem.id, {"level": 1, "importance": current_score}))
                promoted += 1
        elif sustained_low:
            # Spec §1.5.2: demote only after continuously below 0.50 for 30+ days
            updates.append((mem.id, {"level": 3, "importance": current_score}))
            demoted += 1
        elif abs(current_score - initial_importance) > 0.01:
            updates.append((mem.id, {"importance": current_score}))
            adjusted += 1

    for mid, data in updates:
        try:
            await model.update(where={"id": mid}, data=data)
        except Exception as e:
            logger.warning(f"L2 update failed ({side}/{mid}): {e}")

    stats = {
        "side": side,
        "total": len(l2_memories),
        "promoted": promoted,
        "demoted": demoted,
        "adjusted": adjusted,
    }
    logger.info(f"L2 adjustment [{side}] complete: {stats}")
    return stats


async def run_l2_adjustment(user_id: str | None = None) -> dict:
    """Recalculate L2 scores and apply promote/demote rules for BOTH sides.

    Spec §1.5.2 applies to user and AI memories symmetrically. If user_id is
    None, runs for all users. The two sides share no state (distinct tables,
    distinct Redis keys) so we run them concurrently.
    """
    user_stats, ai_stats = await asyncio.gather(
        _adjust_side("user", user_id),
        _adjust_side("ai", user_id),
    )
    return {
        "user": user_stats,
        "ai": ai_stats,
        "total": user_stats["total"] + ai_stats["total"],
        "promoted": user_stats["promoted"] + ai_stats["promoted"],
        "demoted": user_stats["demoted"] + ai_stats["demoted"],
        "adjusted": user_stats["adjusted"] + ai_stats["adjusted"],
    }
