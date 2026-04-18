"""L2 dynamic scoring and level adjustment.

Product spec §1.5.2: L2 memories have a "current score" that decays with
time and grows with mention frequency. Periodically (e.g. daily cron) we
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
  current_score < 0.50 for 30+ continuous days → demote to L3
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from app.db import db

logger = logging.getLogger(__name__)


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


async def _check_promotion_conditions(mem) -> bool:
    """Spec §1.5.2: additional L2→L1 promotion conditions:
    - User expressed importance (changelog contains importance keywords)
    - No conflict with existing L1 memories
    """
    # Condition 3: user expressed importance — check for 'user_emphasized'
    # operation in changelog (tagged at extraction time in pipeline.py)
    emphasis_count = await db.memorychangelog.count(
        where={"memoryId": mem.id, "operation": "user_emphasized"},
    )
    if emphasis_count == 0:
        return False

    # Condition 4: no conflict with existing L1 in same category
    existing_l1 = await db.aimemory.find_many(
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
        # Simple heuristic: if both are about the same sub-category and
        # contain contradictory content, block promotion
        if l1_content and mem_content and l1.id != mem.id:
            # If they share >50% characters, likely same fact — skip
            overlap = sum(1 for c in mem_content if c in l1_content)
            if overlap > len(mem_content) * 0.5:
                continue
            # Different content in same singleton sub = potential conflict
            from app.services.memory.taxonomy import L1_SINGLETON_SUBS
            if (mem.mainCategory, mem.subCategory) in L1_SINGLETON_SUBS:
                logger.info(f"L2→L1 blocked: {mem.id} conflicts with L1 {l1.id} in singleton {mem.mainCategory}/{mem.subCategory}")
                return False

    return True


async def run_l2_adjustment(user_id: str | None = None) -> dict:
    """Recalculate L2 scores and apply promote/demote rules.

    Call this from a daily cron job. If user_id is None, runs for all users.
    Returns summary stats.
    """
    now = datetime.now(UTC)
    one_year_ago = now - timedelta(days=365)

    where: dict = {"level": 2, "isArchived": False}
    if user_id:
        where["userId"] = user_id

    l2_memories = await db.aimemory.find_many(where=where)
    if not l2_memories:
        return {"total": 0, "promoted": 0, "demoted": 0, "adjusted": 0}

    # Batch: count RETRIEVAL events (operation='access') per memory in one
    # query. Spec §1.5.2: "最近1年内被调用的次数" = times retrieved into prompt.
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
    # Collect batch updates to minimize DB roundtrips
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

        if current_score >= 0.85 and mc >= 10:
            # Spec §1.5.2: 4 conditions for L2→L1 promotion:
            #   ✅ current_score ≥ 0.85
            #   ✅ mentions ≥ 10 in past year
            #   ⬇ user expressed importance (keyword scan)
            #   ⬇ no L1 conflict (check existing L1)
            if not await _check_promotion_conditions(mem):
                # Conditions 3-4 not met; stay L2 with updated score
                if abs(current_score - initial_importance) > 0.01:
                    updates.append((mem.id, {"importance": current_score}))
                    adjusted += 1
            else:
                updates.append((mem.id, {"level": 1, "importance": current_score}))
                promoted += 1
        elif current_score < 0.50 and days >= 30:
            updates.append((mem.id, {"level": 3, "importance": current_score}))
            demoted += 1
        elif abs(current_score - initial_importance) > 0.01:
            updates.append((mem.id, {"importance": current_score}))
            adjusted += 1

    # Apply updates (still per-row via Prisma; could batch with raw SQL
    # if volume justifies it)
    for mid, data in updates:
        try:
            await db.aimemory.update(where={"id": mid}, data=data)
        except Exception as e:
            logger.warning(f"L2 update failed for {mid}: {e}")

    stats = {
        "total": len(l2_memories),
        "promoted": promoted,
        "demoted": demoted,
        "adjusted": adjusted,
    }
    logger.info(f"L2 adjustment complete: {stats}")
    return stats
