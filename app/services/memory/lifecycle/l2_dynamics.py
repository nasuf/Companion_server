"""L2 动态分与层级调整 —— **已退为兜底路径**.

主路径现在是 `lifecycle/lazy_update.py`: 记忆被检索到的那一刻就地更新效用值。
改动的原因是这个 cron 的两个结构性问题都在生产上兑现了:

  1. 它因为一处 SQL 类型错死了几个月, 期间所有 L2 零衰减、零升降级, 无任何告警。
     整个记忆生命周期押在一个夜间任务上, 它一停就是彻底静止。
  2. 每晚从不可变的初始 importance 重算, 分数不累积 —— "用过一百次"和"用过十次"
     落在同一个频率档里, 使用信号被档位抹平。

这个模块保留下来是因为惰性更新有个盲区: 彻底没人问津的记忆永远不会被检索到,
也就永远不会被更新 —— 而那恰恰是最该衰减的一批。`sweep_stale_values` 补这个洞。
下面的旧公式仍在跑, 但它现在只是第二道保险: 即使完全不跑, 活跃记忆的值依然正确。

新增记忆或需要改公式时, 改 `lifecycle/value.py` —— 那里是效用值的唯一定义处,
SQL 与离线推演都从它取常数。

--- 以下为旧实现的说明 ---

Product spec §1.5.2: L2 memories have a "current score" that decays with
time and grows with mention frequency. Periodically (daily cron) we
recalculate current scores and promote/demote as needed:

  current_score = initial_importance × time_factor × frequency_factor
  P1 adds a bounded quality factor derived from changelog signals.

`importance` is the IMMUTABLE initial score (the formula's base); the computed
dynamic score is persisted to the separate `current_score` column, which the
retrieval ranker reads via COALESCE(current_score, importance). Never write the
computed score back into `importance` — doing so compounds the factors on every
cron run (upward inflation for frequently-accessed rows, downward spiral for
idle ones) because the next run would treat last night's product as the base.

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
from app.observability.events import EVT_MEMORY_L2_ADJUSTED
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
    """Process L2 adjustments for one side (user or ai)."""
    now = datetime.now(UTC)
    one_year_ago = now - timedelta(days=365)

    model = db.usermemory if side == "user" else db.aimemory

    # 周期性 L2 衰减扫描: 每个 user 的所有 L2 都要算时间+频率因子更新分数,
    # 按 userId 过滤足够 (user_id=None 时 cron 模式全量扫), 不需要 workspaceId
    # 过滤——每个 L2 独立 score, 跨 workspace 不相互依赖.
    where: dict = {"level": 2, "isArchived": False}
    if user_id:
        where["userId"] = user_id

    l2_memories = await model.find_many(where=where)
    if not l2_memories:
        return {"side": side, "total": 0, "promoted": 0, "demoted": 0, "adjusted": 0}

    mem_ids = [m.id for m in l2_memories]
    mention_counts: dict[str, int] = {}
    last_access_at: dict[str, datetime] = {}
    quality_counts: dict[str, dict[str, int]] = {}
    if mem_ids:
        # Spec time_factor is "days since last access". Read the real last access
        # from the changelog instead of `updatedAt` — the row's updatedAt is
        # @updatedAt-refreshed by this cron's own writes (and any admin edit),
        # which would freeze the time factor at 1.0 forever. The 1-year window
        # only applies to the frequency count; last access is MAX(created_at)
        # over retained access rows. NB: changelog_retention purges `access`
        # rows older than 13 months, so a row whose newest access predates that
        # falls back to createdAt (a strictly older or equal timestamp → equal
        # or MORE decay, never inflated). Bounded, acknowledged tradeoff.
        rows = await db.query_raw(
            """
            SELECT memory_id,
                   COUNT(*) FILTER (WHERE created_at >= $2::timestamp)::int AS cnt,
                   MAX(created_at) AS last_access
            FROM memory_changelogs
            WHERE memory_id = ANY($1::text[])
              AND operation = 'access'
            GROUP BY memory_id
            """,
            mem_ids,
            one_year_ago,
        )
        for r in rows:
            mid = r.get("memory_id", "")
            mention_counts[mid] = r.get("cnt", 0)
            raw_last = r.get("last_access")
            if isinstance(raw_last, str):
                try:
                    raw_last = datetime.fromisoformat(raw_last.replace("Z", "+00:00"))
                except ValueError:
                    raw_last = None
            if isinstance(raw_last, datetime):
                if raw_last.tzinfo is None:
                    raw_last = raw_last.replace(tzinfo=UTC)
                last_access_at[mid] = raw_last
        quality_rows = await db.query_raw(
            """
            SELECT
              memory_id,
              SUM(CASE WHEN operation IN (
                'user_edit',
                'contradiction_archived',
                'contradiction_new',
                'retrieval_feedback_confirmed'
              ) THEN 1 ELSE 0 END)::int AS corrections,
              SUM(CASE WHEN operation = 'evidence_linked' THEN 1 ELSE 0 END)::int AS evidence_links
            FROM memory_changelogs
            WHERE memory_id = ANY($1::text[])
              AND created_at >= $2::timestamp
            GROUP BY memory_id
            """,
            mem_ids,
            one_year_ago,
        )
        for r in quality_rows:
            mid = r.get("memory_id", "")
            quality_counts[mid] = {
                "corrections": int(r.get("corrections") or 0),
                "evidence_links": int(r.get("evidence_links") or 0),
            }

    promoted = 0
    demoted = 0
    adjusted = 0
    # (memory_id, update_data, changelog_op_or_None, mem_ref)
    updates: list[tuple[str, dict, str | None, object]] = []

    for mem in l2_memories:
        # IMMUTABLE base — never overwritten by this cron (see module docstring).
        initial_importance = float(mem.importance or 0.5)

        last_access = last_access_at.get(mem.id) or mem.createdAt
        if isinstance(last_access, datetime):
            if last_access.tzinfo is None:
                last_access = last_access.replace(tzinfo=UTC)
            days = (now - last_access).days
        else:
            days = 90

        tf = _time_factor(days)
        mc = mention_counts.get(mem.id, 0)
        ff = _frequency_factor(mc)
        q_counts = quality_counts.get(mem.id, {})
        qf = _quality_factor(
            q_counts.get("corrections", 0),
            q_counts.get("evidence_links", 0),
        )
        current_score = max(0.0, min(1.0, initial_importance * tf * ff * qf))

        # Track the continuous-below-threshold streak regardless of outcome
        sustained_low = await _track_low_score_streak(
            side, mem.id, below_threshold=current_score < 0.50,
        )

        prev_score = getattr(mem, "currentScore", None)
        score_changed = (
            prev_score is None or abs(current_score - float(prev_score)) > 0.01
        )

        if current_score >= 0.85 and mc >= 10:
            if not await _check_promotion_conditions(mem, side):
                if score_changed:
                    updates.append((mem.id, {"currentScore": current_score}, None, mem))
                    adjusted += 1
            else:
                # One-time level transition: promoted rows must land in the L1
                # importance band (≥0.85); this is the only place importance is
                # written, and it's a transition, not a recompute.
                updates.append((
                    mem.id,
                    {
                        "level": 1,
                        "importance": min(1.0, max(initial_importance, 0.85)),
                        "currentScore": current_score,
                    },
                    "promote",
                    mem,
                ))
                promoted += 1
        elif sustained_low:
            # Spec §1.5.2: demote only after continuously below 0.50 for 30+ days.
            # importance (initial score) stays untouched — the demotion itself is
            # recorded by the level change + changelog.
            updates.append((
                mem.id, {"level": 3, "currentScore": current_score}, "demote", mem,
            ))
            demoted += 1
        elif score_changed:
            updates.append((mem.id, {"currentScore": current_score}, None, mem))
            adjusted += 1

    for mid, data, changelog_op, mem_ref in updates:
        try:
            await model.update(where={"id": mid}, data=data)
        except Exception as e:
            logger.warning(f"L2 update failed ({side}/{mid}): {e}")
            continue
        if changelog_op:
            try:
                from app.services.memory.storage.persistence import log_memory_changelog

                await log_memory_changelog(
                    getattr(mem_ref, "userId", user_id or ""),
                    mid,
                    changelog_op,
                    old_value="level=2",
                    new_value=f"level={data.get('level')} current_score={data.get('currentScore'):.3f}",
                    workspace_id=getattr(mem_ref, "workspaceId", None),
                )
            except Exception as e:
                logger.debug(f"L2 {changelog_op} changelog write failed ({mid}): {e}")

    stats = {
        "side": side,
        "total": len(l2_memories),
        "promoted": promoted,
        "demoted": demoted,
        "adjusted": adjusted,
    }
    logger.info(
        f"L2 adjustment [{side}] complete: {stats}",
        extra={
            "event": EVT_MEMORY_L2_ADJUSTED,
            "side": side,
            "n_total": len(l2_memories),
            "n_promoted": promoted,
            "n_demoted": demoted,
            "n_adjusted": adjusted,
        },
    )
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
