"""在检索热路径上就地更新记忆效用值.

衰减发生在**记忆被用到的那一刻**, 不是夜里回头补算。这样值不依赖任何定时任务
活着 —— 生产上那个 cron 死了几个月都没人发现, 期间零衰减。

热路径的三条约束:

    不阻塞    全部在 fire-and-forget 的后台任务里跑, 失败只记日志
    少往返    一轮对话涉及十几条记忆, 用单条 UPDATE ... FROM (VALUES ...) 批量写
    幂等      值更新带 value_updated_at 时间戳, 重复执行只会让 Δt=0 即不衰减,
              不会双重扣分。这让它和 cron 兜底扫描可以安全并存。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.db import db
from app.services.memory.lifecycle.value import (
    ACCESS_CEILING,
    ACCESS_REWARD,
    CONTRIBUTION_REWARD,
    DECAY_LAMBDA,
    HOT_DEMOTE_AT,
    HOT_PROMOTE_AT,
    VALUE_MAX,
    WARM_DEMOTE_AT,
    WARM_PROMOTE_AT,
)
from app.services.memory.taxonomy import L1_SINGLETON_SUBS

logger = logging.getLogger(__name__)

_TABLES = ("memories_user", "memories_ai")


def _singleton_arrays() -> tuple[list[str], list[str]]:
    """把 (主类, 子类) 二元组拆成两个并行数组供 SQL 配对匹配。

    L1_SINGLETON_SUBS 是元组集合, 不是子类名集合 —— 直接当 text[] 传会在驱动层
    报序列化错, 而调用方吞异常, 于是整个效用值更新静默失效。上线前的 EXPLAIN
    验证就是抓这个的。
    """
    pairs = sorted(L1_SINGLETON_SUBS)
    return [main for main, _ in pairs], [sub for _, sub in pairs]


def _render_sql(table: str) -> str:
    """把常数烘进 SQL。常数来自 value.py 这一处定义, 两边不会漂。"""
    return _UPDATE_SQL.format(
        table=table, vmax=VALUE_MAX, lam=DECAY_LAMBDA,
        alpha=ACCESS_REWARD, beta=CONTRIBUTION_REWARD, ceiling=ACCESS_CEILING,
        hot_up=HOT_PROMOTE_AT, hot_down=HOT_DEMOTE_AT,
        warm_up=WARM_PROMOTE_AT, warm_down=WARM_DEMOTE_AT,
    )

# 层级迁移和值更新在同一条 SQL 里做完。放在 SQL 里而不是先 SELECT 再算再 UPDATE,
# 是为了避免热路径上的读-改-写竞态: 两个并发的对话轮次同时更新同一条记忆时,
# 后写的会覆盖先写的回报。让数据库基于当前行值做算术就没有这个问题。
#
# 身份事实豁免降级 —— 见 value.next_level 的说明。singleton 是 (主类, 子类) 二元组
# 而不是子类名, 所以按两个并行数组配对匹配: 只比子类会让不同主类下的同名子类被
# 误判 (taxonomy 里"其他"这类子类在多个主类下都存在)。
#
# 两种信号形式不同 (见 value.ACCESS_CEILING): contribution 是加法, access 是趋向
# 天花板的递减回报。u.is_contribution 区分二者。衰减后的分数在 CTE 里算一次,
# 值和层级都从它派生, 避免同一表达式抄两遍而漂移。
_UPDATE_SQL = """
WITH scored AS (
  SELECT
    m.id,
    m.level AS old_level,
    CASE WHEN u.is_contribution
      THEN LEAST({vmax}, GREATEST(0.0, d.decayed + {beta}))
      ELSE LEAST({vmax}, GREATEST(0.0,
        d.decayed + {alpha} * GREATEST(0.0, {ceiling} - d.decayed)))
    END AS val,
    -- 该 (主类, 子类) 是否属于 singleton
    EXISTS (
      SELECT 1 FROM unnest($3::text[], $4::text[]) AS sg(main, sub)
      WHERE sg.main = m.main_category AND sg.sub = m.sub_category
    ) AS is_singleton,
    -- singleton 闸门: 该类目已存在别的 L1 时禁止再升一条上去。旧的夜间 cron 在
    -- Python 侧做这个检查, 惰性更新把层级迁移搬到了热路径, 必须一并搬过来 ——
    -- 否则两条"姓名"记忆会同时坐在 L1 上, 正是人设分层要消灭的那种数据损坏。
    EXISTS (
      SELECT 1 FROM {table} AS o
      WHERE o.level = 1 AND o.is_archived = false AND o.id <> m.id
        AND o.user_id = m.user_id
        AND o.workspace_id IS NOT DISTINCT FROM m.workspace_id
        AND o.main_category = m.main_category
        AND o.sub_category = m.sub_category
    ) AS l1_taken
  FROM {table} AS m
  JOIN (SELECT unnest($1::text[]) AS id, unnest($2::bool[]) AS is_contribution) AS u
    ON m.id = u.id
  CROSS JOIN LATERAL (SELECT
    COALESCE(m.current_score, m.importance)
      * EXP(-{lam} * GREATEST(0, EXTRACT(EPOCH FROM
          (CURRENT_TIMESTAMP - COALESCE(m.value_updated_at, m.created_at))) / 86400.0))
    AS decayed) AS d
  WHERE m.is_archived = false
)
UPDATE {table} AS t SET
  current_score = s.val,
  level = CASE
    WHEN s.is_singleton AND s.old_level = 1 THEN 1
    WHEN s.old_level = 1 THEN CASE WHEN s.val < {hot_down} THEN 2 ELSE 1 END
    WHEN s.old_level = 2 THEN CASE
      WHEN s.val >= {hot_up} AND NOT (s.is_singleton AND s.l1_taken) THEN 1
      WHEN s.val < {warm_down} THEN 3
      ELSE 2 END
    ELSE CASE WHEN s.val >= {warm_up} THEN 2 ELSE 3 END
  END,
  value_updated_at = CURRENT_TIMESTAMP
FROM scored AS s
WHERE t.id = s.id
"""


def _signals(
    contributed_ids: list[str], accessed_ids: list[str],
) -> dict[str, bool]:
    """把两种使用信号归到每条记忆上, True 表示"被注入过"。

    同时进候选又被注入的记忆只按 contribution 计, 不叠加 —— 注入本来就蕴含了
    "进过候选", 叠加等于给同一件事记两次功。
    """
    signals = {mid: False for mid in accessed_ids if mid}
    signals.update({mid: True for mid in contributed_ids if mid})
    return signals


async def record_memory_usage(
    *,
    contributed_ids: list[str] | None = None,
    accessed_ids: list[str] | None = None,
) -> int:
    """记忆被用到时更新它们的效用值与层级, 返回更新的行数。

    contributed_ids 是真正注入 prompt 的; accessed_ids 是进了候选但没能注入的。
    后者权重更低 (AMV-L 的 α < β), 作用是让"总差一口气"的记忆不至于一路凉到底。
    """
    signals = _signals(contributed_ids or [], accessed_ids or [])
    if not signals:
        return 0

    ids = list(signals)
    is_contribution = [signals[i] for i in ids]
    sg_main, sg_sub = _singleton_arrays()

    total = 0
    for table in _TABLES:
        # ID 全局唯一, 所以对另一张表是空操作 —— 比先查归属再定表少一次往返。
        try:
            total += await db.execute_raw(
                _render_sql(table), ids, is_contribution, sg_main, sg_sub,
            )
        except Exception as e:
            # 效用值更新是尽力而为: 失败只意味着这次使用没被记入, 下次还会记。
            # 绝不能让它影响回复 —— 调用方在后台任务里, 这里再兜一层。
            logger.debug(f"lazy value update failed on {table}: {e}")
    return total


async def sweep_stale_values(*, older_than_days: int = 30, limit: int = 5000) -> dict:
    """兜底扫描: 照顾长期没被用到、因而惰性更新碰不到的记忆。

    惰性更新只在记忆被检索到时触发, 所以彻底没人问津的记忆永远不会衰减 —— 那恰恰
    是最该衰减的一批。这个扫描补上这个盲区。

    它不是主路径。旧实现把整个生命周期押在夜间 cron 上, 结果 cron 死了几个月无人
    察觉。现在即使这个扫描完全不跑, 活跃记忆的值仍然是对的, 只有僵尸记忆会滞留在
    偏高的层级 —— 影响面小得多。
    """
    cutoff_expr = f"CURRENT_TIMESTAMP - INTERVAL '{int(older_than_days)} days'"
    sg_main, sg_sub = _singleton_arrays()
    stats = {"scanned": 0, "demoted": 0}

    for table in _TABLES:
        # 复用同一条 UPDATE, 只把"哪些行参与"换成久未更新的一批, 且不带任何使用
        # 信号 —— 纯衰减。按最旧优先 + LIMIT, 让单次扫描的规模和持锁时间可控。
        scoped = _render_sql(table).replace(
            "JOIN (SELECT unnest($1::text[]) AS id, unnest($2::bool[]) "
            "AS is_contribution) AS u\n    ON m.id = u.id",
            "JOIN (SELECT id, false AS is_contribution FROM {t} "
            "WHERE is_archived = false "
            "AND COALESCE(value_updated_at, created_at) < {cut} "
            "ORDER BY COALESCE(value_updated_at, created_at) ASC "
            "LIMIT {lim}) AS u\n    ON m.id = u.id".format(
                t=table, cut=cutoff_expr, lim=int(limit)),
        )
        # $1/$2 已被内联查询取代, 只剩 singleton 两个数组 —— 重编号为 $1/$2。
        scoped = scoped.replace("$3::text[], $4::text[]", "$1::text[], $2::text[]")
        if "$3" in scoped or "unnest($1::text[]) AS id" in scoped:
            # 占位符改写没生效就会把带参 SQL 当另一组参数执行。宁可不扫。
            logger.error(f"stale sweep SQL rewrite failed for {table}; skipped")
            continue
        try:
            stats["scanned"] += await db.execute_raw(scoped, sg_main, sg_sub)
        except Exception as e:
            logger.warning(f"stale value sweep failed on {table}: {e}")
    stats["swept_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return stats
