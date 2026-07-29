"""数据不变量: 抓"定时任务报了成功, 其实什么都没干成"这一类故障.

cron 健康判读只能看出任务崩没崩。但这个项目里最难发现的失效恰恰是任务**报成功**
的那种 —— 每一条不变量都对应一次真实事故:

    schedule_today       作息落库时用带时区的午夜, @db.Date 截断到前一天, 每天
                         覆盖前一天那行。任务全程报 ok, 错位了很久才被偶然发现。
    l2_scores_fresh      L2 动态分级 cron 因一个 SQL 类型错每晚崩几个月。
    memory_access_logged 检索访问打点断流, 而 L2 的频率因子全靠它。
    embeddings_complete  换 embedding 模型回填时漏行, 漏掉的记忆永远检索不到。
    consolidation_active 聚类阈值没跟着换模型, 整合空转不报错。

判定分三档: ok / warn / violated。warn 留给"看着不对但可能是正常波动"的情形
(比如今天恰好没人聊天, 访问打点自然为零), violated 才是确定出事。

每条检查都各自吞异常 —— 一条查不动不该让整份报告消失, 那等于又回到"看不见"。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

INVARIANTS_HEALTH_KEY = "ops:invariants"
_INVARIANTS_TTL_S = 30 * 24 * 3600


@dataclass
class InvariantResult:
    key: str
    title: str
    status: str  # ok | warn | violated | error
    detail: str = ""
    observed: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "observed": self.observed,
        }


async def _check_schedule_today() -> InvariantResult:
    """每个活跃 agent 今天都该有一行作息.

    热路径读的是 Redis, 所以 DB 这边错了聊天照常 —— 只有 Redis miss 回落查库时
    才会拿到错的。正因如此这条必须由巡检来盯, 用户侧感知不到。
    """
    from app.db import db

    rows = await db.query_raw(
        """
        SELECT
            (SELECT COUNT(*) FROM ai_agents WHERE status = 'active')::int AS agents,
            (SELECT COUNT(DISTINCT agent_id) FROM ai_daily_schedules
              WHERE date = CURRENT_DATE)::int AS scheduled
        """
    )
    agents = int(rows[0]["agents"] or 0)
    scheduled = int(rows[0]["scheduled"] or 0)
    observed = {"active_agents": agents, "with_schedule_today": scheduled}

    if agents == 0:
        return InvariantResult(
            "schedule_today", "今日作息覆盖", "ok", "没有活跃 agent", observed
        )
    # 模板 agent 不参与每日生成, 少一两个属正常。
    missing = agents - scheduled
    if missing <= 2:
        return InvariantResult(
            "schedule_today", "今日作息覆盖", "ok",
            f"{scheduled}/{agents} 个 agent 有今日作息", observed,
        )
    if scheduled == 0:
        return InvariantResult(
            "schedule_today", "今日作息覆盖", "violated",
            f"{agents} 个活跃 agent 今天一份作息都没落库", observed,
        )
    return InvariantResult(
        "schedule_today", "今日作息覆盖", "warn",
        f"{missing} 个 agent 缺今日作息", observed,
    )


async def _check_l2_scores_fresh() -> InvariantResult:
    """L2 动态分该每天被重算一遍."""
    from app.db import db

    rows = await db.query_raw(
        """
        SELECT
            COUNT(*)::int AS total,
            COUNT(*) FILTER (
                WHERE value_updated_at > CURRENT_TIMESTAMP - INTERVAL '48 hours'
            )::int AS fresh
        FROM (
            SELECT value_updated_at FROM memories_ai
             WHERE level = 2 AND is_archived = false
            UNION ALL
            SELECT value_updated_at FROM memories_user
             WHERE level = 2 AND is_archived = false
        ) AS l2
        """
    )
    total = int(rows[0]["total"] or 0)
    fresh = int(rows[0]["fresh"] or 0)
    observed = {"l2_total": total, "updated_within_48h": fresh}

    if total == 0:
        return InvariantResult(
            "l2_scores_fresh", "L2 动态分新鲜度", "ok", "暂无 L2 记忆", observed
        )
    ratio = fresh / total
    if ratio >= 0.9:
        return InvariantResult(
            "l2_scores_fresh", "L2 动态分新鲜度", "ok",
            f"{fresh}/{total} 条近 48h 内重算过", observed,
        )
    if ratio >= 0.5:
        return InvariantResult(
            "l2_scores_fresh", "L2 动态分新鲜度", "warn",
            f"只有 {fresh}/{total} 条近 48h 内重算过", observed,
        )
    return InvariantResult(
        "l2_scores_fresh", "L2 动态分新鲜度", "violated",
        f"{total} 条 L2 里只有 {fresh} 条近 48h 重算过, l2_adjustment 可能已失效",
        observed,
    )


async def _check_memory_access_logged() -> InvariantResult:
    """有人聊天就该有检索访问打点.

    单看"打点为零"会误报 —— 今天没人聊天时本来就是零。所以拿同期的消息量做分母,
    有对话却没有打点才算出事。
    """
    from app.db import db

    rows = await db.query_raw(
        """
        SELECT
            (SELECT COUNT(*) FROM memory_changelogs
              WHERE operation = 'access'
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours')::int AS accesses,
            (SELECT COUNT(*) FROM messages
              WHERE role = 'user'
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours')::int AS user_messages
        """
    )
    accesses = int(rows[0]["accesses"] or 0)
    messages = int(rows[0]["user_messages"] or 0)
    observed = {"accesses_24h": accesses, "user_messages_24h": messages}

    if messages == 0:
        return InvariantResult(
            "memory_access_logged", "记忆访问打点", "ok",
            "近 24h 没有用户消息", observed,
        )
    if accesses == 0:
        return InvariantResult(
            "memory_access_logged", "记忆访问打点", "violated",
            f"近 24h 有 {messages} 条用户消息但零条访问打点, L2 频率因子会失准",
            observed,
        )
    return InvariantResult(
        "memory_access_logged", "记忆访问打点", "ok",
        f"近 24h {accesses} 次访问 / {messages} 条用户消息", observed,
    )


async def _check_embeddings_complete() -> InvariantResult:
    """每条未归档记忆都该有向量, 否则它永远检索不到."""
    from app.db import db

    rows = await db.query_raw(
        """
        SELECT
            (SELECT COUNT(*) FROM memories_ai WHERE is_archived = false)::int
          + (SELECT COUNT(*) FROM memories_user WHERE is_archived = false)::int AS memories,
            (SELECT COUNT(*) FROM memory_embeddings)::int AS embeddings
        """
    )
    memories = int(rows[0]["memories"] or 0)
    embeddings = int(rows[0]["embeddings"] or 0)
    observed = {"active_memories": memories, "embeddings": embeddings}

    if memories == 0:
        return InvariantResult(
            "embeddings_complete", "向量覆盖率", "ok", "暂无记忆", observed
        )
    # 向量表也保留归档记忆的行, 所以只查"少了没", 多出来是正常的。
    missing = memories - embeddings
    if missing <= 0:
        return InvariantResult(
            "embeddings_complete", "向量覆盖率", "ok",
            f"{embeddings} 条向量覆盖 {memories} 条记忆", observed,
        )
    ratio = missing / memories
    status = "violated" if ratio > 0.05 else "warn"
    return InvariantResult(
        "embeddings_complete", "向量覆盖率", status,
        f"{missing} 条记忆没有向量, 这些记忆检索不到", observed,
    )


# 容量水位: 到这两条线时该动手做 nginx least_conn + 多实例了。
#
# 现在刻意不做, 因为倾斜要成为问题的前提是单 worker 接近饱和 —— 实测峰值同时进行
# 的对话约 2.7 轮、CPU 占用 0.21%, 就算连接 10:0 全压在一个 worker 上也毫无压力。
# 提前优化等于解决一个测量不到的问题, 却引入真实的运维复杂度。
#
# 但"到时候再说"如果没人盯就等于永远不做, 所以把触发条件写成巡检项。
CAPACITY_CPU_PERCENT = 40.0
CAPACITY_WS_CONNECTIONS = 50


async def _check_capacity_headroom() -> InvariantResult:
    """离"该上多实例负载均衡"还有多远.

    这条跟其他不变量性质不同: 别的在回答"有没有坏", 它在回答"还能撑多久"。放在同一
    张表里是因为它们共享同一个失效模式 —— 没人主动看就永远发现不了。
    """
    from app.redis_client import get_redis
    from app.services.system_metrics import collect_host_metrics

    observed: dict[str, Any] = {}
    reasons: list[str] = []

    # 复用资源监控那套采集, 不用 psutil.cpu_percent(interval=1.0) —— 后者是同步
    # 阻塞调用, 在 async 函数里会把事件循环卡满一秒, 单 worker 下所有在线用户一起
    # 卡。collect_host_metrics 用 await asyncio.sleep 取两次 /proc/stat 快照,
    # 采样期间事件循环照常转; 顺带也不引入 psutil 这个非声明依赖。
    metrics = await collect_host_metrics()
    cpu = (metrics.get("system") or {}).get("cpu_percent")
    observed["cpu_percent"] = cpu
    if cpu is not None and cpu >= CAPACITY_CPU_PERCENT:
        reasons.append(f"CPU {cpu:.0f}% ≥ {CAPACITY_CPU_PERCENT:.0f}%")

    # WS 连接数从 Redis ZSET 取 —— ConnectionManager 的内存表每个 worker 只看得到
    # 自己那一半, 多 worker 下会systematically 低估一倍。
    try:
        redis = await get_redis()
        ws_count = int(await redis.zcard("presence:online:ws") or 0)
    except Exception:
        ws_count = -1
    observed["ws_connections"] = ws_count
    if ws_count >= CAPACITY_WS_CONNECTIONS:
        reasons.append(f"在线 WS {ws_count} ≥ {CAPACITY_WS_CONNECTIONS}")

    if reasons:
        return InvariantResult(
            "capacity_headroom", "容量水位", "warn",
            "、".join(reasons)
            + " —— 该考虑 nginx least_conn + 多实例了 (单端口多 worker 的连接分配"
              "由内核决定, 不均匀且不再平衡)",
            observed,
        )
    cpu_text = f"{cpu:.0f}%" if cpu is not None else "未知"
    return InvariantResult(
        "capacity_headroom", "容量水位", "ok",
        f"CPU {cpu_text} / 在线 WS {ws_count} —— 距离需要负载均衡还有余量",
        observed,
    )


async def _check_consolidation_active() -> InvariantResult:
    """整合开着就该有产出, 否则说明它在空转."""
    from app.config import settings
    from app.db import db

    if not getattr(settings, "memory_consolidation_enabled", False):
        return InvariantResult(
            "consolidation_active", "记忆整合", "ok", "整合未启用", {}
        )

    # 只看 l3_compression: 这张表同时记 hygiene (近重复合并), 而
    # memory_consolidation_enabled 管的是簇压缩那条线。混在一起会被 hygiene 的
    # 产出掩盖, 压缩空转照样显示健康。
    rows = await db.query_raw(
        """
        SELECT
            COUNT(*)::int AS runs,
            COALESCE(SUM(archived + merged), 0)::int AS compressed
        FROM memory_consolidation_runs
        WHERE job = 'l3_compression'
          AND created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
        """
    )
    runs = int(rows[0]["runs"] or 0)
    compressed = int(rows[0]["compressed"] or 0)
    observed = {"runs_7d": runs, "compressed_7d": compressed}

    if runs == 0:
        return InvariantResult(
            "consolidation_active", "记忆整合", "violated",
            "整合已启用但近 7 天没有任何一次运行记录", observed,
        )
    if compressed == 0:
        return InvariantResult(
            "consolidation_active", "记忆整合", "warn",
            f"近 7 天跑了 {runs} 次但一条都没压缩 —— 阈值可能过严", observed,
        )
    return InvariantResult(
        "consolidation_active", "记忆整合", "ok",
        f"近 7 天 {runs} 次运行, 压缩 {compressed} 条", observed,
    )


# (key, 标题, 检查函数)。key/标题放在注册表而不是另建一张按函数名索引的映射 ——
# 后者在函数改名时会静默失配, 而失配只在检查抛异常时才暴露, 正是最需要它准的时候。
_CHECKS: tuple[tuple[str, str, Callable[[], Awaitable[InvariantResult]]], ...] = (
    ("schedule_today", "今日作息覆盖", _check_schedule_today),
    ("l2_scores_fresh", "L2 动态分新鲜度", _check_l2_scores_fresh),
    ("memory_access_logged", "记忆访问打点", _check_memory_access_logged),
    ("embeddings_complete", "向量覆盖率", _check_embeddings_complete),
    ("consolidation_active", "记忆整合", _check_consolidation_active),
    ("capacity_headroom", "容量水位", _check_capacity_headroom),
)


async def run_invariant_checks() -> list[InvariantResult]:
    """跑一遍全部不变量。单条失败降级成 error 条目, 不影响其余."""
    results: list[InvariantResult] = []
    for key, title, check in _CHECKS:
        try:
            results.append(await check())
        except Exception as exc:
            logger.warning(f"invariant {key} failed to evaluate: {exc}")
            results.append(
                InvariantResult(key, title, "error", f"检查本身出错: {exc}"[:200])
            )
    return results


async def run_and_store() -> list[InvariantResult]:
    """跑一遍并把结果写进 Redis, 供后台页面读取."""
    import json

    from app.redis_client import get_redis

    results = await run_invariant_checks()
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": [r.as_dict() for r in results],
    }
    try:
        redis = await get_redis()
        await redis.set(
            INVARIANTS_HEALTH_KEY, json.dumps(payload, ensure_ascii=False),
            ex=_INVARIANTS_TTL_S,
        )
    except Exception as exc:  # 存不下也别让巡检本身失败
        logger.warning(f"invariants: failed to persist result: {exc}")

    violated = [r.key for r in results if r.status == "violated"]
    if violated:
        # 用 error 级: 不变量被破坏意味着某个功能已经在静默失效。
        logger.error(f"[INVARIANT] violated: {', '.join(violated)}")
    return results


async def load_last_report() -> dict[str, Any]:
    """读最近一次巡检结果. 没有就返回空壳, 让页面显示"尚未运行"."""
    import json

    from app.redis_client import get_redis

    try:
        redis = await get_redis()
        raw = await redis.get(INVARIANTS_HEALTH_KEY)
    except Exception as exc:
        logger.warning(f"invariants: failed to load result: {exc}")
        raw = None

    if not raw:
        return {"checked_at": None, "results": [], "violated_count": 0}

    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {"checked_at": None, "results": [], "violated_count": 0}

    results = payload.get("results") or []
    payload["violated_count"] = sum(
        1 for r in results if r.get("status") in ("violated", "error")
    )
    return payload
