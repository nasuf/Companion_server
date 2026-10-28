"""UXM01.02 每日派生表: user_daily_facts (用户×本地日) 与 metrics_daily (日×指标×人群×维度).

两张表都是有版本号的派生缓存, 任何一天都能从原始数据重算, 不是事实来源:
- 指标口径只在 effect.definitions (以及它引用的 effect.signals 回合/回访定义) 里定义,
  这里只负责按本地日落库, 不另写一套"什么算活跃/回应/回合"。
- user_daily_facts 只存计数和时间点, 不存消息原文; 随账号级联删除。
- metrics_daily 只存人群聚合, 不含用户 ID。每行带口径版本和内部账号指纹, 读取时
  不一致就算过期 (stale), 不当作有效数字展示。

一天的计算在单个事务里完成, 按天加 advisory lock (夜间任务、启动补算、页面按需刷新
可以并发而不互相覆盖); 计算前后各读一次内部账号指纹, 中途变了就整天回滚重算。
"""

from __future__ import annotations

import json
import logging
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.effect.definitions import (
    COHORTS,
    DEFINITION_VERSION,
    METRICS,
    MIN_SAMPLE,
    NO_REPLY_WINDOW_S,
    PROACTIVE_SHORT_WINDOW_S,
    Cohort,
    conversation_population_sql,
    local_day_bounds,
    local_today,
    population_fingerprint,
    population_sql,
    proactive_events_sql,
    user_row_population_sql,
)
from app.services.effect.signals import (
    PROACTIVE_RESPONSE_WINDOW,
    SESSION_GAP,
    _BOUNDARY_LOOKAHEAD,
    _fetch_core,
    _fetch_retention,
)
from app.services.interaction.reply_telemetry import VERSION as TIMING_VERSION
from app.services.schedule_domain.sql_time import local_date_sql

logger = logging.getLogger(__name__)

# 未结束的游戏会话 (ended_at 为空) 按这么长算"游戏中", 免得一局异常不结束就永远排除。
_OPEN_GAME_SPAN = timedelta(hours=2)
# 夜间任务每次重算的已结束天数。最长的定型窗口 (次日回访) 是 24 小时, 3 天保证
# 每一天都会在全部窗口走完之后被至少重算一次。
NIGHTLY_DAYS = 3
# 自愈与启动补算的回看上限。夜间任务修最近 90 天; 启动补算从最早数据算起但有总量上限。
HEAL_LOOKBACK_DAYS = 90
MAX_BACKFILL_DAYS = 400
# 当天数据按需刷新的最短间隔。
TODAY_REFRESH_S = 600
# 夜间重算的本地时刻; jobs/scheduler.py 的 daily_facts cron 与 ops 不变量都按它判断
# "夜间那一轮应该已经算到哪里"。改时刻要两处一起改 (有测试核对)。
NIGHTLY_HOUR, NIGHTLY_MINUTE = 4, 40

_ALL_FINGERPRINT = "-"


class PopulationChanged(RuntimeError):
    """内部账号集合在一天的计算过程中变了; 这一天回滚, 由下次重算补上."""


def _utc_naive(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def is_settled(metric: str, day: date_cls, now: datetime) -> bool:
    """本地日结束后过了该指标的定型窗口, 这个数才不再变化."""
    end = local_day_bounds(day)[1]
    return _utc_naive(now) >= end + timedelta(seconds=METRICS[metric]["settle_after_s"])


def facts_settled(day: date_cls, now: datetime) -> bool:
    """事实行里最晚定型的输入是主动消息回应 (发出后 6 小时)."""
    return _utc_naive(now) >= local_day_bounds(day)[1] + PROACTIVE_RESPONSE_WINDOW


def _interval(delta: timedelta | float) -> str:
    seconds = delta.total_seconds() if isinstance(delta, timedelta) else float(delta)
    return f"{int(seconds)} seconds"


# 用户×日的计数。回合/消息串/会话/自发开口的判定只在这里写一次 (文字说明见
# definitions.METRICS), 页面与日报都读派生表, 不另写。消息串 (burst): 同一会话里连续的用户消息, 中间没有 AI 消息且相邻
# 两条间隔不超过无回复窗口; 串的最后一条之后无回复窗口内有 AI 非主动回复即算回合。
# 人工接管中、游戏进行中的消息串不计入回合和无回复。
_FACTS_SQL = """
WITH
msgs AS (
    SELECT m.id, m.conversation_id, c.user_id, m.role, m.created_at, m.metadata,
           LAG(m.role) OVER w AS prev_role, LAG(m.created_at) OVER w AS prev_at
    FROM messages m JOIN conversations c ON c.id = m.conversation_id
    -- 往前多看一天, 判断当天第一条消息是不是新串 / 新会话的开始; 往后多看一段, 跨过
    -- 午夜的消息串要拿到它真正的最后一条, 否则会在两天各算一次
    WHERE m.role IN ('user', 'assistant')
      AND m.created_at >= $1::timestamp - INTERVAL '1 day'
      AND m.created_at < $2::timestamp + $11::interval
    WINDOW w AS (PARTITION BY m.conversation_id ORDER BY m.created_at, m.id)
),
um AS (
    SELECT msgs.*,
           (prev_role IS DISTINCT FROM 'user' OR created_at - prev_at > $7::interval) AS burst_start,
           LAG(created_at) OVER (PARTITION BY conversation_id ORDER BY created_at, id) AS prev_user_at
    FROM msgs WHERE role = 'user'
),
numbered AS (
    SELECT um.*, SUM(CASE WHEN burst_start THEN 1 ELSE 0 END) OVER (
        PARTITION BY conversation_id ORDER BY created_at, id ROWS UNBOUNDED PRECEDING) AS burst_no
    FROM um
),
pro_bubbles AS (
    SELECT user_id, created_at FROM msgs
    WHERE role = 'assistant' AND metadata->>'proactive' = 'true'
),
bursts AS (
    SELECT conversation_id, user_id, burst_no, MAX(created_at) AS last_at,
           bool_or(COALESCE(metadata->>'human_takeover', 'false') = 'true') AS takeover
    FROM numbered GROUP BY conversation_id, user_id, burst_no
),
day_bursts AS (
    SELECT b.user_id, EXISTS (
        SELECT 1 FROM messages r
        WHERE r.conversation_id = b.conversation_id AND r.role = 'assistant'
          AND COALESCE(r.metadata->>'proactive', 'false') <> 'true'
          AND r.created_at > b.last_at AND r.created_at <= b.last_at + $7::interval
    ) AS replied
    FROM bursts b
    WHERE b.last_at >= $1::timestamp AND b.last_at < $2::timestamp AND NOT b.takeover
      AND NOT EXISTS (
          SELECT 1 FROM game_sessions g
          WHERE g.conversation_id = b.conversation_id AND g.started_at IS NOT NULL
            AND b.last_at >= g.started_at
            AND b.last_at <= COALESCE(g.ended_at, g.started_at + $10::interval)
      )
),
burst_stats AS (
    SELECT user_id, COUNT(*)::int AS bursts, COUNT(*) FILTER (WHERE NOT replied)::int AS unreplied
    FROM day_bursts GROUP BY user_id
),
day_msgs AS (
    SELECT user_id, COUNT(*)::int AS user_messages, MIN(created_at) AS first_message_at,
           COUNT(*) FILTER (
               WHERE prev_user_at IS NULL OR created_at - prev_user_at > $8::interval
           )::int AS sessions
    FROM um WHERE created_at >= $1::timestamp AND created_at < $2::timestamp
    GROUP BY user_id
),
pro AS (
    SELECT user_id, COUNT(*)::int AS received, COUNT(first_reply_at)::int AS answered
    FROM (__PROACTIVE_EVENTS__) e GROUP BY user_id
),
opened AS (SELECT DISTINCT user_id FROM user_daily_activity WHERE local_date = $3::date),
fb AS (
    SELECT user_id, COUNT(*)::int AS n FROM user_feedback
    WHERE created_at >= $1::timestamp AND created_at < $2::timestamp GROUP BY user_id
),
spend AS (
    SELECT user_id, SUM(cost_cny)::float8 AS cny FROM llm_usage
    WHERE user_id IS NOT NULL AND created_at >= $1::timestamp AND created_at < $2::timestamp
    GROUP BY user_id
),
ids AS (
    SELECT user_id FROM day_msgs UNION SELECT user_id FROM burst_stats
    UNION SELECT user_id FROM pro UNION SELECT user_id FROM opened
    UNION SELECT user_id FROM fb UNION SELECT user_id FROM spend
)
INSERT INTO user_daily_facts (
    user_id, local_date, opened, chatted, user_messages, turns, sessions, bursts,
    unreplied_bursts, first_message_at, spontaneous, proactive_received, proactive_answered,
    feedback_count, cost_cny, settled, definition_version, computed_at
)
SELECT
    i.user_id, $3::date, o.user_id IS NOT NULL, COALESCE(dm.user_messages, 0) > 0,
    COALESCE(dm.user_messages, 0), COALESCE(bs.bursts, 0) - COALESCE(bs.unreplied, 0),
    COALESCE(dm.sessions, 0), COALESCE(bs.bursts, 0), COALESCE(bs.unreplied, 0),
    dm.first_message_at,
    -- 自发开口: 当天第一句之前 6 小时内没有收到过主动消息 (msgs 已往前多取一天)
    CASE WHEN dm.first_message_at IS NULL THEN NULL ELSE NOT EXISTS (
        SELECT 1 FROM pro_bubbles p
        WHERE p.user_id = i.user_id
          AND p.created_at >= dm.first_message_at - $9::interval
          AND p.created_at < dm.first_message_at
    ) END,
    COALESCE(pr.received, 0), COALESCE(pr.answered, 0), COALESCE(fb.n, 0),
    COALESCE(sp.cny, 0), $6::boolean, $4, $5::timestamptz
FROM ids i
-- 注销进行中的账号不再写入。外键写入时对账号行加 KEY SHARE 锁: 改资料/改角色/改状态
-- 这类普通 UPDATE 不受影响; 只有删除账号和注销开始时的 SELECT ... FOR UPDATE 要等这一天
-- 算完 (生产实测十几毫秒)。账号删除时事实行随外键级联删除。
JOIN users u ON u.id = i.user_id AND u.status <> 'deleting'
LEFT JOIN day_msgs dm ON dm.user_id = i.user_id
LEFT JOIN burst_stats bs ON bs.user_id = i.user_id
LEFT JOIN pro pr ON pr.user_id = i.user_id
LEFT JOIN opened o ON o.user_id = i.user_id
LEFT JOIN fb ON fb.user_id = i.user_id
LEFT JOIN spend sp ON sp.user_id = i.user_id
"""


def _facts_sql() -> str:
    events = proactive_events_sql("$1::timestamp", "$2::timestamp", "$9::interval", "all")
    return _FACTS_SQL.replace("__PROACTIVE_EVENTS__", events)


def _timing_sql(cohort: Cohort) -> str:
    """带 server-turn-timing-v1 的回合; 同一回合的多个气泡按首气泡时刻去重."""
    return f"""
        WITH t AS (
            SELECT DISTINCT ON (m.conversation_id, d->>'first_bubble_at')
                m.created_at AS anchor_at,
                COALESCE(d->>'reply_path', 'unknown') AS reply_path,
                CASE WHEN pg_input_is_valid(d->>'turn_received_at', 'timestamptz')
                       AND pg_input_is_valid(d->>'first_bubble_at', 'timestamptz')
                     THEN EXTRACT(EPOCH FROM ((d->>'first_bubble_at')::timestamptz
                                              - (d->>'turn_received_at')::timestamptz)) * 1000
                END AS first_ms,
                CASE WHEN jsonb_typeof(d->'generation_ms') = 'number'
                       AND jsonb_typeof(d->'in_graph_delay_ms') = 'number'
                     THEN (d->>'generation_ms')::float8 - (d->>'in_graph_delay_ms')::float8
                END AS processing_ms
            FROM messages m CROSS JOIN LATERAL (SELECT m.metadata->'response_diagnostics' AS d) x
            WHERE m.role = 'assistant'
              -- 多看一段: 跨午夜的多气泡回合按它最早的气泡归日, 只算一次
              AND m.created_at >= $1::timestamp - INTERVAL '1 hour'
              AND m.created_at < $2::timestamp + INTERVAL '1 hour'
              AND d->>'timing_definition_version' = $3
              AND {conversation_population_sql('m.conversation_id', cohort)}
            ORDER BY m.conversation_id, d->>'first_bubble_at', m.created_at
        )
        SELECT COALESCE(reply_path, '') AS dimension_value, GROUPING(reply_path) = 1 AS overall,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY first_ms) FILTER (WHERE first_ms >= 0) AS first_p50,
            percentile_cont(0.9) WITHIN GROUP (ORDER BY first_ms) FILTER (WHERE first_ms >= 0) AS first_p90,
            COUNT(first_ms) FILTER (WHERE first_ms >= 0)::int AS first_n,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY processing_ms) FILTER (WHERE processing_ms >= 0) AS proc_p50,
            percentile_cont(0.9) WITHIN GROUP (ORDER BY processing_ms) FILTER (WHERE processing_ms >= 0) AS proc_p90,
            COUNT(processing_ms) FILTER (WHERE processing_ms >= 0)::int AS proc_n
        FROM t WHERE anchor_at >= $1::timestamp AND anchor_at < $2::timestamp
        GROUP BY GROUPING SETS ((), (reply_path))
    """


def _proactive_sql(cohort: Cohort) -> str:
    events = proactive_events_sql("$1::timestamp", "$2::timestamp", "$3::interval", cohort)
    return f"""
        SELECT COALESCE(trigger_type, '') AS dimension_value, GROUPING(trigger_type) = 1 AS overall,
            COUNT(*)::int AS sent, COUNT(first_reply_at)::int AS answered,
            COUNT(*) FILTER (WHERE first_reply_at <= sent_at + $4::interval)::int AS answered_short
        FROM ({events}) e GROUP BY GROUPING SETS ((), (trigger_type))
    """


def _row(metric: str, cohort: Cohort, *, settled: bool, numerator: float | None = None,
         denominator: float | None = None, value: float | None = None,
         sample_size: int | None = None, dimension: str = "", dimension_value: str = "") -> dict:
    kind = METRICS[metric]["kind"]
    if kind in ("rate", "mean") and value is None and denominator:
        value = round(float(numerator or 0) / float(denominator), 6)
    return {"metric": metric, "cohort": cohort, "dimension": dimension,
            "dimension_value": dimension_value, "numerator": numerator,
            "denominator": denominator, "value": value, "sample_size": sample_size,
            "settled": settled}


async def _cohort_rows(tx: Any, day: date_cls, cohort: Cohort, now: datetime) -> list[dict]:
    start, end = local_day_bounds(day)
    s, e = start.isoformat(), end.isoformat()
    settled = {key: is_settled(key, day, now) for key in METRICS}
    facts = (await tx.query_raw(
        f"""
        SELECT COUNT(*) FILTER (WHERE chatted)::int AS chatted,
               COUNT(*) FILTER (WHERE opened)::int AS opened,
               COALESCE(SUM(user_messages), 0)::int AS user_messages,
               COALESCE(SUM(turns), 0)::int AS turns,
               COALESCE(SUM(bursts), 0)::int AS bursts,
               COALESCE(SUM(unreplied_bursts), 0)::int AS unreplied,
               COALESCE(SUM(cost_cny), 0)::float8 AS cost
        FROM user_daily_facts f
        WHERE f.local_date = $1::date AND {population_sql('f.user_id', cohort)}
        """,
        day.isoformat(),
    ))[0]
    new_users = (await tx.query_raw(
        f"""SELECT COUNT(*)::int AS n FROM users u
            WHERE u.created_at >= $1::timestamp AND u.created_at < $2::timestamp
              AND {user_row_population_sql('u', cohort)}""",
        s, e,
    ))[0]["n"]
    core = await _fetch_core(start, end, cohort, client=tx)
    retention = await _fetch_retention(start, end, cohort, client=tx)
    chatted = int(facts["chatted"] or 0)
    rows = [
        _row("chat_active_users", cohort, value=chatted, settled=settled["chat_active_users"]),
        _row("opened_users", cohort, value=int(facts["opened"] or 0), settled=settled["opened_users"]),
        _row("new_users", cohort, value=int(new_users or 0), settled=settled["new_users"]),
        _row("user_messages", cohort, value=int(facts["user_messages"] or 0), settled=settled["user_messages"]),
        _row("turns", cohort, value=int(facts["turns"] or 0), settled=settled["turns"]),
        _row("turns_per_active_user", cohort, numerator=int(facts["turns"] or 0),
             denominator=chatted, settled=settled["turns_per_active_user"]),
        _row("next_day_return", cohort, numerator=int(retention.get("returned") or 0),
             denominator=int(retention.get("active_users") or 0), settled=settled["next_day_return"]),
        _row("continuation", cohort, numerator=int(core.get("continued") or 0),
             denominator=int(core.get("turns") or 0), settled=settled["continuation"]),
        _row("no_reply_10m", cohort, numerator=int(facts["unreplied"] or 0),
             denominator=int(facts["bursts"] or 0), settled=settled["no_reply_10m"]),
        _row("cost_per_active_user", cohort, numerator=round(float(facts["cost"] or 0), 6),
             denominator=chatted, settled=settled["cost_per_active_user"]),
    ]
    for r in await tx.query_raw(
        _proactive_sql(cohort), s, e, _interval(PROACTIVE_RESPONSE_WINDOW),
        _interval(PROACTIVE_SHORT_WINDOW_S),
    ):
        dim = {} if r["overall"] else {"dimension": "trigger_type", "dimension_value": r["dimension_value"]}
        sent = int(r["sent"] or 0)
        rows += [
            _row("proactive_events", cohort, value=sent, settled=settled["proactive_events"], **dim),
            _row("proactive_response", cohort, numerator=int(r["answered"] or 0), denominator=sent,
                 settled=settled["proactive_response"], **dim),
            _row("proactive_response_3h", cohort, numerator=int(r["answered_short"] or 0),
                 denominator=sent, settled=settled["proactive_response_3h"], **dim),
        ]
    # GROUPING SETS ((), ...) 在没有任何输入时也返回总计行: 真实为零 / 无样本会被写出来。
    for r in await tx.query_raw(_timing_sql(cohort), s, e, TIMING_VERSION):
        dim = {} if r["overall"] else {"dimension": "reply_path", "dimension_value": r["dimension_value"]}
        for metric, value, n in (
            ("reply_first_bubble_ms_p50", r["first_p50"], r["first_n"]),
            ("reply_first_bubble_ms_p90", r["first_p90"], r["first_n"]),
            ("reply_processing_ms_p50", r["proc_p50"], r["proc_n"]),
            ("reply_processing_ms_p90", r["proc_p90"], r["proc_n"]),
        ):
            rows.append(_row(metric, cohort, value=None if value is None else round(float(value), 3),
                             sample_size=int(n or 0), settled=settled[metric], **dim))
    return rows


async def compute_day(
    day: date_cls, *, now: datetime | None = None, skip_if_fresher_than: float | None = None,
    wait_for_lock: bool = True,
) -> dict[str, Any]:
    """重算一个本地日的事实行和两种人群的指标行 (先删后写, 幂等).

    skip_if_fresher_than: 拿到当天的锁之后, 若已有当前口径、当前人群的结果且算出不到
    这么多秒, 就不重算 —— 多个管理员同时打开页面时只算一次。
    wait_for_lock=False: 这一天正被别处重算时直接返回, 不排队 (页面请求用, 不能被
    夜间任务卡住)。
    """
    import time

    from app.db import db

    started = time.perf_counter()

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    computed_at = current.isoformat()
    start, end = local_day_bounds(day)
    # READ COMMITTED + 按天 advisory lock: 拿到锁之后的每条语句都能看到前一个持锁者
    # 已提交的结果。REPEATABLE READ 的快照在第一条语句 (就是拿锁那条) 开始时就定了,
    # 排队等锁的那个会用旧快照去删别人刚写的行, 触发冲突或主键重复。
    async with db.tx(max_wait=timedelta(seconds=10), timeout=timedelta(seconds=60)) as tx:
        lock = f"beta-daily-facts:{day.isoformat()}"
        if wait_for_lock:
            await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", lock)
        else:
            got = await tx.query_raw(
                "SELECT pg_try_advisory_xact_lock(hashtextextended($1, 0)) AS ok", lock,
            )
            if not got[0]["ok"]:
                return {"day": day.isoformat(), "skipped": "locked"}
        fingerprint = await population_fingerprint(tx)
        if skip_if_fresher_than is not None:
            fresh = await tx.query_raw(
                """
                SELECT 1 FROM metrics_daily
                WHERE local_date = $1::date AND metric = 'chat_active_users' AND dimension = ''
                  AND cohort = 'external' AND definition_version = $2 AND population_fingerprint = $3
                  AND computed_at > $4::timestamptz - make_interval(secs => $5)
                """,
                day.isoformat(), DEFINITION_VERSION, fingerprint, computed_at, float(skip_if_fresher_than),
            )
            if fresh:
                return {"day": day.isoformat(), "skipped": True}
        await tx.execute_raw("DELETE FROM user_daily_facts WHERE local_date = $1::date", day.isoformat())
        facts = await tx.execute_raw(
            _facts_sql(), start.isoformat(), end.isoformat(), day.isoformat(), DEFINITION_VERSION,
            computed_at, facts_settled(day, current), _interval(NO_REPLY_WINDOW_S),
            _interval(SESSION_GAP), _interval(PROACTIVE_RESPONSE_WINDOW), _interval(_OPEN_GAME_SPAN),
            _interval(_BOUNDARY_LOOKAHEAD),
        )
        fingerprints = {"external": fingerprint, "all": _ALL_FINGERPRINT}
        rows: list[dict] = []
        for cohort in COHORTS:
            for row in await _cohort_rows(tx, day, cohort, current):
                rows.append({**row, "population_fingerprint": fingerprints[cohort]})
        # 计算期间有人改了内部账号: 外部口径的数不属于开头读到的指纹, 整天回滚, 下次重算。
        if await population_fingerprint(tx) != fingerprint:
            raise PopulationChanged(day.isoformat())
        await tx.execute_raw("DELETE FROM metrics_daily WHERE local_date = $1::date", day.isoformat())
        await tx.execute_raw(
            """
            INSERT INTO metrics_daily (
                local_date, metric, cohort, dimension, dimension_value, numerator, denominator,
                value, sample_size, settled, definition_version, population_fingerprint, computed_at
            )
            SELECT $1::date, r.metric, r.cohort, r.dimension, r.dimension_value, r.numerator,
                   r.denominator, r.value, r.sample_size, r.settled, $2, r.population_fingerprint,
                   $3::timestamptz
            FROM jsonb_to_recordset($4::jsonb) AS r(
                metric text, cohort text, dimension text, dimension_value text,
                numerator float8, denominator float8, value float8, sample_size int,
                settled boolean, population_fingerprint text)
            """,
            day.isoformat(), DEFINITION_VERSION, computed_at, json.dumps(rows),
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("[DAILY-FACTS] day=%s facts=%d metrics=%d ms=%.0f",
                day.isoformat(), int(facts or 0), len(rows), elapsed_ms)
    return {"day": day.isoformat(), "facts": int(facts or 0), "metrics": len(rows),
            "window_utc": [start.isoformat(), end.isoformat()], "ms": round(elapsed_ms)}


def point_state(metric: str, day: date_cls, row: dict | None, *, expected_fingerprint: str) -> tuple[str, float | None]:
    """一个数据点的展示状态和可展示的值. 页面只按 state 渲染, 不自己推断.

    not_collected  该日早于这类原始数据开始采集的日期 (未采集)
    not_computed   还没有派生结果 (未运行)
    stale          结果来自旧口径或旧的内部账号集合, 后台正在重算
    no_sample      分母 / 样本为 0
    insufficient_sample  分母或样本少于 MIN_SAMPLE: 只给分子分母, 不给百分比或分位数
    unsettled      窗口还没走完; 比率不给值 (必然偏低), 计数和分位数照给
    ok             可直接展示; 计数为 0 时是真实为零
    """
    spec = METRICS[metric]
    since = spec.get("collected_since")
    if since and day < date_cls.fromisoformat(since):
        return "not_collected", None
    if row is None:
        return "not_computed", None
    if row["definition_version"] != DEFINITION_VERSION or row["population_fingerprint"] != expected_fingerprint:
        return "stale", None
    kind, value = spec["kind"], row["value"]
    if kind in ("rate", "mean"):
        denominator = row["denominator"] or 0
        if not denominator:
            return "no_sample", None
        if kind == "rate" and not row["settled"]:
            return "unsettled", None
        if kind == "rate" and denominator < MIN_SAMPLE:
            return "insufficient_sample", None
    elif kind == "quantile":
        sample = row["sample_size"] or 0
        if not sample:
            return "no_sample", None
        if sample < MIN_SAMPLE:
            return "insufficient_sample", None
    return ("ok" if row["settled"] else "unsettled"), value


async def _heal_in_background(days: list[date_cls]) -> None:
    """读取时发现过期的日子交给后台重算; 同一时刻只有一个重算在跑."""
    from app.services.runtime.distributed_lock import DistributedLockNotAcquired, distributed_lock

    try:
        async with distributed_lock("beta-daily-facts-heal", ttl_s=900):
            await compute_days(days)
    except DistributedLockNotAcquired:
        return
    except Exception as exc:
        logger.warning("daily facts background heal failed type=%s", type(exc).__name__)


async def read_series(
    start: date_cls, end: date_cls, *, cohort: Cohort, metrics: list[str], dimension: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """[start, end] 的逐日指标. 今天和昨天按需刷新 (10 分钟内只算一次), 过期日子后台重算."""
    from app.db import db
    from app.services.runtime.tasks import fire_background

    today = local_today(now)
    for day in (today - timedelta(days=1), today):
        if start <= day <= end:
            try:
                await compute_day(day, now=now, skip_if_fresher_than=TODAY_REFRESH_S, wait_for_lock=False)
            except Exception as exc:
                # 刷新失败不挡读取: 已有的结果照常返回, 缺的那天显示"未运行", 后台再补。
                logger.warning("daily facts on-demand refresh failed day=%s type=%s", day, type(exc).__name__)
    expected = await population_fingerprint() if cohort == "external" else _ALL_FINGERPRINT
    rows = await db.query_raw(
        """
        SELECT local_date::text AS day, metric, dimension, dimension_value, numerator, denominator,
               value, sample_size, settled, definition_version, population_fingerprint,
               EXTRACT(EPOCH FROM computed_at)::float8 AS computed_epoch
        FROM metrics_daily
        WHERE local_date BETWEEN $1::date AND $2::date AND cohort = $3
          AND ((dimension = $4 AND metric = ANY($5::text[]))
               OR (dimension = '' AND (metric = ANY($5::text[]) OR metric = 'chat_active_users')))
        """,
        start.isoformat(), end.isoformat(), cohort, dimension, metrics,
    )
    # 每天的总计行说明"这一天已经算过、用的是哪个口径"; 维度明细里缺的格子据此判为
    # 当天没有这一类数据 (真实为零 / 无样本), 而不是"未运行"。
    overall = {(r["metric"], r["day"]): r for r in rows if r["dimension"] == ""}
    day_rows = {day: r for (metric, day), r in overall.items() if metric == "chat_active_users"}
    rows = [r for r in rows if r["dimension"] == dimension and r["metric"] in metrics]
    by_key = {(r["metric"], r["day"], r["dimension_value"]): r for r in rows}
    # 整个窗口没有任何分类时也逐日给一个占位点, 页面分得清"没算"和"没有这类数据"。
    values = (sorted({r["dimension_value"] for r in rows}) or [""]) if dimension else [""]
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    series: dict[str, list[dict]] = {}
    for metric in metrics:
        points = []
        for day in days:
            for dimension_value in values:
                row = by_key.get((metric, day.isoformat(), dimension_value))
                if row is None and dimension and day.isoformat() in day_rows:
                    computed = day_rows[day.isoformat()]
                    total = overall.get((metric, day.isoformat()))
                    row = {"value": 0 if METRICS[metric]["kind"] == "count" else None,
                           "numerator": 0, "denominator": 0, "sample_size": 0,
                           # 定型与否跟当天这一指标的总计行一致, 不按现在的时间推断
                           "settled": bool(total and total["settled"]),
                           "definition_version": computed["definition_version"],
                           "population_fingerprint": computed["population_fingerprint"]}
                state, value = point_state(metric, day, row, expected_fingerprint=expected)
                points.append({
                    "date": day.isoformat(), "dimension_value": dimension_value, "state": state,
                    "value": value,
                    "numerator": row["numerator"] if row and state != "stale" else None,
                    "denominator": row["denominator"] if row and state != "stale" else None,
                    "sample_size": row["sample_size"] if row and state != "stale" else None,
                })
        series[metric] = points
    # 缺失、口径过期或该定型未定型的日子交给后台重算; 判定只有 days_needing_compute 一处。
    stale = await days_needing_compute(start, min(end, today), now=now)
    if stale:
        fire_background(_heal_in_background(stale))
    computed = [r["computed_epoch"] for r in rows if r["definition_version"] == DEFINITION_VERSION]
    return {
        "start": start.isoformat(), "end": end.isoformat(), "cohort": cohort, "dimension": dimension,
        "series": series, "recomputing": [d.isoformat() for d in stale],
        "computed_at": (datetime.fromtimestamp(max(computed), tz=timezone.utc).isoformat()
                        if computed else None),
    }


async def days_needing_compute(start: date_cls, end: date_cls, *, now: datetime | None = None) -> list[date_cls]:
    """[start, end] 里缺失、口径过期、人群过期, 或窗口已走完却仍未定型的日子."""
    from app.db import db

    if end < start:
        return []
    fingerprint = await population_fingerprint()
    settled_before = local_today(now) - timedelta(days=2)
    rows = await db.query_raw(
        """
        SELECT d::date::text AS day FROM generate_series($1::date, $2::date, INTERVAL '1 day') d
        WHERE NOT EXISTS (
                SELECT 1 FROM metrics_daily x
                WHERE x.local_date = d::date AND x.metric = 'chat_active_users' AND x.dimension = ''
                  AND x.cohort = 'external' AND x.definition_version = $3
                  AND x.population_fingerprint = $4)
           OR NOT EXISTS (
                SELECT 1 FROM metrics_daily x
                WHERE x.local_date = d::date AND x.metric = 'chat_active_users' AND x.dimension = ''
                  AND x.cohort = 'all' AND x.definition_version = $3)
           OR (d::date <= $5::date AND EXISTS (
                SELECT 1 FROM metrics_daily x WHERE x.local_date = d::date AND NOT x.settled))
        ORDER BY 1
        """,
        start.isoformat(), end.isoformat(), DEFINITION_VERSION, fingerprint, settled_before.isoformat(),
    )
    return [date_cls.fromisoformat(r["day"]) for r in rows]


async def earliest_data_day() -> date_cls | None:
    """原始数据最早的本地日; 回填从这里算起."""
    from app.db import db

    rows = await db.query_raw(
        f"""
        SELECT LEAST(
            (SELECT {local_date_sql("MIN(created_at)")} FROM users),
            (SELECT {local_date_sql("MIN(created_at)")} FROM messages),
            (SELECT MIN(local_date) FROM user_daily_activity)
        )::text AS day
        """
    )
    return date_cls.fromisoformat(rows[0]["day"]) if rows and rows[0]["day"] else None


async def compute_days(days: list[date_cls], *, now: datetime | None = None) -> dict[str, Any]:
    """逐天重算; 一天失败不影响其余, 失败的日子汇总返回 (调用方决定是否上报)."""
    done, failed = [], []
    for day in days:
        try:
            try:
                await compute_day(day, now=now)
            except PopulationChanged:
                # 恰好有人在算这一天时改了内部账号; 新集合已提交, 重算一次即可。
                await compute_day(day, now=now)
            done.append(day.isoformat())
        except Exception as exc:
            logger.warning("daily facts compute failed day=%s type=%s", day, type(exc).__name__)
            failed.append(day.isoformat())
    return {"computed": done, "failed": failed}


async def run_nightly(*, now: datetime | None = None) -> dict[str, Any]:
    """夜间任务: 最近 3 个已结束的日子必算, 再加今天, 以及 90 天内缺失或过期的日子."""
    today = local_today(now)
    recent = [today - timedelta(days=back) for back in range(NIGHTLY_DAYS, 0, -1)]
    heal = await days_needing_compute(today - timedelta(days=HEAL_LOOKBACK_DAYS), today, now=now)
    # 从近往远算: 口径或内部账号变了、90 天都要重算时, 页面最常看的今天和昨天先就绪。
    days = sorted(set(recent) | set(heal) | {today}, reverse=True)
    result = await compute_days(days, now=now)
    if result["failed"]:
        raise RuntimeError(f"daily facts failed for {len(result['failed'])} day(s): {result['failed'][:5]}")
    return result


async def run_catchup(*, now: datetime | None = None) -> dict[str, Any]:
    """启动补算: 从最早数据起, 只算缺失或过期的日子; 首次上线即完成回填, 之后几乎无事可做."""
    first = await earliest_data_day()
    today = local_today(now)
    if first is None:
        return {"computed": [], "failed": []}
    first = max(first, today - timedelta(days=MAX_BACKFILL_DAYS - 1))
    pending = await days_needing_compute(first, today, now=now)
    result = await compute_days(sorted(pending, reverse=True), now=now)
    if result["failed"]:
        raise RuntimeError(f"daily facts catch-up failed for {len(result['failed'])} day(s)")
    return result
