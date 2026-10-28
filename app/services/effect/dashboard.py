"""Admin dashboard reads: derived facts, weighted windows and bounded queries.

No message text leaves this module. Raw timing is the explicit exception to
materialized reads and uses a shared bounded cache, never a per-day p90 mean.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from uuid import uuid4
from datetime import date, datetime, timedelta, timezone

from app.services.effect import definitions as defs
from app.services.effect.daily_facts import _timing_sql, timing_cte
from app.services.interaction.reply_telemetry import VERSION as TIMING_VERSION

logger = logging.getLogger(__name__)

CACHE_SECONDS = 600
MAX_TIMING_READS_PER_DAY = 500
SORTS = {
    "username": "username",
    "registered_at": "registered_at",
    "last_active_day": "last_active_day",
    "active_days": "active_days",
    "user_messages": "user_messages",
    "turns": "turns",
    "feedback_count": "feedback_count",
    "cost_cny": "cost_cny",
}


def window(days: int, end: str | None = None) -> tuple[date, date]:
    if not 1 <= days <= 92:
        raise ValueError("时间窗须为1–92天")
    last = date.fromisoformat(end) if end else defs.local_today()
    if last < date(2026, 1, 1) or last > defs.local_today():
        raise ValueError("日期超出可查询范围")
    return last - timedelta(days=days - 1), last


async def cached_timing(start: date, end: date, cohort: str) -> dict:
    from app.db import db
    from app.redis_client import get_redis

    redis = await get_redis()
    if redis is None:
        raise RuntimeError("timing cache unavailable")
    fp = await defs.population_fingerprint() if cohort == "external" else "-"
    key = f"stats:timing:{defs.DEFINITION_VERSION}:{cohort}:{fp}:{start}:{end}"
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)
    # Prevent concurrent admins/workers repeatedly scanning the same window.
    token = uuid4().hex
    if not await redis.set(key + ":lock", token, nx=True, ex=15):
        raise RuntimeError("timing computation in progress")
    try:
        budget = f"stats:timing:budget:{defs.local_today()}"
        n = await redis.incr(budget)
        if n == 1:
            await redis.expire(budget, 172800)
        if n > MAX_TIMING_READS_PER_DAY:
            raise RuntimeError("timing daily read limit reached")
        first, _ = defs.local_day_bounds(start)
        _, last = defs.local_day_bounds(end)
        async with db.tx(timeout=timedelta(seconds=5)) as tx:
            await tx.execute_raw("SET LOCAL statement_timeout = '3s'")
            summary = await tx.query_raw(
                _timing_sql(cohort), first.isoformat(), last.isoformat(), TIMING_VERSION
            )
            histogram = await tx.query_raw(
                timing_cte(cohort)
                + """
                SELECT width_bucket(first_ms,ARRAY[0,1000,3000,5000,10000,20000,30000,60000]::float8[]) AS bucket,
                  COUNT(*)::int AS count,
                  COUNT(*) FILTER(WHERE aggregation_ms>=0 AND delay_ms>=0 AND processing_ms>=0)::int AS measured_stages,
                  AVG(aggregation_ms) FILTER(WHERE aggregation_ms>=0 AND delay_ms>=0 AND processing_ms>=0) AS aggregation_ms,
                  AVG(delay_ms) FILTER(WHERE aggregation_ms>=0 AND delay_ms>=0 AND processing_ms>=0) AS delay_ms,
                  AVG(processing_ms) FILTER(WHERE aggregation_ms>=0 AND delay_ms>=0 AND processing_ms>=0) AS processing_ms
                FROM t WHERE anchor_at >= $1::timestamp AND anchor_at < $2::timestamp AND first_ms>=0
                GROUP BY 1 ORDER BY 1""",
                first.isoformat(),
                last.isoformat(),
                TIMING_VERSION,
            )
            from app.services.effect.daily_facts import (
                burst_details_cte,
                burst_query_params,
            )

            args = list(burst_query_params(start, datetime.now(timezone.utc)))
            args[0], args[1] = first.isoformat(), last.isoformat()
            coverage = (
                await tx.query_raw(
                    burst_details_cte()
                    + f"""
                SELECT COUNT(*)::int AS turns,
                  COUNT(*) FILTER(WHERE r.d->>'timing_definition_version'=$12
                    AND pg_input_is_valid(r.d->>'first_bubble_at','timestamptz')
                    AND pg_input_is_valid(r.d->>'turn_received_at','timestamptz')
                    AND CASE WHEN pg_input_is_valid(r.d->>'first_bubble_at','timestamptz')
                              AND pg_input_is_valid(r.d->>'turn_received_at','timestamptz')
                      THEN (r.d->>'first_bubble_at')::timestamptz >= (r.d->>'turn_received_at')::timestamptz
                      ELSE false END)::int AS measured
                FROM day_bursts b CROSS JOIN bind_types binds
                JOIN users u ON u.id=b.user_id AND u.status <> 'deleting'
                LEFT JOIN LATERAL (SELECT metadata->'response_diagnostics' AS d FROM messages m
                  WHERE m.conversation_id=b.conversation_id AND m.role='assistant'
                    AND COALESCE(m.metadata->>'proactive','false')<>'true'
                    AND m.created_at>b.last_at AND m.created_at<=b.last_at+$7::interval
                  ORDER BY m.created_at,m.id LIMIT 1) r ON true
                WHERE b.replied AND {defs.user_row_population_sql("u", cohort)}
            """,
                    *args,
                    TIMING_VERSION,
                )
            )[0]
            result = {
                "summary": summary,
                "histogram": histogram,
                "coverage": coverage,
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }
        if cohort == "external" and await defs.population_fingerprint() != fp:
            raise RuntimeError("population changed during timing query")
        await redis.set(key, json.dumps(result), ex=CACHE_SECONDS)
        return result
    finally:
        try:
            await redis.eval(
                "if redis.call('GET',KEYS[1])==ARGV[1] then return redis.call('DEL',KEYS[1]) end return 0",
                1,
                key + ":lock",
                token,
            )
        except Exception as exc:
            # Lock has a short TTL. Do not replace the original query/cache error.
            logger.warning("stats timing unlock failed type=%s", type(exc).__name__)


def aggregate_points(
    metric: str,
    points: list[dict],
    *,
    distinct: int | None = None,
    quantile: dict | None = None,
) -> dict:
    """Window totals, never daily averages. Pending days remain visibly pending."""
    spec = defs.METRICS[metric]
    settled = bool(points) and all(
        p.get("settled", p["state"] != "unsettled") for p in points
    )
    base = {
        "value": None,
        "numerator": None,
        "denominator": None,
        "sample_size": None,
        "state": "not_computed",
        "settled": settled,
    }
    if not points:
        return base
    for state in ("failed", "stale", "not_computed", "not_collected", "expired"):
        if any(p["state"] == state for p in points):
            return {**base, "state": state}
    kind = spec["kind"]
    if kind == "quantile":
        if quantile is None:
            return {**base, "state": "failed"}
        prefix = "first" if metric.startswith("reply_first") else "proc"
        n = int(quantile[prefix + "_n"])
        value = quantile[prefix + "_" + metric.rsplit("_", 1)[1]]
        state = (
            "no_sample"
            if n == 0
            else "insufficient_sample"
            if n < defs.MIN_SAMPLE
            else "ok"
            if settled
            else "unsettled"
        )
        return {
            **base,
            "state": state,
            "sample_size": n,
            "value": value if n >= defs.MIN_SAMPLE else None,
        }
    if kind == "max":
        return {
            **base,
            "state": "ok" if settled else "unsettled",
            "value": max(
                (p["value"] for p in points if p["value"] is not None), default=None
            ),
        }
    if kind == "count":
        value = (
            distinct
            if distinct is not None
            else sum(p["value"] for p in points if p["value"] is not None)
        )
        return {**base, "state": "ok" if settled else "unsettled", "value": value}
    numerator = sum(p.get("numerator") or 0 for p in points)
    denominator = sum(p.get("denominator") or 0 for p in points)
    if denominator == 0:
        state = "no_sample"
    elif kind == "rate" and not settled:
        state = "unsettled"
    elif kind == "rate" and denominator < defs.MIN_SAMPLE:
        state = "insufficient_sample"
    else:
        state = "ok" if settled else "unsettled"
    known = denominator > 0 and (kind != "rate" or state == "ok")
    return {
        **base,
        "numerator": numerator,
        "denominator": denominator,
        "sample_size": denominator,
        "state": state,
        "value": numerator / denominator * spec.get("scale", 1) if known else None,
    }


async def summarize(start: date, end: date, cohort: str, series: dict) -> dict:
    from app.db import db

    counts = (
        await db.query_raw(
            f"""SELECT COUNT(DISTINCT user_id) FILTER(WHERE chatted)::int AS chatted,
        COUNT(DISTINCT user_id) FILTER(WHERE opened)::int AS opened
        FROM user_daily_facts f WHERE local_date BETWEEN $1::date AND $2::date
        AND definition_version=$3 AND {defs.population_sql("f.user_id", cohort)}""",
            str(start),
            str(end),
            defs.DEFINITION_VERSION,
        )
    )[0]
    timing = None
    if any(defs.METRICS[m]["kind"] == "quantile" for m in series):
        try:
            timing = next(
                r
                for r in (await cached_timing(start, end, cohort))["summary"]
                if r["overall"]
            )
        except Exception:
            pass  # state=failed, not an invented zero percentile
    return {
        metric: aggregate_points(
            metric,
            points,
            distinct=counts["chatted"]
            if metric == "chat_active_users"
            else counts["opened"]
            if metric == "opened_users"
            else None,
            quantile=timing,
        )
        for metric, points in series.items()
    }


def _user_filters(cohort: str) -> str:
    return f"""u.status <> 'deleting' AND {defs.user_row_population_sql("u", cohort)}
        AND $1::timestamp <= $2::timestamp AND u.created_at < $2::timestamp
        AND ($3::timestamp IS NULL OR (u.created_at >= $3::timestamp AND u.created_at < $3::timestamp + INTERVAL '7 days'))
        AND ($4 = '' OR COALESCE(u.signup_platform,'unknown')=$4)
        AND ($5 = '' OR u.id ILIKE '%'||$5||'%' OR u.username ILIKE '%'||$5||'%' OR u.display_name ILIKE '%'||$5||'%')"""


async def users(
    start: date,
    end: date,
    *,
    cohort: str,
    signup_week: str = "",
    platform: str = "",
    status: str = "",
    q: str = "",
    sort: str = "user_messages",
    order: str = "desc",
    page: int = 1,
    page_size: int = 25,
    export: bool = False,
) -> dict:
    from app.db import db

    if (
        sort not in SORTS
        or order not in ("asc", "desc")
        or status not in ("", "not_started", "new", "active", "silent", "churned")
    ):
        raise ValueError("不支持的排序或状态")
    if (
        not 1 <= page_size <= 100
        or not 1 <= page <= 100000
        or len(q) > 100
        or len(platform) > 64
    ):
        raise ValueError("分页或筛选超过上限")
    week = date.fromisoformat(signup_week) if signup_week else None
    if week and week.weekday() != 0:
        raise ValueError("注册周须为星期一")
    coverage = await fact_coverage(start, end)
    if status and coverage["state"] != "ok":
        return {
            "coverage": coverage,
            "items": [],
            "total": None,
            "page": page,
            "page_size": page_size,
            "truncated": False,
            "start": str(start),
            "end": str(end),
            "cohort": cohort,
            "definition_version": defs.DEFINITION_VERSION,
        }
    first, _ = defs.local_day_bounds(start)
    _, stop = defs.local_day_bounds(end)
    week_start = defs.local_day_bounds(week)[0].isoformat() if week else None
    params = [
        first.isoformat(),
        stop.isoformat(),
        week_start,
        platform,
        q,
        defs.DEFINITION_VERSION,
        str(start),
        str(end),
        status,
    ]
    cte = f"""WITH base AS (
      SELECT u.id, COALESCE(NULLIF(u.display_name,''),u.username) AS username, u.created_at AS registered_at, u.signup_platform AS platform,
        {defs.internal_user_sql("u")} AS internal,
        MAX(f.local_date) FILTER(WHERE f.chatted) AS last_chat_day,
        MAX(f.local_date) FILTER(WHERE f.chatted OR f.opened) AS last_active_day,
        COUNT(*) FILTER(WHERE f.local_date >= $7::date AND f.chatted)::int AS active_days,
        COALESCE(SUM(f.user_messages) FILTER(WHERE f.local_date >= $7::date),0)::int AS user_messages,
        COALESCE(SUM(f.turns) FILTER(WHERE f.local_date >= $7::date),0)::int AS turns,
        COALESCE(SUM(f.proactive_received) FILTER(WHERE f.local_date >= $7::date),0)::int AS proactive_received,
        COALESCE(SUM(f.proactive_answered) FILTER(WHERE f.local_date >= $7::date),0)::int AS proactive_answered,
        COALESCE(SUM(f.feedback_count) FILTER(WHERE f.local_date >= $7::date),0)::int AS feedback_count,
        COALESCE(SUM(f.cost_cny) FILTER(WHERE f.local_date >= $7::date),0)::float8 AS cost_cny,
        bool_and(f.definition_version=$6) FILTER(WHERE f.local_date >= $7::date) AS current_definition,
        MAX(f.computed_at) AS computed_at
      FROM users u LEFT JOIN user_daily_facts f ON f.user_id=u.id AND f.local_date <= $8::date
      WHERE {_user_filters(cohort)} GROUP BY u.id
    ), labeled AS (SELECT *, CASE
       WHEN last_chat_day IS NULL THEN 'not_started'
       WHEN registered_at >= $2::timestamp-INTERVAL '7 days' THEN 'new'
       WHEN last_chat_day >= $8::date-2 THEN 'active'
       WHEN last_active_day >= $8::date-6 THEN 'silent' ELSE 'churned' END AS status
      FROM base), selected AS (SELECT * FROM labeled WHERE ($9='' OR status=$9))"""
    total = (
        await db.query_raw(cte + " SELECT COUNT(*)::int AS n FROM selected", *params)
    )[0]["n"]
    rows = await db.query_raw(
        cte
        + f" SELECT * FROM selected ORDER BY {SORTS[sort]} {order} NULLS LAST, id ASC LIMIT $10 OFFSET $11",
        *params,
        5000 if export else page_size,
        0 if export else (page - 1) * page_size,
    )
    for row in rows:
        row["state"] = (
            coverage["state"]
            if coverage["state"] != "ok"
            else "stale"
            if row["current_definition"] is False
            else "ok"
        )
        if row["state"] != "ok":
            row["status"] = "unknown"
            for field in (
                "active_days",
                "user_messages",
                "turns",
                "proactive_received",
                "proactive_answered",
                "feedback_count",
                "cost_cny",
            ):
                row[field] = None
    return {
        "coverage": coverage,
        "items": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "truncated": export and total > 5000,
        "start": str(start),
        "end": str(end),
        "cohort": cohort,
        "definition_version": defs.DEFINITION_VERSION,
    }


async def fact_coverage(start: date, end: date) -> dict:
    """All-cohort markers certify zero activity days; no marker is not zero."""
    from app.db import db

    rows = await db.query_raw(
        "SELECT local_date::text AS day, definition_version, computed_at, settled FROM metrics_daily "
        "WHERE metric='chat_active_users' AND cohort='all' AND dimension='' "
        "AND local_date BETWEEN $1::date AND $2::date",
        str(start),
        str(end),
    )
    index = {r["day"]: r for r in rows}
    missing = [
        str(start + timedelta(days=n))
        for n in range((end - start).days + 1)
        if str(start + timedelta(days=n)) not in index
    ]
    stale = [
        r["day"] for r in rows if r["definition_version"] != defs.DEFINITION_VERSION
    ]
    return {
        "state": "not_computed" if missing else "stale" if stale else "ok",
        "missing_days": missing,
        "stale_days": stale,
        "computed_at": min((r["computed_at"] for r in rows), default=None),
        "settled": bool(rows) and all(r["settled"] for r in rows),
    }


async def journey(user_id: str, start: date, end: date) -> dict | None:
    from app.db import db

    identity = await db.query_raw(
        f"SELECT id,COALESCE(NULLIF(display_name,''),username) AS username,{defs.internal_user_sql('u')} AS internal FROM users u "
        "WHERE id=$1 AND status <> 'deleting'",
        user_id,
    )
    if not identity:
        return None
    rows = await db.query_raw(
        """SELECT local_date::text AS day, opened,chatted,user_messages,turns,sessions,
           proactive_received,proactive_answered,feedback_count,cost_cny::float8,spontaneous,first_message_at,
           computed_at,settled,definition_version
           FROM user_daily_facts WHERE user_id=$1 AND local_date BETWEEN $2::date AND $3::date ORDER BY local_date""",
        user_id,
        str(start),
        str(end),
    )
    proactive = await db.query_raw(
        "SELECT local_date::text AS day,trigger_type,data FROM stats_proactive_daily "
        "WHERE user_id=$1 AND local_date BETWEEN $2::date AND $3::date AND definition_version=$4",
        user_id,
        str(start),
        str(end),
        defs.DEFINITION_VERSION,
    )
    proactive_by_day = {}
    for event in proactive:
        proactive_by_day.setdefault(event["day"], []).append(
            {
                "type": event["trigger_type"],
                "received": event["data"]["sent"],
                "answered_6h": event["data"]["answered_6h"],
                "legacy": event["data"]["legacy"],
            }
        )
    index = {r["day"]: r for r in rows}
    coverage = await fact_coverage(start, end)
    items = []
    for n in range((end - start).days + 1):
        day = str(start + timedelta(days=n))
        row = index.get(day)
        state = (
            "not_computed"
            if day in coverage["missing_days"]
            else "stale"
            if day in coverage["stale_days"]
            else "ok"
        )
        if row and row["definition_version"] != defs.DEFINITION_VERSION:
            state = "stale"
        if state != "ok":
            items.append({"day": day, "state": state})
        else:
            zero = {
                "opened": False,
                "chatted": False,
                "user_messages": 0,
                "turns": 0,
                "sessions": 0,
                "proactive_received": 0,
                "proactive_answered": 0,
                "feedback_count": 0,
                "cost_cny": 0,
                "spontaneous": None,
                "first_message_at": None,
            }
            items.append(
                {
                    "day": day,
                    "state": state,
                    **(row or zero),
                    "proactive_types": proactive_by_day.get(day, []),
                }
            )
    return {
        "user": identity[0],
        "items": items,
        "coverage": coverage,
        "start": str(start),
        "end": str(end),
        "definition_version": defs.DEFINITION_VERSION,
    }


def users_csv(rows: list[dict]) -> str:
    columns = [
        "id",
        "username",
        "registered_at",
        "platform",
        "last_active_day",
        "active_days",
        "user_messages",
        "turns",
        "proactive_received",
        "proactive_answered",
        "feedback_count",
        "cost_cny",
        "status",
    ]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(columns)
    for row in rows[:5000]:
        values = []
        for key in columns:
            value = "" if row.get(key) is None else str(row[key])
            if value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            values.append(value)
        writer.writerow(values)
    return "\ufeff" + out.getvalue()


async def review_items(
    start: date,
    end: date,
    *,
    cohort: str,
    kind: str,
    page: int = 1,
    page_size: int = 25,
) -> dict:
    from app.db import db
    from app.services.effect.experience import REVIEW_KINDS

    if kind not in REVIEW_KINDS or not 1 <= page <= 100000 or not 1 <= page_size <= 100:
        raise ValueError("不支持的类别或分页")
    coverage = await fact_coverage(start, end)
    where = f"""FROM stats_review_items r JOIN users u ON u.id=r.user_id
      WHERE r.local_date BETWEEN $1::date AND $2::date AND r.kind=$3 AND r.definition_version=$4
        AND u.status <> 'deleting' AND {defs.user_row_population_sql("u", cohort)}"""
    args = [str(start), str(end), kind, defs.DEFINITION_VERSION]
    total = (await db.query_raw("SELECT COUNT(*)::int AS n " + where, *args))[0]["n"]
    rows = await db.query_raw(
        "SELECT r.source_id AS id,r.kind,r.occurred_at AS created_at,"
        "r.trace_id,r.conversation_id,r.user_id "
        + where
        + " ORDER BY r.occurred_at DESC,r.source_id LIMIT $5 OFFSET $6",
        *args,
        page_size,
        (page - 1) * page_size,
    )
    return {
        "items": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "state": coverage["state"],
        "coverage": coverage,
        "definition_version": defs.DEFINITION_VERSION,
        "start": str(start),
        "end": str(end),
    }


async def reply_timing(start: date, end: date, cohort: str) -> dict:
    data = await cached_timing(start, end, cohort)
    known, turns = data["coverage"]["measured"], data["coverage"]["turns"]
    coverage_state = (
        "not_collected"
        if end < date(2026, 10, 28)
        else "no_sample"
        if turns == 0
        else "ok"
    )
    return {
        **data,
        "start": str(start),
        "end": str(end),
        "cohort": cohort,
        "coverage": {
            "state": coverage_state,
            "measured": known,
            "turns": turns,
            "unmeasured": max(0, turns - known),
            "note": "覆盖率按窗口内已回复消息串，逐条检查对应首回复；耗时图按首气泡日期，跨日样本可不同。",
        },
        "definition_version": defs.DEFINITION_VERSION,
        "last_bubble_state": "not_collected",
    }


async def chat_target(
    user_id: str, *, day: str = "", trace_id: str = "", conversation_id: str = ""
) -> dict | None:
    """Resolve an authorized drill-down to a real message, never trust URL ownership."""
    from app.db import db

    if any(len(value) > 128 for value in (user_id, trace_id, conversation_id)):
        raise ValueError("定位参数过长")
    if not day and not trace_id and not conversation_id:
        raise ValueError("需要日期、Trace 或会话 ID")
    first, stop = (
        defs.local_day_bounds(date.fromisoformat(day)) if day else (None, None)
    )
    rows = await db.query_raw(
        """SELECT m.id AS message_id,c.id AS conversation_id,
          (SELECT COUNT(*)::int FROM messages newer WHERE newer.conversation_id=c.id
             AND (newer.created_at,newer.id)>(m.created_at,m.id)) AS rank
        FROM messages m JOIN conversations c ON c.id=m.conversation_id JOIN users u ON u.id=c.user_id
        WHERE u.id=$1 AND u.status<>'deleting' AND ($2='' OR c.id=$2)
          AND ($3='' OR m.metadata->>'trace_id'=$3)
          AND ($4::timestamp IS NULL OR (m.created_at >= $4::timestamp AND m.created_at < $5::timestamp))
        ORDER BY m.created_at,m.id LIMIT 1""",
        user_id,
        conversation_id,
        trace_id,
        first.isoformat() if first else None,
        stop.isoformat() if stop else None,
    )
    return rows[0] if rows else None


async def message_distribution(start: date, end: date, cohort: str) -> dict:
    """Preserve the old user/message buckets with the selected derived window."""
    from app.db import db

    coverage = await fact_coverage(start, end)
    rows = await db.query_raw(
        f"""WITH counts AS (
      SELECT u.id,COALESCE(SUM(f.user_messages),0)::bigint AS messages
      FROM users u LEFT JOIN user_daily_facts f ON f.user_id=u.id
        AND f.local_date BETWEEN $1::date AND $2::date AND f.definition_version=$3
      WHERE u.status<>'deleting' AND {defs.user_row_population_sql("u", cohort)}
        AND u.created_at<$4::timestamp GROUP BY u.id
    ), grouped AS (
      SELECT messages,CASE WHEN messages=0 THEN 0 WHEN messages<=10 THEN 1
        WHEN messages<=20 THEN 2 WHEN messages<=30 THEN 3 WHEN messages<=40 THEN 4
        WHEN messages<=50 THEN 5 ELSE 6 END AS bucket FROM counts
    ) SELECT bucket,COUNT(*)::int AS count FROM grouped GROUP BY bucket ORDER BY bucket""",
        str(start),
        str(end),
        defs.DEFINITION_VERSION,
        defs.local_day_bounds(end)[1].isoformat(),
    )
    index = {r["bucket"]: r["count"] for r in rows}
    return {
        "coverage": coverage,
        "buckets": [
            {"label": label, "count": index.get(n, 0)}
            for n, label in enumerate(
                (
                    "0 条",
                    "1–10 条",
                    "11–20 条",
                    "21–30 条",
                    "31–40 条",
                    "41–50 条",
                    "51 条以上",
                )
            )
        ]
        if coverage["state"] == "ok"
        else [],
        "unit": "用户数",
        "definition": "所选窗口内每位用户发送的消息条数；包括窗口结束前注册但未聊天的用户。",
    }
