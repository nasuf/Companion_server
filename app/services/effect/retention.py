"""Retention dashboard from materialized user/day facts; no message scans."""

from datetime import date, timedelta

from app.services.effect import definitions as defs
from app.services.effect.dashboard import fact_coverage
from app.services.schedule_domain.sql_time import local_date_sql

HORIZONS = (1, 3, 7, 14, 30)


async def read_retention(
    start: date, end: date, *, cohort: str, signup_week: str = "", platform: str = ""
) -> dict:
    from app.db import db

    if len(platform) > 64:
        raise ValueError("平台筛选过长")
    week = date.fromisoformat(signup_week) if signup_week else None
    if week and week.weekday() != 0:
        raise ValueError("注册周须为星期一")
    registration = local_date_sql("u.created_at")
    population = defs.user_row_population_sql("u", cohort)
    base = f"""WITH people AS (
        SELECT u.id, {registration} AS registered
        FROM users u WHERE u.status <> 'deleting' AND {population}
        AND {registration} <= $2::date
        AND ($4::date IS NULL OR date_trunc('week', {registration})::date = $4::date)
        AND ($5 = '' OR COALESCE(u.signup_platform,'unknown') = $5)
    ), chat_days AS (
        SELECT f.user_id, f.local_date AS day, p.registered,
          LAG(f.local_date) OVER(PARTITION BY f.user_id ORDER BY f.local_date) AS previous_active
        FROM user_daily_facts f JOIN people p ON p.id=f.user_id
        WHERE f.local_date <= $2::date AND f.definition_version=$3 AND f.chatted
    )"""
    args = [
        str(start),
        str(end),
        defs.DEFINITION_VERSION,
        str(week) if week else None,
        platform,
    ]
    activity = await db.query_raw(
        base
        + """
        SELECT day::text, COUNT(*) FILTER(WHERE day-registered<7)::int AS new,
          COUNT(*) FILTER(WHERE day-registered>=7 AND previous_active>=day-7)::int AS retained,
          COUNT(*) FILTER(WHERE day-registered>=7 AND previous_active<day-7)::int AS returned,
          COUNT(*) FILTER(WHERE day-registered>=7 AND previous_active IS NULL)::int AS first_chat,
          COUNT(*)::int AS active
        FROM chat_days WHERE day BETWEEN $1::date AND $2::date GROUP BY day ORDER BY day""",
        *args,
    )
    gaps = await db.query_raw(
        base
        + """
        SELECT CASE WHEN day-previous_active=1 THEN '1 天' WHEN day-previous_active<=3 THEN '2–3 天'
                    WHEN day-previous_active<=7 THEN '4–7 天' WHEN day-previous_active<=30 THEN '8–30 天'
                    ELSE '31 天以上' END AS label, COUNT(*)::int AS count
        FROM chat_days WHERE day BETWEEN $1::date AND $2::date AND previous_active IS NOT NULL GROUP BY 1""",
        *args,
    )
    gap_quantiles = (
        await db.query_raw(
            base
            + """
        SELECT percentile_cont(.5) WITHIN GROUP(ORDER BY day-previous_active)::float8 AS p50,
               percentile_cont(.9) WITHIN GROUP(ORDER BY day-previous_active)::float8 AS p90
        FROM chat_days WHERE day BETWEEN $1::date AND $2::date AND previous_active IS NOT NULL""",
            *args,
        )
    )[0]
    heatmap = await db.query_raw(
        base
        + """
        SELECT date_trunc('week',p.registered)::date::text AS signup_week, h.n AS horizon,
          COUNT(*)::int AS registered_users,
          COUNT(*) FILTER(WHERE p.registered+h.n <= $2::date)::int AS denominator,
          COUNT(*) FILTER(WHERE p.registered+h.n <= $2::date AND EXISTS (
            SELECT 1 FROM chat_days c WHERE c.user_id=p.id AND c.day=p.registered+h.n))::int AS numerator
        FROM people p CROSS JOIN (VALUES(1),(3),(7),(14),(30)) h(n)
        WHERE p.registered BETWEEN $1::date AND $2::date
        GROUP BY 1,2 ORDER BY 1,2""",
        *args,
    )
    opened = await db.query_raw(
        base
        + """
        SELECT f.local_date::text AS day, COUNT(*) FILTER(WHERE f.opened)::int AS denominator,
          COUNT(*) FILTER(WHERE f.opened AND NOT f.chatted)::int AS numerator
        FROM user_daily_facts f JOIN people p ON p.id=f.user_id
        WHERE f.local_date BETWEEN $1::date AND $2::date AND f.definition_version=$3
        GROUP BY f.local_date ORDER BY f.local_date""",
        *args,
    )
    hourly = await db.query_raw(
        base
        + """
        SELECT EXTRACT(ISODOW FROM d.local_date)::int-1 AS weekday,
               (b->>'hour')::int AS hour, SUM((b->>'count')::bigint)::int AS count
        FROM stats_detail_daily d JOIN people p ON p.id=d.user_id
        CROSS JOIN LATERAL jsonb_array_elements(d.data) b
        WHERE d.local_date BETWEEN $1::date AND $2::date AND d.kind='hourly' AND d.definition_version=$3
        GROUP BY 1,2 ORDER BY 1,2""",
        *args,
    )
    registrations = await db.query_raw(
        base
        + """
        SELECT registered::text AS day, COUNT(*)::int AS count FROM people
        WHERE registered BETWEEN $1::date AND $2::date GROUP BY 1 ORDER BY 1""",
        *args,
    )
    coverage = await fact_coverage(start, end)
    for row in heatmap:
        state = defs.ratio(row["numerator"], row["denominator"])
        row.update(state)
        row["pending_users"] = row["registered_users"] - row["denominator"]
        if row["pending_users"] and row["denominator"] == 0:
            row["state"] = "unsettled"
    for row in opened:
        row.update(defs.ratio(row["numerator"], row["denominator"]))
    if coverage["state"] != "ok":
        # An absent active day is not evidence that a user failed to return.
        activity, gaps, hourly, registrations = [], [], [], []
        gap_quantiles = {"p50": None, "p90": None}
        for row in heatmap + opened:
            row.update(
                state=coverage["state"], value=None, numerator=None, denominator=None
            )
    if coverage["state"] == "ok":
        activity_by_day = {r["day"]: r for r in activity}
        registered_by_day = {r["day"]: r for r in registrations}
        dates = [str(start + timedelta(days=n)) for n in range((end - start).days + 1)]
        activity = [
            activity_by_day.get(
                d, dict(day=d, new=0, retained=0, returned=0, first_chat=0, active=0)
            )
            for d in dates
        ]
        registrations = [registered_by_day.get(d, dict(day=d, count=0)) for d in dates]
    order = ["1 天", "2–3 天", "4–7 天", "8–30 天", "31 天以上"]
    gaps.sort(key=lambda r: order.index(r["label"]))
    return {
        "start": str(start),
        "end": str(end),
        "cohort": cohort,
        "coverage": coverage,
        "gap_quantiles": gap_quantiles,
        "registrations": registrations,
        "hourly": hourly,
        "activity": activity,
        "gaps": gaps,
        "heatmap": heatmap,
        "opened_not_chatted": opened,
        "definition_version": defs.DEFINITION_VERSION,
        "notes": [
            "首次开聊单列注册超过 7 天且没有历史聊天的用户，不伪装成回流。",
            "留存按注册后第 N 个自然日聊天计算；未走完观察窗的用户不进入分母。",
        ],
    }
