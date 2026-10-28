"""Bounded daily diagnostics for existing admin cards, never raw reads on page load."""

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import json

from app.services.effect import definitions as defs
from app.services.schedule_domain.sql_time import local_hour_sql


async def save(tx, day, kind, owner, data, now, *, user_id=None, fingerprint="-"):
    await tx.execute_raw(
        """INSERT INTO stats_detail_daily
        (local_date,kind,owner_key,user_id,data,definition_version,population_fingerprint,computed_at)
        VALUES ($1::date,$2,$3,$4,$5::jsonb,$6,$7,$8::timestamptz)""",
        str(day),
        kind,
        owner,
        user_id,
        json.dumps(data),
        defs.DEFINITION_VERSION,
        fingerprint,
        now.isoformat(),
    )


async def reply_details(tx, day, cohort, now, core, retention, fingerprint):
    from app.services.effect.signals import (
        EffectMetrics,
        _fetch_slices,
        _fetch_sessions,
        _fetch_proactive,
    )

    start, end = defs.local_day_bounds(day)
    sessions = await _fetch_sessions(start, end, cohort, client=tx)
    proactive = await _fetch_proactive(start, end, cohort, client=tx)
    value = EffectMetrics(
        date=str(day),
        cohort=cohort,
        turns=int(core.get("turns") or 0),
        continued=int(core.get("continued") or 0),
        median_gap_s=core.get("median_gap_s"),
        sessions=int(sessions.get("sessions") or 0),
        median_session_turns=sessions.get("median_turns"),
        proactive_sent=int(proactive.get("sent") or 0),
        proactive_answered=int(proactive.get("answered") or 0),
        proactive_legacy_messages=int(proactive.get("legacy_messages") or 0),
        proactive_bubbles=int(proactive.get("bubbles") or 0),
        active_users=int(retention.get("active_users") or 0),
        returned_next_day=int(retention.get("returned") or 0)
        if defs.local_today(now) > day + timedelta(days=1)
        else None,
        slices=await _fetch_slices(start, end, cohort, client=tx),
    )
    await save(
        tx, day, "reply_effect", cohort, value.as_dict(), now, fingerprint=fingerprint
    )


async def compute(tx, day: date, now: datetime):
    start, end = defs.local_day_bounds(day)
    hourly = await tx.query_raw(
        f"""SELECT c.user_id,{local_hour_sql("m.created_at")} AS hour,COUNT(*)::int AS n
        FROM messages m JOIN conversations c ON c.id=m.conversation_id JOIN users u ON u.id=c.user_id
        WHERE m.role='user' AND m.created_at>=$1::timestamp AND m.created_at<$2::timestamp
          AND u.status<>'deleting' GROUP BY c.user_id,2""",
        start.isoformat(),
        end.isoformat(),
    )
    by_user = defaultdict(list)
    for row in hourly:
        by_user[row["user_id"]].append({"hour": row["hour"], "count": row["n"]})
    for user, rows in by_user.items():
        await save(tx, day, "hourly", user, rows, now, user_id=user)
    health = await tx.query_raw(
        """WITH counters AS (
        SELECT user_id,'memory_'||operation AS key,COUNT(*)::bigint AS n FROM memory_changelogs
        WHERE created_at>=$1::timestamp AND created_at<$2::timestamp GROUP BY user_id,operation
        UNION ALL SELECT user_id,'visible_selected',SUM(selected_count)::bigint FROM memory_visible_use_events
        WHERE created_at>=$1::timestamp AND created_at<$2::timestamp GROUP BY user_id
        UNION ALL SELECT user_id,'visible_used',SUM(likely_used_count)::bigint FROM memory_visible_use_events
        WHERE created_at>=$1::timestamp AND created_at<$2::timestamp GROUP BY user_id
        UNION ALL SELECT user_id,'crisis_'||COALESCE(severity,'unknown'),COUNT(*)::bigint FROM crisis_events
        WHERE created_at>=$1::timestamp AND created_at<$2::timestamp GROUP BY user_id,severity
        UNION ALL SELECT user_id,'proactive_'||event_type,COUNT(*)::bigint FROM proactive_event_logs
        WHERE created_at>=$1::timestamp AND created_at<$2::timestamp GROUP BY user_id,event_type
    ) SELECT c.* FROM counters c LEFT JOIN users u ON u.id=c.user_id
      WHERE c.user_id IS NULL OR u.status<>'deleting'""",
        start.isoformat(),
        end.isoformat(),
    )
    health_users = defaultdict(dict)
    for row in health:
        health_users[row["user_id"]][row["key"]] = row["n"] or 0
    for user, counters in health_users.items():
        await save(tx, day, "health", user or "-", counters, now, user_id=user)
    # Persist only aggregate configuration/token counts, never prompts or image data.
    context = await tx.query_raw(
        """SELECT l.user_id,l.agent_id,l.tokens_by_model,l.cost_cny,l.web_search_calls,
        t.extra_json->'metadata'->'chat_context_policy'->>'effective_rounds' AS rounds,
        t.extra_json->'metadata'->'chat_context_policy'->>'main_history_token_budget' AS main_tokens
        FROM llm_usage l LEFT JOIN users u ON u.id=l.user_id LEFT JOIN trace_runs t ON t.run_id=l.trace_id
        WHERE l.scope='chat' AND l.created_at>=$1::timestamp AND l.created_at<$2::timestamp
        AND (l.user_id IS NULL OR u.status<>'deleting')""",
        start.isoformat(),
        end.isoformat(),
    )
    from app.services.effect.context_summary import summarize

    context_users = defaultdict(list)
    for row in context:
        context_users[(row["user_id"], row["agent_id"])].append(row)
    for (user, agent), rows in context_users.items():
        await save(
            tx,
            day,
            "context",
            json.dumps([user, agent]),
            {"agent_id": agent, **summarize(rows, start, end, now)},
            now,
            user_id=user,
        )
    # Reuse the existing sanitization and field definitions for activity diagnostics.
    from app.services.offline import copy_quality as copy

    query = (
        copy._QUERY.replace("a.id, a.created_at", "a.user_id, a.id, a.created_at")
        .replace(
            "a.created_at >= now() - make_interval(days => $1::int)",
            "a.created_at >= $1::timestamp AND a.created_at < $2::timestamp AND EXISTS (SELECT 1 FROM users u WHERE u.id=a.user_id AND u.status<>'deleting')",
        )
        .replace("LIMIT $2::int", "LIMIT 5001")
    )
    activity = await tx.query_raw(query, start.isoformat(), end.isoformat())
    if len(activity) > 5000:
        raise ValueError("Activity diagnostics daily row limit exceeded")
    copies = defaultdict(list)
    for row in activity:
        copies[row["user_id"]].append(row)
    for user, rows in copies.items():
        data = copy.summarize_copy_rows(rows, window=1, now=now, row_limit=5000)
        await save(tx, day, "copy_quality", user, data, now, user_id=user)


async def read(
    start: date, end: date, kind: str, *, cohort: str = "all", agent: str = ""
):
    from app.db import db
    from app.services.effect.dashboard import fact_coverage

    if kind not in {"reply_effect", "copy_quality", "context", "hourly", "health"}:
        raise ValueError("未知诊断栏目")
    coverage = await fact_coverage(start, end)
    fingerprint = await defs.population_fingerprint() if cohort == "external" else "-"
    rows = await db.query_raw(
        f"""SELECT d.local_date::text AS day,d.data FROM stats_detail_daily d
        LEFT JOIN users u ON u.id=d.user_id
        WHERE d.local_date BETWEEN $1::date AND $2::date AND d.kind=$3 AND d.definition_version=$4
        AND (d.user_id IS NULL OR u.status<>'deleting')
        AND (CASE WHEN d.kind='reply_effect' THEN d.owner_key=$5 AND d.population_fingerprint=$6
             ELSE {defs.population_sql("d.user_id", cohort)} END)
        AND ($7='' OR d.data->>'agent_id'=$7)
        ORDER BY d.local_date""",
        str(start),
        str(end),
        kind,
        defs.DEFINITION_VERSION,
        cohort,
        fingerprint,
        agent,
    )
    if coverage["state"] != "ok":
        return {"coverage": coverage, "data": None}
    payloads = [r["data"] for r in rows]
    if kind == "reply_effect":
        from app.services.effect.store import summarise, merge_slices
        from app.services.effect.signals import CONTINUATION_WINDOW, MIN_SLICE_TURNS

        if len(rows) != (end - start).days + 1:
            return {"coverage": {**coverage, "state": "not_computed"}, "data": None}
        data = {
            "days": payloads,
            "summary": summarise(payloads),
            "slices": merge_slices(payloads),
            "cohort": cohort,
            "definitions": defs.describe(
                ("continuation", "proactive_response", "next_day_return")
            ),
            "config": {
                "continuation_window_s": int(CONTINUATION_WINDOW.total_seconds()),
                "min_slice_turns": MIN_SLICE_TURNS,
            },
        }
    elif kind == "copy_quality":
        from app.services.offline.copy_quality import FIELDS, RECENT_LIMIT

        fields = {k: Counter() for k in FIELDS}
        recent = []
        for p in payloads:
            for k in fields:
                fields[k].update(p["fields"][k])
            recent.extend(p["recent"])
        data = {
            "sampled_at": coverage["computed_at"],
            "window_days": (end - start).days + 1,
            "row_limit": 5000,
            "truncated": False,
            "total": sum(p["total"] for p in payloads),
            "fields": {k: dict(v) for k, v in fields.items()},
            "recent": sorted(
                recent, key=lambda r: (r["created_at"], r["id"]), reverse=True
            )[:RECENT_LIMIT],
        }
    elif kind == "context":
        from app.services.effect.context_summary import merge

        data = merge(
            payloads,
            defs.local_day_bounds(start)[0],
            defs.local_day_bounds(end)[1],
            coverage["computed_at"],
        )
    elif kind == "health":
        counts = Counter()
        for payload in payloads:
            counts.update(payload)
        data = dict(counts)
    else:
        totals = Counter()
        for row in rows:
            weekday = date.fromisoformat(row["day"]).weekday()
            for bucket in row["data"]:
                totals[weekday, bucket["hour"]] += bucket["count"]
        data = [
            {"weekday": day, "hour": hour, "count": totals[day, hour]}
            for day in range(7)
            for hour in range(24)
        ]
    return {"coverage": coverage, "data": data}
