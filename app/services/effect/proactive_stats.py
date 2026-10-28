"""Event-ID based proactive funnel; absent attribution is never guessed.

A provider accepting a push is not proof of device delivery. Existing window
logs lack send-event IDs, so they remain independent diagnostics, not a linked
conversion denominator. No client/open receipts are manufactured.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json
import re

from app.services.effect import definitions as defs

FIELDS = (
    "sent",
    "legacy",
    "push_accepted",
    "answered_3h",
    "answered_6h",
    "three_turns",
    "silent_2d",
    "awakened_24h",
    "pending_24h",
)
BUCKETS = ("≤1分钟", "1–5分钟", "5–30分钟", "30分钟–3小时", "3–6小时", "6–24小时")


def blank():
    return {**dict.fromkeys(FIELDS, 0), "latency": dict.fromkeys(BUCKETS, 0)}


def add(target, data):
    for key in FIELDS:
        target[key] += int(data.get(key, 0))
    for key in BUCKETS:
        target["latency"][key] += int(data.get("latency", {}).get(key, 0))


def safe_label(value):
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_:-]{1,64}", value)
        else "unknown"
    )


async def compute(tx, day: date, now: datetime) -> list[dict]:
    from app.services.effect.daily_facts import _row, is_settled

    start, end = defs.local_day_bounds(day)
    events = await tx.query_raw(
        """WITH e AS ("""
        + defs.proactive_events_sql(
            "$1::timestamp", "$2::timestamp", "INTERVAL '24 hours'", "all"
        )
        + """)
      SELECT e.*,EXTRACT(EPOCH FROM (first_reply_at-sent_at))::float8 AS response_seconds,
        sent_at+INTERVAL '24 hours'<= $3::timestamp AS mature,
        EXISTS(SELECT 1 FROM messages p WHERE p.conversation_id=e.conversation_id AND p.role='user'
          AND p.created_at<e.sent_at-INTERVAL '2 days') AND NOT EXISTS(
          SELECT 1 FROM messages p WHERE p.conversation_id=e.conversation_id AND p.role='user'
            AND p.created_at>=e.sent_at-INTERVAL '2 days' AND p.created_at<e.sent_at) AS silent_2d,
        EXISTS(SELECT 1 FROM notification_events n JOIN messages m ON m.id=n.message_id
          WHERE n.user_id=e.user_id AND n.conversation_id=e.conversation_id AND n.type='agent_message'
            AND n.status='sent' AND m.metadata->>'proactive_event_id'=e.event_key) AS push_accepted,
        (SELECT COUNT(*)::int FROM messages u WHERE u.conversation_id=e.conversation_id AND u.role='user'
          AND u.created_at>=e.first_reply_at AND u.created_at<=e.first_reply_at+INTERVAL '3 hours'
          AND COALESCE(u.metadata->>'human_takeover','false')<>'true'
          AND NOT EXISTS(SELECT 1 FROM game_sessions g WHERE g.conversation_id=u.conversation_id
            AND u.created_at>=g.started_at AND u.created_at<=COALESCE(g.ended_at,g.started_at+INTERVAL '2 hours'))
          AND EXISTS(SELECT 1 FROM messages a WHERE a.conversation_id=u.conversation_id AND a.role='assistant'
            AND COALESCE(a.metadata->>'proactive','false')<>'true' AND a.created_at>u.created_at
            AND a.created_at<=u.created_at+INTERVAL '10 minutes'
            AND NOT EXISTS(SELECT 1 FROM messages next WHERE next.conversation_id=u.conversation_id AND next.role='user'
              AND (next.created_at,next.id)>(u.created_at,u.id) AND next.created_at<a.created_at))) AS turns_after_reply
      FROM e JOIN users owner ON owner.id=e.user_id AND owner.status<>'deleting' """,
        start.isoformat(),
        end.isoformat(),
        now.astimezone(timezone.utc).replace(tzinfo=None).isoformat(),
    )
    groups = defaultdict(blank)
    for event in events:
        out = groups[
            (event["user_id"], safe_label(event["trigger_type"]).removesuffix("_card"))
        ]
        if not event["identified"]:
            out["legacy"] += 1
            continue
        out["sent"] += 1
        out["push_accepted"] += int(event["push_accepted"])
        seconds = event["response_seconds"]
        out["answered_3h"] += int(seconds is not None and seconds <= 10800)
        out["answered_6h"] += int(seconds is not None and seconds <= 21600)
        out["three_turns"] += int(
            seconds is not None and seconds <= 10800 and event["turns_after_reply"] >= 3
        )
        if not event["mature"]:
            out["pending_24h"] += 1
        if event["silent_2d"] and event["mature"]:
            out["silent_2d"] += 1
            out["awakened_24h"] += int(seconds is not None)
        if seconds is not None:
            bucket = next(
                label
                for limit, label in zip((60, 300, 1800, 10800, 21600, 86400), BUCKETS)
                if seconds <= limit
            )
            out["latency"][bucket] += 1
    await tx.execute_raw(
        "DELETE FROM stats_proactive_daily WHERE local_date=$1::date", str(day)
    )
    if groups:
        await tx.execute_raw(
            """INSERT INTO stats_proactive_daily(user_id,local_date,trigger_type,data,definition_version,computed_at)
          SELECT r.user_id,$1::date,r.trigger_type,r.data,$2,$3::timestamptz
          FROM jsonb_to_recordset($4::jsonb) r(user_id text,trigger_type text,data jsonb)""",
            str(day),
            defs.DEFINITION_VERSION,
            now.isoformat(),
            json.dumps(
                [
                    dict(user_id=u, trigger_type=t, data=d)
                    for (u, t), d in groups.items()
                ]
            ),
        )
    logs = await tx.query_raw(
        """SELECT p.user_id,p.event_type,p.payload->>'reason' AS reason,COUNT(*)::int AS n
      FROM proactive_event_logs p JOIN users u ON u.id=p.user_id AND u.status<>'deleting'
      WHERE p.created_at>=$1::timestamp AND p.created_at<$2::timestamp GROUP BY 1,2,3""",
        start.isoformat(),
        end.isoformat(),
    )
    rows = []
    for cohort in defs.COHORTS:
        allowed = {
            x["id"]
            for x in await tx.query_raw(
                f"SELECT id FROM users u WHERE u.status<>'deleting' AND {defs.user_row_population_sql('u', cohort)}"
            )
        }
        scoped = defaultdict(blank)
        for (user, trigger), data in groups.items():
            if user in allowed:
                add(scoped[trigger], data)
        overall = blank()
        for data in scoped.values():
            add(overall, data)
        for trigger, data in [(None, overall), *sorted(scoped.items())]:
            dim = (
                {}
                if trigger is None
                else dict(dimension="trigger_type", dimension_value=trigger)
            )
            for field in FIELDS:
                metric = "proactive_" + field
                rows.append(
                    _row(
                        metric,
                        cohort,
                        value=data[field],
                        settled=is_settled(metric, day, now),
                        **dim,
                    )
                )
        for metric, source in (
            ("proactive_logs", "event_type"),
            ("proactive_skip_reasons", "reason"),
        ):
            counts = defaultdict(int)
            for row in logs:
                if row["user_id"] in allowed and (
                    source == "event_type" or row["event_type"] == "send_skipped"
                ):
                    counts[safe_label(row[source])] += row["n"]
            rows.append(
                _row(
                    metric,
                    cohort,
                    value=sum(counts.values()),
                    settled=is_settled(metric, day, now),
                )
            )
            rows.extend(
                _row(
                    metric,
                    cohort,
                    value=count,
                    dimension="reason",
                    dimension_value=label,
                    settled=is_settled(metric, day, now),
                )
                for label, count in sorted(counts.items())
            )
    return rows


async def read(start: date, end: date, cohort: str) -> dict:
    from app.db import db
    from app.services.effect.dashboard import fact_coverage

    coverage = await fact_coverage(start, end)
    rows = await db.query_raw(
        f"""SELECT trigger_type,data FROM stats_proactive_daily f
      JOIN users u ON u.id=f.user_id WHERE f.local_date BETWEEN $1::date AND $2::date
        AND f.definition_version=$3 AND u.status<>'deleting' AND {defs.user_row_population_sql("u", cohort)}""",
        str(start),
        str(end),
        defs.DEFINITION_VERSION,
    )
    groups = defaultdict(blank)
    total = blank()
    for row in rows:
        add(groups[row["trigger_type"]], row["data"])
        add(total, row["data"])
    settled = end < defs.local_today() - timedelta(days=1)
    return dict(
        start=str(start),
        end=str(end),
        cohort=cohort,
        coverage=coverage,
        settled=settled,
        total=total if coverage["state"] == "ok" else None,
        by_type=[
            dict(
                type=k,
                **v,
                response=defs.ratio(v["answered_3h"], v["sent"], settled=settled),
                awakening=defs.ratio(v["awakened_24h"], v["silent_2d"]),
            )
            for k, v in sorted(groups.items())
        ]
        if coverage["state"] == "ok"
        else [],
        definition_version=defs.DEFINITION_VERSION,
        unavailable={
            "due": "现有窗口日志没有发送事件 ID，不能连接为同一漏斗",
            "generated": "尚无可与发送事件一一关联的生成记录",
            "delivered": "仅有供应商接受回执，不代表手机实收",
            "opened": "依赖 App 点开回执，暂缓",
        },
        notes=[
            "按主动事件 ID 去重；无 ID 的历史气泡单列，不算进漏斗。",
            "用户回应不要求推送已接受，因为用户也可能正在 App 内聊天。",
            "聊满 3 回合在回应后 3 小时内观察；沉默 2 天以之前的用户聊天记录为准。",
        ],
    )
