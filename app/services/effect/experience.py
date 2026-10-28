"""Materialized conversation signals and review IDs. Never persist message text.

Signals are reasons to inspect, not user negative ratings. All burst-derived
signals reuse the daily-facts definition (including game/takeover exclusions).
"""

from datetime import date, datetime

from app.services.effect import definitions as defs
from app.services.schedule_domain.sql_time import local_hour_sql

REVIEW_KINDS = (
    "no_reply",
    "reply_failed",
    "slow_first_bubble",
    "feedback",
    "resend_60s",
    "meta_comment",
)


def normalize_text(text: str) -> str:
    """Match real text only, ignoring punctuation/space; pure emoji are empty."""
    return "".join(char.casefold() for char in text if char.isalnum())


async def compute(tx, day: date, now: datetime) -> list[dict]:
    from app.services.effect.daily_facts import (
        burst_details_cte,
        burst_query_params,
        _row,
        is_settled,
    )

    start, end = defs.local_day_bounds(day)
    # The raw content is used transiently inside this computation, never in a cache,
    # response, log or review item. Fetch only the last message of each eligible burst.
    turns = await tx.query_raw(
        burst_details_cte()
        + """
      SELECT b.user_id,b.conversation_id,b.last_at,b.replied,
        b.last_at+$7::interval <= ($5::timestamptz AT TIME ZONE 'UTC') AS mature,
        u.id AS message_id,u.content AS user_text,u.has_attachment,
        r.id AS reply_id,r.created_at AS replied_at,r.metadata AS reply_metadata,
        n.id AS next_id,n.content AS next_text,n.created_at AS next_at,n.has_attachment AS next_has_attachment
      FROM day_bursts b CROSS JOIN bind_types p
      JOIN users owner ON owner.id=b.user_id AND owner.status <> 'deleting'
      JOIN LATERAL (SELECT id,content,EXISTS(SELECT 1 FROM chat_message_attachments a WHERE a.message_id=messages.id) AS has_attachment FROM messages WHERE conversation_id=b.conversation_id
        AND role='user' AND created_at=b.last_at ORDER BY id DESC LIMIT 1) u ON true
      LEFT JOIN LATERAL (SELECT id,created_at,metadata FROM messages
        WHERE conversation_id=b.conversation_id AND role='assistant'
          AND COALESCE(metadata->>'proactive','false') <> 'true'
          AND created_at>b.last_at AND created_at<=b.last_at+$7::interval
        ORDER BY created_at,id LIMIT 1) r ON true
      LEFT JOIN LATERAL (SELECT next.id,next.content,next.created_at,EXISTS(SELECT 1 FROM chat_message_attachments a WHERE a.message_id=next.id) AS has_attachment FROM messages next
        WHERE next.conversation_id=b.conversation_id AND next.role='user'
          AND next.created_at>r.created_at AND next.created_at<=r.created_at+INTERVAL '60 seconds'
          AND COALESCE(next.metadata->>'human_takeover','false') <> 'true'
          AND NOT EXISTS (SELECT 1 FROM game_sessions g WHERE g.conversation_id=b.conversation_id
            AND next.created_at>=g.started_at AND next.created_at<=COALESCE(g.ended_at,g.started_at+$10::interval))
        ORDER BY next.created_at,next.id LIMIT 1) n ON true
    """,
        *burst_query_params(day, now),
    )
    await tx.execute_raw(
        "DELETE FROM stats_review_items WHERE local_date=$1::date", str(day)
    )
    records = []
    for turn in turns:
        meta = turn["reply_metadata"] or {}
        trace_id = meta.get("trace_id") if isinstance(meta, dict) else None
        base = dict(
            user_id=turn["user_id"],
            conversation_id=turn["conversation_id"],
            trace_id=trace_id,
        )
        if not turn["replied"] and turn["mature"]:
            records.append(
                dict(
                    base,
                    kind="no_reply",
                    source_id=turn["message_id"],
                    occurred_at=turn["last_at"],
                )
            )
        normalized = normalize_text(turn["user_text"] or "")
        if (
            turn["next_id"]
            and not turn["has_attachment"]
            and not turn["next_has_attachment"]
            and normalized
            and normalized == normalize_text(turn["next_text"] or "")
        ):
            records.append(
                dict(
                    base,
                    kind="resend_60s",
                    source_id=turn["next_id"],
                    occurred_at=turn["next_at"],
                )
            )
    # Per-message meta comments, rather than just the last bubble in a burst.
    comments = await tx.query_raw(
        """
      SELECT m.id,m.created_at,m.conversation_id,c.user_id FROM messages m
      JOIN conversations c ON c.id=m.conversation_id JOIN users u ON u.id=c.user_id
      WHERE m.role='user' AND m.created_at >= $1::timestamp AND m.created_at<$2::timestamp
        AND u.status <> 'deleting' AND COALESCE(m.metadata->>'human_takeover','false') <> 'true'
        AND m.content LIKE ANY($3::text[])
        AND NOT EXISTS (SELECT 1 FROM game_sessions g WHERE g.conversation_id=m.conversation_id
          AND m.created_at>=g.started_at AND m.created_at<=COALESCE(g.ended_at,g.started_at+INTERVAL '2 hours'))
    """,
        start.isoformat(),
        end.isoformat(),
        ["%" + word + "%" for word in defs.META_COMMENT_PHRASES],
    )
    for row in comments:
        records.append(
            dict(
                kind="meta_comment",
                source_id=row["id"],
                occurred_at=row["created_at"],
                user_id=row["user_id"],
                conversation_id=row["conversation_id"],
                trace_id=None,
            )
        )
    feedback = await tx.query_raw(
        """SELECT f.id,f.user_id,f.created_at FROM user_feedback f
        JOIN users u ON u.id=f.user_id WHERE f.created_at >= $1::timestamp AND f.created_at<$2::timestamp
          AND u.status <> 'deleting' """,
        start.isoformat(),
        end.isoformat(),
    )
    records.extend(
        dict(
            kind="feedback",
            source_id=r["id"],
            occurred_at=r["created_at"],
            user_id=r["user_id"],
            conversation_id=None,
            trace_id=None,
        )
        for r in feedback
    )
    # Failed replies are stored failure placeholders; ordinary empty/content cards are not failures.
    failed = await tx.query_raw(
        """SELECT m.id,m.created_at,m.conversation_id,c.user_id,m.metadata->>'trace_id' AS trace_id
      FROM messages m JOIN conversations c ON c.id=m.conversation_id JOIN users u ON u.id=c.user_id
      WHERE m.created_at >= $1::timestamp AND m.created_at<$2::timestamp AND u.status <> 'deleting'
        AND m.role='assistant' AND m.metadata->>'reply_failed'='true' """,
        start.isoformat(),
        end.isoformat(),
    )
    records.extend(
        dict(
            kind="reply_failed",
            source_id=r["id"],
            occurred_at=r["created_at"],
            user_id=r["user_id"],
            conversation_id=r["conversation_id"],
            trace_id=r["trace_id"],
        )
        for r in failed
    )
    from app.services.effect.daily_facts import timing_cte
    from app.services.interaction.reply_telemetry import VERSION

    slow = await tx.query_raw(
        timing_cte("all")
        + """SELECT t.message_id AS id,t.anchor_at,t.conversation_id,
        t.trace_id,c.user_id FROM t JOIN conversations c ON c.id=t.conversation_id
        JOIN users u ON u.id=c.user_id WHERE t.anchor_at >= $1::timestamp AND t.anchor_at<$2::timestamp
          AND t.first_ms>20000 AND u.status <> 'deleting'""",
        start.isoformat(),
        end.isoformat(),
        VERSION,
    )
    records.extend(
        dict(
            kind="slow_first_bubble",
            source_id=r["id"],
            occurred_at=r["anchor_at"],
            user_id=r["user_id"],
            conversation_id=r["conversation_id"],
            trace_id=r["trace_id"],
        )
        for r in slow
    )
    if records:
        import json

        await tx.execute_raw(
            """INSERT INTO stats_review_items
          (kind,source_id,local_date,user_id,conversation_id,trace_id,occurred_at,definition_version,computed_at)
          SELECT r.kind,r.source_id,$1::date,r.user_id,r.conversation_id,r.trace_id,r.occurred_at,$2,$3::timestamptz
          FROM jsonb_to_recordset($4::jsonb) r(kind text,source_id text,user_id text,conversation_id text,
             trace_id text,occurred_at timestamp)
          ON CONFLICT(kind,source_id) DO UPDATE SET local_date=EXCLUDED.local_date,
            trace_id=EXCLUDED.trace_id,definition_version=EXCLUDED.definition_version,computed_at=EXCLUDED.computed_at""",
            str(day),
            defs.DEFINITION_VERSION,
            now.isoformat(),
            json.dumps(records, default=str),
        )
    await tx.execute_raw(
        """UPDATE user_daily_facts f SET
        resend_60s=(SELECT COUNT(*) FROM stats_review_items r WHERE r.user_id=f.user_id AND r.local_date=f.local_date AND r.kind='resend_60s'),
        meta_comments=(SELECT COUNT(*) FROM stats_review_items r WHERE r.user_id=f.user_id AND r.local_date=f.local_date AND r.kind='meta_comment')
        WHERE f.local_date=$1::date""",
        str(day),
    )
    rows = []
    for cohort in defs.COHORTS:
        sums = (
            await tx.query_raw(
                f"""SELECT COALESCE(SUM(resend_60s),0)::int AS resend,
          COALESCE(SUM(meta_comments),0)::int AS comments,COALESCE(SUM(turns),0)::int AS turns,
          COALESCE(SUM(user_messages),0)::int AS messages,
          COUNT(*) FILTER(WHERE opened)::int AS opened,
          COUNT(*) FILTER(WHERE opened AND NOT chatted)::int AS silent
          FROM user_daily_facts f WHERE local_date=$1::date AND {defs.population_sql("f.user_id", cohort)}""",
                str(day),
            )
        )[0]
        for metric, n, d in (
            ("resend_60s", sums["resend"], sums["turns"]),
            ("meta_comment", sums["comments"], sums["messages"]),
            ("opened_not_chatted", sums["silent"], sums["opened"]),
        ):
            rows.append(
                _row(
                    metric,
                    cohort,
                    numerator=n,
                    denominator=d,
                    settled=is_settled(metric, day, now),
                )
            )
        hours = await tx.query_raw(
            f"""SELECT {local_hour_sql("m.created_at")}::int AS hour, COUNT(*)::int AS n
          FROM messages m JOIN conversations c ON c.id=m.conversation_id
          JOIN users owner ON owner.id=c.user_id AND owner.status<>'deleting' WHERE m.role='user'
            AND m.created_at >= $1::timestamp AND m.created_at<$2::timestamp
            AND {defs.population_sql("c.user_id", cohort)} GROUP BY 1""",
            start.isoformat(),
            end.isoformat(),
        )
        for r in hours:
            rows.append(
                _row(
                    "hourly_messages",
                    cohort,
                    value=r["n"],
                    dimension="hour",
                    dimension_value=str(r["hour"]),
                    settled=is_settled("hourly_messages", day, now),
                )
            )
        rows.append(
            _row(
                "hourly_messages",
                cohort,
                value=sum(r["n"] for r in hours),
                settled=is_settled("hourly_messages", day, now),
            )
        )
        allowed = {
            x["id"]
            for x in await tx.query_raw(
                f"SELECT u.id FROM users u WHERE {defs.user_row_population_sql('u', cohort)}"
            )
        }
        rows.append(
            _row(
                "feedback_count",
                cohort,
                value=sum(
                    r["kind"] == "feedback" and r["user_id"] in allowed for r in records
                ),
                settled=is_settled("feedback_count", day, now),
            )
        )
        labels = await tx.query_raw(
            f"""SELECT label,COUNT(*)::int AS n FROM bug_reports b
          JOIN messages m ON m.id=b.message_id JOIN conversations c ON c.id=m.conversation_id
          JOIN users u ON u.id=c.user_id CROSS JOIN LATERAL unnest(b.error_types) label
          WHERE b.created_at >= $1::timestamp AND b.created_at<$2::timestamp
            AND u.status <> 'deleting' AND {defs.user_row_population_sql("u", cohort)} GROUP BY label""",
            start.isoformat(),
            end.isoformat(),
        )
        for r in labels:
            rows.append(
                _row(
                    "admin_bug_labels",
                    cohort,
                    value=r["n"],
                    dimension="label",
                    dimension_value=r["label"],
                    settled=is_settled("admin_bug_labels", day, now),
                )
            )
        rows.append(
            _row(
                "admin_bug_labels",
                cohort,
                value=sum(r["n"] for r in labels),
                settled=is_settled("admin_bug_labels", day, now),
            )
        )
        # Preserve the existing continuation definition exactly, including its
        # per-user-message denominator; only add the requested path dimension.
        from app.services.effect.signals import (
            _TURNS_CTE,
            _with_population,
            CONTINUATION_WINDOW,
            _BOUNDARY_LOOKAHEAD,
        )

        sliced = await tx.query_raw(
            _with_population(_TURNS_CTE, "m.conversation_id", cohort)
            + """
          SELECT COALESCE(diag->>'reply_path','unknown') AS path,COUNT(*)::int AS n,
            COUNT(*) FILTER(WHERE continued)::int AS continued FROM scored GROUP BY 1""",
            start.isoformat(),
            end.isoformat(),
            f"{CONTINUATION_WINDOW.total_seconds()} seconds",
            f"{_BOUNDARY_LOOKAHEAD.total_seconds()} seconds",
        )
        rows.extend(
            _row(
                "continuation",
                cohort,
                numerator=r["continued"],
                denominator=r["n"],
                dimension="reply_path",
                dimension_value=r["path"],
                settled=is_settled("continuation", day, now),
            )
            for r in sliced
        )
    return rows
