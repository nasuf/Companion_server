"""Settled conversation episodes, computed offline and stored without chat text.

An episode is separated by a three-hour gap between user messages, including
across midnight. Depth counts eligible answered bursts rather than bubbles.
The bounded 92-day lookback is conservative: a left-censored episode is excluded
and reported separately, never silently treated as a fresh short conversation.
"""

from datetime import date, datetime, timedelta, timezone
import re

from app.services.effect import definitions as defs

LOOKBACK_DAYS = 92
QUESTION_ENDINGS = ("吗", "呢", "么", "嘛", "对吧", "是不是", "怎么样", "如何")


def is_question(text: str) -> bool:
    if "?" in text or "？" in text:
        return True
    words = re.sub(r"[^\w\u4e00-\u9fff]+$", "", text.strip())
    return words.endswith(QUESTION_ENDINGS)


async def compute(tx, day: date, now: datetime) -> list[dict]:
    from app.services.effect.daily_facts import _row, is_settled

    start, end = defs.local_day_bounds(day)
    lower = start - timedelta(days=LOOKBACK_DAYS)
    current = now.astimezone(timezone.utc).replace(tzinfo=None)
    episodes = await tx.query_raw(
        """
      WITH users_messages AS (
        SELECT m.*,c.user_id,LAG(m.created_at) OVER w AS previous_at
        FROM messages m JOIN conversations c ON c.id=m.conversation_id
        JOIN users u ON u.id=c.user_id AND u.status <> 'deleting'
        WHERE m.role='user' AND m.created_at >= $1::timestamp
          AND m.created_at < LEAST($3::timestamp+INTERVAL '3 hours',$4::timestamp)
        WINDOW w AS (PARTITION BY m.conversation_id ORDER BY m.created_at,m.id)
      ), grouped AS (
        SELECT *,SUM(CASE WHEN previous_at IS NULL OR created_at-previous_at>INTERVAL '3 hours'
                         THEN 1 ELSE 0 END) OVER (PARTITION BY conversation_id ORDER BY created_at,id) AS grp
        FROM users_messages
      ), episodes AS (
        SELECT user_id,conversation_id,grp,MIN(created_at) AS first_at,MAX(created_at) AS last_at,
          bool_or(COALESCE(metadata->>'human_takeover','false')='true') AS takeover,
          bool_or(EXISTS(SELECT 1 FROM game_sessions g WHERE g.conversation_id=grouped.conversation_id
            AND grouped.created_at>=g.started_at
            AND grouped.created_at<=COALESCE(g.ended_at,g.started_at+INTERVAL '2 hours'))) AS game
        FROM grouped GROUP BY user_id,conversation_id,grp
      ), completed AS (
        SELECT e.*,EXISTS(SELECT 1 FROM messages old WHERE old.conversation_id=e.conversation_id
          AND old.role='user' AND old.created_at<$1::timestamp
          AND old.created_at>=e.first_at-INTERVAL '3 hours') AS left_censored
        FROM episodes e WHERE last_at>=$2::timestamp AND last_at<$3::timestamp
          AND last_at+INTERVAL '3 hours'<=$4::timestamp AND NOT takeover AND NOT game
      )
      SELECT e.*,last.role AS end_role,last.content AS end_text,
        (SELECT COUNT(*)::int FROM grouped m WHERE m.conversation_id=e.conversation_id AND m.grp=e.grp
          AND EXISTS(SELECT 1 FROM messages a WHERE a.conversation_id=m.conversation_id AND a.role='assistant'
            AND COALESCE(a.metadata->>'proactive','false')<>'true'
            AND a.created_at>m.created_at AND a.created_at<=m.created_at+INTERVAL '10 minutes'
            AND NOT EXISTS(SELECT 1 FROM messages next WHERE next.conversation_id=m.conversation_id
              AND next.role='user' AND (next.created_at,next.id)>(m.created_at,m.id)
              AND next.created_at<a.created_at))) AS turns
      FROM completed e
      LEFT JOIN LATERAL (SELECT role,content FROM messages m WHERE m.conversation_id=e.conversation_id
        AND m.role IN ('user','assistant') AND m.created_at>=e.first_at
        AND m.created_at<=e.last_at+INTERVAL '3 hours'
        AND COALESCE(m.metadata->>'proactive','false')<>'true'
        ORDER BY m.created_at DESC,m.id DESC LIMIT 1) last ON true
    """,
        lower.isoformat(),
        start.isoformat(),
        end.isoformat(),
        current.isoformat(),
    )
    # Facts may be absent on the close day for an episode crossing midnight only
    # if there were no user messages; by construction last_at is a user message.
    for user_id in {e["user_id"] for e in episodes}:
        selected = [
            e for e in episodes if e["user_id"] == user_id and not e["left_censored"]
        ]
        await tx.execute_raw(
            "UPDATE user_daily_facts SET completed_sessions=$3,short_sessions=$4 "
            "WHERE user_id=$1 AND local_date=$2::date",
            user_id,
            str(day),
            len(selected),
            sum(e["turns"] <= 3 for e in selected),
        )
    rows = []
    for cohort in defs.COHORTS:
        allowed = {
            r["id"]
            for r in await tx.query_raw(
                f"SELECT u.id FROM users u WHERE u.status <> 'deleting' AND {defs.user_row_population_sql('u', cohort)}"
            )
        }
        scoped = [e for e in episodes if e["user_id"] in allowed]
        valid = [e for e in scoped if not e["left_censored"]]
        rows.append(
            _row(
                "short_session",
                cohort,
                numerator=sum(e["turns"] <= 3 for e in valid),
                denominator=len(valid),
                settled=is_settled("short_session", day, now),
            )
        )
        rows.append(
            _row(
                "session_unknown",
                cohort,
                value=sum(e["left_censored"] for e in scoped),
                settled=is_settled("session_unknown", day, now),
            )
        )
        for label, predicate in (
            ("0", lambda e: e["turns"] == 0),
            ("1–3", lambda e: 1 <= e["turns"] <= 3),
            ("4–10", lambda e: 4 <= e["turns"] <= 10),
            ("11–30", lambda e: 11 <= e["turns"] <= 30),
            ("31+", lambda e: e["turns"] > 30),
        ):
            rows.append(
                _row(
                    "session_depth",
                    cohort,
                    value=sum(predicate(e) for e in valid),
                    dimension="bucket",
                    dimension_value=label,
                    settled=is_settled("session_depth", day, now),
                )
            )
        for label, predicate in (
            ("user", lambda e: e["end_role"] == "user"),
            (
                "ai_question",
                lambda e: e["end_role"] == "assistant"
                and is_question(e["end_text"] or ""),
            ),
            (
                "ai_statement",
                lambda e: e["end_role"] == "assistant"
                and not is_question(e["end_text"] or ""),
            ),
        ):
            rows.append(
                _row(
                    "session_ending",
                    cohort,
                    value=sum(predicate(e) for e in valid),
                    dimension="bucket",
                    dimension_value=label,
                    settled=is_settled("session_ending", day, now),
                )
            )
        for metric in ("session_depth", "session_ending"):
            rows.append(
                _row(
                    metric,
                    cohort,
                    value=len(valid),
                    settled=is_settled(metric, day, now),
                )
            )
    return rows
