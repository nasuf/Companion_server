"""Processing-time monitor: intentional waits are excluded, empty windows aren't recovery."""

from datetime import datetime, timedelta

from app.config import settings
from app.services.ops.recovery import timestamp


def classify(rows: list[dict], now: datetime, carry: dict) -> dict:
    limit = settings.ops_reply_slow_seconds * 1000
    window_start = now - timedelta(seconds=settings.ops_reply_window_seconds)
    window = [r for r in rows if timestamp(r["at"]) >= window_start]
    values = sorted(float(r["ms"]) for r in window)
    n = len(values)
    # PostgreSQL percentile_cont semantics; a small window is never a percentile alert.
    p90 = None
    if n:
        pos = (n - 1) * 0.9
        i = int(pos)
        p90 = (values[i] + (values[min(i + 1, n - 1)] - values[i]) * (pos - i)) / 1000
    slow = [r for r in window if r["ms"] > limit]
    eligible = [
        r
        for r in window
        if not carry.get("recovered_through")
        or timestamp(r["at"]) > timestamp(carry["recovered_through"])
    ]
    eligible_values = sorted(float(r["ms"]) for r in eligible)
    eligible_p90 = None
    if eligible_values:
        pos = (len(eligible_values) - 1) * 0.9
        i = int(pos)
        eligible_p90 = (
            eligible_values[i]
            + (
                eligible_values[min(i + 1, len(eligible_values) - 1)]
                - eligible_values[i]
            )
            * (pos - i)
        ) / 1000
    raised = sum(
        r["ms"] > limit for r in eligible
    ) >= settings.ops_reply_slow_count or (
        len(eligible) >= settings.ops_reply_p90_min_samples
        and eligible_p90 > settings.ops_reply_p90_seconds
    )
    prior = carry.get("since")
    # Retain the initial incident anchor. Moving it on every still-failing
    # window would make recovery wait for the window to expire unnecessarily.
    since = prior or (
        max(timestamp(r["at"]) for r in window).isoformat() if raised else None
    )
    later = [r for r in rows if since and timestamp(r["at"]) > timestamp(since)]
    streak = 0
    for row in sorted(later, key=lambda r: (timestamp(r["at"]), r["id"])):
        streak = streak + 1 if row["ms"] <= limit else 0
    recovered = bool(prior) and streak >= settings.ops_reply_recovery_samples
    # A later slow reply ends a successful streak, even within this sample.
    alerting = (raised or bool(prior)) and not recovered
    return {
        "count": len(slow),
        "threshold": settings.ops_reply_slow_count,
        "timed_turns": n,
        "p90_seconds": p90,
        "p90_threshold_seconds": settings.ops_reply_p90_seconds,
        "p90_min_samples": settings.ops_reply_p90_min_samples,
        "slow_after_seconds": settings.ops_reply_slow_seconds,
        "window_seconds": settings.ops_reply_window_seconds,
        "recovery_turns": min(streak, settings.ops_reply_recovery_samples),
        "recovery_required_turns": settings.ops_reply_recovery_samples,
        "sample_state": "measured" if n else "no_sample",
        "alerting": alerting,
        "_carry": {
            "since": since if alerting else None,
            "recovered_through": max(timestamp(r["at"]) for r in later).isoformat()
            if recovered
            else carry.get("recovered_through"),
        },
    }


async def measure(now: datetime, carry: dict) -> dict:
    from app.services.ops.service_health import _query, _naive_utc

    # The latest bounded page also suffices to establish a consecutive streak;
    # incomplete windows explicitly report unknown instead of optimistic p90.
    rows = await _query(
        """
      WITH turns AS (
        SELECT DISTINCT ON (conversation_id,d->>'first_bubble_at') id,created_at AS at,
          CASE WHEN jsonb_typeof(d->'generation_ms')='number' AND jsonb_typeof(d->'in_graph_delay_ms')='number'
            THEN (d->>'generation_ms')::numeric - (d->>'in_graph_delay_ms')::numeric END AS ms
        FROM messages m CROSS JOIN LATERAL (SELECT metadata->'response_diagnostics' AS d) x
        WHERE role='assistant' AND created_at >= $1::timestamp AND created_at <= $2::timestamp
          AND d->>'timing_definition_version'='server-turn-timing-v1'
          AND pg_input_is_valid(d->>'first_bubble_at','timestamptz')
          AND NOT (COALESCE(metadata,'{}'::jsonb) ? 'human_takeover')
        ORDER BY conversation_id,d->>'first_bubble_at',created_at,id
      ) SELECT id,at,ms::float8 FROM turns WHERE ms>=0 ORDER BY at DESC,id DESC LIMIT 1001
    """,
        _naive_utc(
            min(
                now - timedelta(seconds=settings.ops_reply_window_seconds),
                timestamp(carry["since"]) if carry.get("since") else now,
            )
        ),
        _naive_utc(now),
    )
    result = classify(rows[:1000], now, carry)
    if len(rows) > 1000:
        result.update(status="unknown", truncated=True)
    return result
