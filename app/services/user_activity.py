from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
import logging

from app.db import db

logger = logging.getLogger(__name__)

_LOCAL_TZ = timezone(timedelta(hours=8))


class UserActivityWriteError(RuntimeError):
    """Raised when an online heartbeat could not be recorded anywhere."""


def local_activity_date(now: datetime | None = None) -> date:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(_LOCAL_TZ).date()


async def record_user_activity(
    user_id: str,
    *,
    source: str,
    now: datetime | None = None,
    raise_on_total_failure: bool = False,
) -> None:
    """Record one app/auth activity heartbeat per local calendar day.

    A unique (user_id, local_date) row makes "连续未登录天数" deterministic:
    same-day repeated app opens only bump seen_count/last_seen_at, while
    cross-day gaps remain visible to the last-will trigger scanner.
    """
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    day = local_activity_date(current)

    user_seen_error: Exception | None = None
    ledger_error: Exception | None = None

    try:
        await db.execute_raw(
            """
            UPDATE users
            SET last_seen_at = $2::timestamp, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            user_id,
            current,
        )
    except Exception as exc:
        user_seen_error = exc
        logger.warning(
            "record_user_activity failed to touch user last_seen_at for user=%s source=%s: %r",
            user_id,
            source,
            exc,
        )

    try:
        await db.execute_raw(
            """
            INSERT INTO user_daily_activity (
                id, user_id, local_date, source, seen_count,
                first_seen_at, last_seen_at, created_at, updated_at
            )
            VALUES (
                gen_random_uuid(), $1, $2::date, $3, 1,
                $4::timestamp, $4::timestamp, $4::timestamp, $4::timestamp
            )
            ON CONFLICT (user_id, local_date) DO UPDATE SET
                last_seen_at = EXCLUDED.last_seen_at,
                source = EXCLUDED.source,
                seen_count = user_daily_activity.seen_count + 1,
                updated_at = EXCLUDED.updated_at
            """,
            user_id,
            day.isoformat(),
            source[:40],
            current,
        )
    except Exception as exc:
        ledger_error = exc
        logger.warning(
            "record_user_activity failed to upsert daily ledger for user=%s source=%s: %r",
            user_id,
            source,
            exc,
        )
    if user_seen_error is not None and ledger_error is not None and raise_on_total_failure:
        raise UserActivityWriteError(
            f"failed to record activity heartbeat for user={user_id}"
        ) from ledger_error


async def get_login_streak_days(user_id: str, *, today: date | None = None) -> int:
    """Return consecutive active local days ending today."""
    current_day = today or local_activity_date()
    rows = await db.query_raw(
        """
        SELECT local_date AS "localDate"
        FROM user_daily_activity
        WHERE user_id = $1 AND local_date <= $2::date
        ORDER BY local_date DESC
        LIMIT 366
        """,
        user_id,
        current_day.isoformat(),
    )
    streak = 0
    expected = current_day
    for row in rows:
        raw = row.get("localDate") if isinstance(row, dict) else getattr(row, "localDate", None)
        if raw is None:
            continue
        active_day = raw if isinstance(raw, date) else date.fromisoformat(str(raw)[:10])
        if active_day != expected:
            break
        streak += 1
        expected = expected - timedelta(days=1)
    return streak
