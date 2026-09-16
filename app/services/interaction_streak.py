"""Workspace-scoped consecutive interaction days.

A day counts when the user sent at least one message (or filled the gap with a
makeup card). AI-only days do not count. Scoped to workspace so deleting an AI
and creating a new one starts from zero.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
import logging
from typing import Any
from uuid import uuid4

from app.db import db
from app.observability.events import EVT_INTERACTION_MAKEUP, EVT_INTERACTION_RECORDED
from app.redis_client import get_redis
from app.services.store_catalog import MAKEUP_CARD_KIND
from app.services.store_inventory import batch_summary, consume_batch_units
from app.services.user_activity import local_activity_date

logger = logging.getLogger(__name__)

SOURCE_USER_MESSAGE = "user_message"
SOURCE_MAKEUP = "makeup"
MAKEUP_LOOKBACK_DAYS = 30
_ACTIVITY_DAY_TTL_S = 90_000  # ~25h — covers UTC+8 day boundary drift
_STREAK_LOOKBACK_DAYS = 400


class MakeupError(ValueError):
    """Raised with a stable code the HTTP layer maps to status + copy."""


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text[:10])


def _as_local_date(value: Any) -> date | None:
    """Fold a timestamptz / naive-UTC datetime into the UTC+8 calendar day."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        current = value if value.tzinfo else value.replace(tzinfo=UTC)
        return local_activity_date(current)
    return _as_date(value)


def streak_from_dates(marked: set[date], today: date) -> int:
    """Consecutive marked days ending today, or yesterday if today is still open."""
    start = today if today in marked else today - timedelta(days=1)
    if start not in marked:
        return 0
    streak = 0
    expected = start
    while expected in marked:
        streak += 1
        expected -= timedelta(days=1)
    return streak


def is_makeup_eligible(
    day: date,
    *,
    today: date,
    workspace_created_on: date | None,
    already_marked: bool,
) -> bool:
    if already_marked:
        return False
    if day >= today:
        return False
    if workspace_created_on is not None and day < workspace_created_on:
        return False
    if (today - day).days > MAKEUP_LOOKBACK_DAYS:
        return False
    return True


def _redis_gate_key(workspace_id: str, day: date) -> str:
    return f"interaction:recorded:{workspace_id}:{day.isoformat()}"


async def record_user_message_day(
    workspace_id: str | None,
    user_id: str | None,
    *,
    now: datetime | None = None,
) -> bool:
    """Mark today as interacted. Fail-open: never raise into the chat hot path."""
    if not workspace_id or not user_id:
        return False
    day = local_activity_date(now)
    redis = None
    gated = False
    try:
        redis = await get_redis()
        if not await redis.set(
            _redis_gate_key(workspace_id, day), "1", nx=True, ex=_ACTIVITY_DAY_TTL_S
        ):
            return False
        gated = True
    except Exception as exc:
        logger.debug(
            "interaction redis gate miss workspace=%s: %r",
            workspace_id,
            exc,
        )
    try:
        inserted = await db.query_raw(
            """
            INSERT INTO workspace_interaction_days (
                id, workspace_id, user_id, local_date, source, created_at
            )
            VALUES ($1, $2, $3, $4::date, $5, CURRENT_TIMESTAMP)
            ON CONFLICT (workspace_id, local_date) DO NOTHING
            RETURNING id
            """,
            str(uuid4()),
            workspace_id,
            user_id,
            day.isoformat(),
            SOURCE_USER_MESSAGE,
        )
        if inserted:
            logger.info(
                "interaction day recorded",
                extra={
                    "event": EVT_INTERACTION_RECORDED,
                    "workspace_id": workspace_id,
                    "source": SOURCE_USER_MESSAGE,
                },
            )
        return bool(inserted)
    except Exception as exc:
        logger.warning(
            "record_user_message_day failed workspace=%s: %r",
            workspace_id,
            exc,
        )
        if gated and redis is not None:
            try:
                await redis.delete(_redis_gate_key(workspace_id, day))
            except Exception:
                pass
        return False


async def record_user_message_day_for_conversation(
    conversation_id: str | None,
    *,
    workspace_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """Hot-path helper: prefer ids the caller already has, else load once."""
    ws_id = workspace_id
    uid = user_id
    if (not ws_id or not uid) and conversation_id:
        try:
            conv = await db.conversation.find_unique(where={"id": conversation_id})
        except Exception as exc:
            logger.warning(
                "interaction lookup failed conversation=%s: %r",
                conversation_id,
                exc,
            )
            conv = None
        if conv is not None:
            ws_id = ws_id or getattr(conv, "workspaceId", None)
            uid = uid or getattr(conv, "userId", None)
    await record_user_message_day(ws_id, uid)


async def _load_marked_rows(
    workspace_id: str,
    *,
    start: date,
    end: date,
) -> dict[date, str]:
    rows = await db.query_raw(
        """
        SELECT local_date AS "localDate", source
        FROM workspace_interaction_days
        WHERE workspace_id = $1
          AND local_date >= $2::date
          AND local_date <= $3::date
        """,
        workspace_id,
        start.isoformat(),
        end.isoformat(),
    )
    marked: dict[date, str] = {}
    for row in rows:
        payload = row if isinstance(row, dict) else None
        raw_day = (
            payload.get("localDate") if payload is not None
            else getattr(row, "localDate", None)
        )
        day = _as_date(raw_day)
        if day is None:
            continue
        source = (
            payload.get("source") if payload is not None
            else getattr(row, "source", None)
        )
        marked[day] = str(source or SOURCE_USER_MESSAGE)
    return marked


async def get_current_streak(
    workspace_id: str | None,
    *,
    today: date | None = None,
) -> int:
    if not workspace_id:
        return 0
    current = today or local_activity_date()
    try:
        marked = await _load_marked_rows(
            workspace_id,
            start=current - timedelta(days=_STREAK_LOOKBACK_DAYS),
            end=current,
        )
    except Exception as exc:
        logger.warning(
            "get_current_streak failed workspace=%s: %r",
            workspace_id,
            exc,
        )
        return 0
    return streak_from_dates(set(marked), current)


async def get_interaction_overview(
    workspace_id: str,
    user_id: str,
    *,
    year: int | None = None,
    month: int | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    workspace = await db.chatworkspace.find_unique(where={"id": workspace_id})
    if workspace is None:
        raise MakeupError("workspace_not_found")
    current = today or local_activity_date()
    view_year = year or current.year
    view_month = month or current.month
    last_day = monthrange(view_year, view_month)[1]
    month_start = date(view_year, view_month, 1)
    month_end = date(view_year, view_month, last_day)
    created_on = _as_local_date(getattr(workspace, "createdAt", None))
    load_start = min(month_start, current - timedelta(days=_STREAK_LOOKBACK_DAYS))
    load_end = max(month_end, current)
    marked = await _load_marked_rows(workspace_id, start=load_start, end=load_end)
    summary = await batch_summary(user_id, MAKEUP_CARD_KIND)
    days: list[dict[str, Any]] = []
    cursor = month_start
    while cursor <= month_end:
        source = marked.get(cursor)
        days.append(
            {
                "date": cursor.isoformat(),
                "source": source,
                "makeup_eligible": is_makeup_eligible(
                    cursor,
                    today=current,
                    workspace_created_on=created_on,
                    already_marked=source is not None,
                ),
            }
        )
        cursor += timedelta(days=1)
    return {
        "current_streak": streak_from_dates(set(marked), current),
        "today": current.isoformat(),
        "today_marked": current in marked,
        "makeup_cards": int(summary.get("quantity") or 0),
        "lookback_days": MAKEUP_LOOKBACK_DAYS,
        "workspace_created_on": created_on.isoformat() if created_on else None,
        "month": f"{view_year:04d}-{view_month:02d}",
        "days": days,
    }


def _validate_makeup_day(
    day: date,
    *,
    today: date,
    workspace_created_on: date | None,
) -> None:
    if day >= today:
        raise MakeupError("cannot_makeup_today" if day == today else "cannot_makeup_future")
    if workspace_created_on is not None and day < workspace_created_on:
        raise MakeupError("before_workspace")
    if (today - day).days > MAKEUP_LOOKBACK_DAYS:
        raise MakeupError("outside_lookback")


async def apply_makeup(
    workspace_id: str,
    user_id: str,
    day: date,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    workspace = await db.chatworkspace.find_unique(where={"id": workspace_id})
    if workspace is None:
        raise MakeupError("workspace_not_found")
    if getattr(workspace, "status", "active") != "active":
        raise MakeupError("workspace_not_found")
    current = today or local_activity_date()
    created_on = _as_local_date(getattr(workspace, "createdAt", None))
    _validate_makeup_day(day, today=current, workspace_created_on=created_on)

    async with db.tx() as tx:
        inserted = await tx.query_raw(
            """
            INSERT INTO workspace_interaction_days (
                id, workspace_id, user_id, local_date, source, created_at
            )
            VALUES ($1, $2, $3, $4::date, $5, CURRENT_TIMESTAMP)
            ON CONFLICT (workspace_id, local_date) DO NOTHING
            RETURNING id
            """,
            str(uuid4()),
            workspace_id,
            user_id,
            day.isoformat(),
            SOURCE_MAKEUP,
        )
        if not inserted:
            raise MakeupError("already_marked")
        try:
            await consume_batch_units(
                user_id, MAKEUP_CARD_KIND, 1, client=tx
            )
        except ValueError as exc:
            if "insufficient_inventory" in str(exc):
                raise MakeupError("insufficient_inventory") from exc
            raise

    logger.info(
        "interaction makeup applied",
        extra={
            "event": EVT_INTERACTION_MAKEUP,
            "workspace_id": workspace_id,
            "source": SOURCE_MAKEUP,
        },
    )
    streak = await get_current_streak(workspace_id, today=current)
    remaining = await batch_summary(user_id, MAKEUP_CARD_KIND)
    today_rows = await db.query_raw(
        """
        SELECT 1
        FROM workspace_interaction_days
        WHERE workspace_id = $1 AND local_date = $2::date
        """,
        workspace_id,
        current.isoformat(),
    )
    return {
        "current_streak": streak,
        "today": current.isoformat(),
        "today_marked": bool(today_rows),
        "makeup_cards": int(remaining.get("quantity") or 0),
        "day": {
            "date": day.isoformat(),
            "source": SOURCE_MAKEUP,
            "makeup_eligible": False,
        },
    }
