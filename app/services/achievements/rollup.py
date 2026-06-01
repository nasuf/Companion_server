"""End-of-day achievement rollups for exact daily conditions."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.db import db
from app.services.achievements.repository import (
    _day_role_char_counts,
    _day_user_messages,
    _event_count,
    record_event,
    unlock_achievement,
)
from app.services.achievements.schedule_status import has_schedule_status_streak
from app.services.achievements.utils import (
    _day_bounds,
    _field,
    _has_symbol_or_punctuation,
    _local,
    _normalized_message,
    count_chars,
)


async def run_daily_rollup(target_local_day: datetime | None = None) -> None:
    """Run exact/end-of-day achievements for the previous local day."""
    local_day = target_local_day or (_local() - timedelta(days=1))
    start, end = _day_bounds(local_day)
    pairs = await db.query_raw(
        """
        SELECT
            c.user_id,
            c.agent_id,
            MIN(c.workspace_id) AS workspace_id,
            MIN(c.id) AS conversation_id
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.is_deleted = FALSE
          AND m.role = 'user'
          AND m.created_at >= $1::timestamp
          AND m.created_at < $2::timestamp
        GROUP BY c.user_id, c.agent_id
        """,
        start,
        end,
    )
    for pair in pairs:
        conversation_id = str(_field(pair, "conversation_id"))
        user_id = str(_field(pair, "user_id"))
        agent_id = str(_field(pair, "agent_id"))
        workspace_id = _field(pair, "workspace_id")
        rows = await _day_user_messages(user_id, agent_id, start)
        if not rows:
            continue
        counts = [count_chars(str(row["content"])) for row in rows]
        times = [_field(row, "created_at") for row in rows]
        local_times = [_local(ts) for ts in times if ts]
        user_chars, ai_chars = await _day_role_char_counts(user_id, agent_id, start)
        chat_total = user_chars + ai_chars
        if len(rows) == 1:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=2)
        if all(12 <= _local(ts).hour < 14 for ts in times if ts):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=7)
        if len(rows) % 3 == 0:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=28)
        if counts[0] == counts[-1]:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=43)
        if chat_total == 100:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=80)
        if len(rows) >= 20 and all(count <= 10 for count in counts):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=33)
        if len(rows) >= 10 and all(count % 2 == 0 for count in counts):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=45)
        if len(rows) >= 10 and all(count % 2 == 1 for count in counts):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=46)
        if ai_chars and user_chars >= ai_chars * 2:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=40)
        if user_chars and ai_chars > user_chars * 3:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=44)
        if chat_total >= 10000:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=60)
        if times and _is_time_mirror(_local(times[0]), _local(times[-1])):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=90)
        if local_times and (local_times[-1] - local_times[0]).total_seconds() >= 12 * 3600:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=18)
            await _record_day_flag(user_id, agent_id, workspace_id, conversation_id, "span_12h_day", local_day)
            if await _has_consecutive_day_flags(user_id, agent_id, "span_12h_day", local_day, 3):
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=50)
        if sum(1 for ts in local_times if 18 <= ts.hour < 21) >= 3:
            await _record_day_flag(user_id, agent_id, workspace_id, conversation_id, "evening_3_day", local_day)
            if await _has_consecutive_day_flags(user_id, agent_id, "evening_3_day", local_day, 3):
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=6)
        if (
            any(9 <= ts.hour < 10 for ts in local_times)
            and any(19 <= ts.hour < 20 for ts in local_times)
            and any(ts.hour == 23 for ts in local_times)
        ):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=63)
        if len(rows) >= 20 and all(not _has_symbol_or_punctuation(str(row["content"])) for row in rows):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=36)
            await _record_day_flag(user_id, agent_id, workspace_id, conversation_id, "clean_chat_day", local_day)
            if await _has_consecutive_day_flags(user_id, agent_id, "clean_chat_day", local_day, 2):
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=51)
        first_norm = _normalized_message(str(rows[0]["content"]))
        if first_norm:
            await _record_day_flag(user_id, agent_id, workspace_id, conversation_id, "first_message_norm", local_day, value_text=first_norm)
            if await _previous_day_first_message_matches(user_id, agent_id, local_day, first_norm):
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=26)
        if await _has_complete_unique_48h_window(user_id, agent_id, local_day):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=5)
        if any(ts.hour >= 23 or ts.hour < 7 for ts in local_times):
            await _record_day_flag(user_id, agent_id, workspace_id, conversation_id, "sleep_disturb_day", local_day)
        elif len(local_times) >= 3:
            await _record_day_flag(user_id, agent_id, workspace_id, conversation_id, "sleep_respect_day", local_day)
            if await _has_consecutive_day_flags(user_id, agent_id, "sleep_respect_day", local_day, 7):
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=56)
        if await has_schedule_status_streak(user_id=user_id, agent_id=agent_id, local_day=local_day, days=7):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=74)
        if await has_schedule_status_streak(user_id=user_id, agent_id=agent_id, local_day=local_day, days=30):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=93)


def _is_time_mirror(first: datetime, last: datetime) -> bool:
    return f"{first.hour:02d}{first.minute:02d}" == f"{last.hour:02d}{last.minute:02d}"[::-1]


async def _record_day_flag(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str | None,
    event_type: str,
    local_day: datetime,
    *,
    value_text: str | None = None,
) -> None:
    local_date = local_day.date()
    await record_event(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        event_type=event_type,
        source_id=f"{event_type}:{local_date.isoformat()}",
        value_text=value_text,
        occurred_at=_day_bounds(local_day)[0],
    )


async def _has_consecutive_day_flags(
    user_id: str,
    agent_id: str,
    event_type: str,
    local_day: datetime,
    days: int,
) -> bool:
    for offset in range(days):
        source_id = f"{event_type}:{(local_day.date() - timedelta(days=offset)).isoformat()}"
        rows = await db.query_raw(
            """
            SELECT 1
            FROM achievement_events
            WHERE user_id = $1 AND agent_id = $2 AND event_type = $3 AND source_id = $4
            LIMIT 1
            """,
            user_id,
            agent_id,
            event_type,
            source_id,
        )
        if not rows:
            return False
    return True


async def _previous_day_first_message_matches(
    user_id: str,
    agent_id: str,
    local_day: datetime,
    first_norm: str,
) -> bool:
    source_id = f"first_message_norm:{(local_day.date() - timedelta(days=1)).isoformat()}"
    rows = await db.query_raw(
        """
        SELECT value_text
        FROM achievement_events
        WHERE user_id = $1 AND agent_id = $2 AND event_type = 'first_message_norm' AND source_id = $3
        LIMIT 1
        """,
        user_id,
        agent_id,
        source_id,
    )
    return bool(rows and _field(rows[0], "value_text") == first_norm)


async def _has_complete_unique_48h_window(
    user_id: str,
    agent_id: str,
    local_day: datetime,
) -> bool:
    """Check the completed two-local-day window ending at local_day."""
    first_day = local_day - timedelta(days=1)
    start, _ = _day_bounds(first_day)
    _, end = _day_bounds(local_day)
    rows = await db.query_raw(
        """
        SELECT m.content, m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'user'
          AND m.created_at >= $3::timestamp
          AND m.created_at < $4::timestamp
        ORDER BY m.created_at ASC
        """,
        user_id,
        agent_id,
        start,
        end,
    )
    normalized = []
    active_days = set()
    for row in rows:
        value = _normalized_message(str(_field(row, "content") or ""))
        if not value:
            continue
        normalized.append(value)
        created_at = _field(row, "created_at")
        if created_at:
            active_days.add(_local(created_at).date())
    return len(active_days) >= 2 and len(normalized) >= 2 and len(normalized) == len(set(normalized))
