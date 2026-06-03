"""Achievement rules driven by persisted user chat messages."""

from __future__ import annotations

from datetime import datetime, time, timezone

from app.db import db
from app.services.achievements.events import UserMessageAchievementEvent
from app.services.achievements.repository import (
    _birthday_mmdd,
    _day_role_char_counts,
    _day_user_messages,
    _event_count,
    record_event,
    unlock_achievement,
)
from app.services.achievements.schedule_status import (
    has_schedule_status_streak,
    record_schedule_status_chat,
)
from app.services.achievements.utils import (
    QUESTION_END,
    _aware,
    _day_bounds,
    _field,
    _first_counted_char,
    _local,
    _normalized_message,
    _now,
    count_chars,
)

FUTURE_PLAN_CUES = ("之后有什么安排", "接下来有什么安排", "明天干嘛", "今晚干嘛", "计划")


async def evaluate_user_message(event: UserMessageAchievementEvent) -> None:
    await _evaluate_user_message(
        user_id=event.user_id,
        agent_id=event.agent_id,
        workspace_id=event.workspace_id,
        conversation_id=event.conversation_id,
        message_id=event.message_id,
        text=event.text,
        agent_name=event.agent_name,
        reply_context=event.reply_context,
        aggregation_route=event.aggregation_route,
        component_card=event.component_card,
        occurred_at=event.occurred_at,
    )


async def _evaluate_user_message(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    message_id: str,
    text: str,
    agent_name: str | None = None,
    reply_context: dict | None = None,
    aggregation_route: str | None = None,
    component_card: dict | None = None,
    occurred_at: datetime | None = None,
) -> None:
    occurred_at = await _message_created_at(message_id, user_id, agent_id) or _aware(occurred_at or _now())
    char_count = count_chars(text)
    normalized = _normalized_message(text)
    await record_event(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        event_type="user_message",
        source_id=message_id,
        value_int=char_count,
        value_text=normalized,
        metadata={"raw_len": len(text), "aggregation_route": aggregation_route},
        occurred_at=occurred_at,
    )

    await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=1)
    if component_card:
        await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="component_card_sent", source_id=message_id)

    today = await _day_user_messages(user_id, agent_id, occurred_at)

    if normalized == "哈哈":
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=9)
    if text.rstrip().endswith("～"):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=27)
    if any(cue in text for cue in FUTURE_PLAN_CUES):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=21)
    if aggregation_route == "fragment_window":
        await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="aggregation_fragment", source_id=message_id)
        if await _event_count(user_id, agent_id, "aggregation_fragment") >= 50:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=86)

    local = _local(occurred_at)
    schedule_bucket = await record_schedule_status_chat(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        message_id=message_id,
        occurred_at=occurred_at,
    )
    if schedule_bucket:
        if schedule_bucket == "sleep":
            await record_event(
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                event_type="sleep_status_message",
                source_id=message_id,
            )
            if await _event_count(user_id, agent_id, "sleep_status_message") >= 10:
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=48)
        if await has_schedule_status_streak(user_id=user_id, agent_id=agent_id, local_day=local, days=7):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=74)
        if await has_schedule_status_streak(user_id=user_id, agent_id=agent_id, local_day=local, days=30):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=93)
    if local.hour == 5 and local.minute == 20:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=82)
    if local.hour == 13 and local.minute == 14:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=76)
    ai_birthday = await _birthday_mmdd(user_id, workspace_id, source="ai")
    if ai_birthday:
        month, day = ai_birthday
        if local.hour == month and local.minute == day:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=95)
        if local.month == month and local.day == day and "生日快乐" in text:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=83)
    if char_count == sum(int(ch) for ch in f"{local.hour:02d}{local.minute:02d}"):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=69)
    if local.time() >= time(23, 59, 58) or local.time() <= time(0, 0, 2):
        await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="midnight_edge_message", source_id=message_id)
        if await _event_count(user_id, agent_id, "midnight_edge_message") >= 10:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=94)
    if 22 <= local.hour or local.hour < 1:
        if "晚安" in text:
            await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="goodnight_late", source_id=message_id)
            if await _event_count(user_id, agent_id, "goodnight_late") >= 3:
                await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=58)
    if 2 <= local.hour < 5:
        await _record_daily_once(user_id, agent_id, workspace_id, conversation_id, "late_night_day", message_id, local)
        if await _event_count(user_id, agent_id, "late_night_day") >= 10:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=52)
    if 5 <= local.hour < 7:
        await _record_daily_once(user_id, agent_id, workspace_id, conversation_id, "early_morning_day", message_id, local)
        if await _event_count(user_id, agent_id, "early_morning_day") >= 10:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=53)
    if normalized == "嗯":
        await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="um_message", source_id=message_id)
        if await _event_count(user_id, agent_id, "um_message") >= 50:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=72)

    if len(today) >= 100:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=64)
    if len(today) >= 200:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=73)
    if _has_scene_experience_windows(today):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=63)
    user_chars, ai_chars = await _day_role_char_counts(user_id, agent_id, occurred_at)
    if user_chars + ai_chars >= 10000:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=60)
    if sum(1 for row in today if str(row["content"]).rstrip().endswith(QUESTION_END)) >= 5:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=31)

    await _check_sequences(user_id, agent_id, workspace_id, conversation_id, today)
    await _check_reply_timing_and_echo(user_id, agent_id, workspace_id, conversation_id, message_id, char_count, occurred_at)
    await _check_proactive_response(user_id, agent_id, workspace_id, conversation_id, message_id, occurred_at)
    await _check_daily_chat_day_milestones(user_id, agent_id, workspace_id, conversation_id)
    await _check_intimacy(user_id, agent_id, workspace_id, conversation_id)


async def _message_created_at(message_id: str, user_id: str, agent_id: str) -> datetime | None:
    try:
        rows = await db.query_raw(
            """
            SELECT m.created_at
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.id = $1
              AND m.role = 'user'
              AND c.user_id = $2
              AND c.agent_id = $3
              AND c.is_deleted = FALSE
            LIMIT 1
            """,
            message_id,
            user_id,
            agent_id,
        )
    except Exception:
        return None
    if not rows:
        return None
    created_at = _field(rows[0], "created_at")
    return _aware(created_at) if created_at else None


async def _record_daily_once(user_id: str, agent_id: str, workspace_id: str | None, conversation_id: str | None, event_type: str, source_id: str, local: datetime) -> None:
    await record_event(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        event_type=event_type,
        source_id=f"{event_type}:{local.date().isoformat()}",
        metadata={"message_id": source_id},
        occurred_at=local.astimezone(timezone.utc),
    )


async def _check_sequences(user_id: str, agent_id: str, workspace_id: str | None, conversation_id: str, rows: list[dict]) -> None:
    texts = [str(row["content"]) for row in rows]
    norms = [_normalized_message(text) for text in texts]
    counts = [count_chars(text) for text in texts]
    if len(counts) >= 4 and all(c <= 5 for c in counts[-4:]):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=8)
    if len(texts) >= 3 and len(set(norms[-3:])) == 1 and len(norms[-1]) > 3:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=57)
    if texts and texts[-1].strip() and texts.count(texts[-1]) >= 10:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=71)
    if len(texts) >= 3 and len({_first_counted_char(t) for t in texts[-3:]}) == 1 and _first_counted_char(texts[-1]):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=19)
    if len(texts) >= 10 and len({_first_counted_char(t) for t in texts[-10:] if _first_counted_char(t)}) == 10:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=37)
    if len(counts) >= 5 and all(a < b for a, b in zip(counts[-5:], counts[-4:])):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=38)
    if len(counts) >= 5 and all(a > b for a, b in zip(counts[-5:], counts[-4:])):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=41)
    if len(counts) >= 2:
        pairs = sum(1 for a, b in zip(counts, counts[1:]) if a >= 12 and b <= 4)
        if pairs >= 3:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=42)
    if len(texts) >= 6:
        common = set(_normalized_message(texts[-1]))
        for text in texts[-6:-1]:
            common &= set(_normalized_message(text))
        if common:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=47)
    if len(texts) >= 10 and all(t.rstrip().endswith(QUESTION_END) for t in texts[-10:]):
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=70)


def _has_scene_experience_windows(rows: list[dict]) -> bool:
    local_times = [_local(_field(row, "created_at")) for row in rows if _field(row, "created_at")]
    return (
        any(9 <= ts.hour < 10 for ts in local_times)
        and any(19 <= ts.hour < 20 for ts in local_times)
        and any(ts.hour == 23 for ts in local_times)
    )


async def _check_reply_timing_and_echo(
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str,
    message_id: str,
    char_count: int,
    at: datetime,
) -> None:
    rows = await db.query_raw(
        """
        SELECT m.id, m.content, m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'assistant'
          AND m.created_at < $3::timestamp
        ORDER BY m.created_at DESC
        LIMIT 1
        """,
        user_id,
        agent_id,
        at,
    )
    if not rows:
        return
    assistant = rows[0]
    assistant_at = _field(assistant, "created_at")
    if assistant_at:
        assistant_at = _aware(assistant_at)
    if assistant_at and (at - assistant_at).total_seconds() <= 10:
        await record_event(
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            event_type="quick_reply_10s",
            source_id=message_id,
        )
        start, end = _day_bounds(_local(at))
        day_rows = await db.query_raw(
            """
            SELECT COUNT(*) AS count
            FROM achievement_events
            WHERE user_id = $1 AND agent_id = $2
              AND event_type = 'quick_reply_10s'
              AND occurred_at >= $3::timestamp AND occurred_at < $4::timestamp
            """,
            user_id,
            agent_id,
            start,
            end,
        )
        if int(_field(day_rows[0], "count", 0)) >= 20 and await _has_quick_reply_streak(user_id, agent_id, at, required=20):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=62)
    if count_chars(str(_field(assistant, "content") or "")) == char_count:
        await record_event(
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            event_type="echo_same_len",
            source_id=message_id,
        )
        if await _has_echo_same_len_streak(user_id, agent_id, at, required=3):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=85)


async def _day_messages_until(user_id: str, agent_id: str, at: datetime, *, limit: int = 260) -> list[dict]:
    start, _ = _day_bounds(_local(at))
    bounded_limit = max(1, min(int(limit), 1000))
    rows = await db.query_raw(
        f"""
        SELECT m.role, m.content, m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role IN ('user', 'assistant')
          AND m.created_at >= $3::timestamp
          AND m.created_at <= $4::timestamp
        ORDER BY m.created_at DESC
        LIMIT {bounded_limit}
        """,
        user_id,
        agent_id,
        start,
        at,
    )
    return list(reversed(rows))


async def _has_quick_reply_streak(user_id: str, agent_id: str, at: datetime, *, required: int) -> bool:
    rows = await _day_messages_until(user_id, agent_id, at)
    pairs = []
    for index, row in enumerate(rows):
        if _field(row, "role") != "assistant":
            continue
        assistant_at = _field(row, "created_at")
        next_user_at = None
        for following in rows[index + 1:]:
            if _field(following, "role") == "user":
                next_user_at = _field(following, "created_at")
                break
        if not assistant_at or not next_user_at:
            pairs.append(False)
            continue
        pairs.append((_aware(next_user_at) - _aware(assistant_at)).total_seconds() <= 10)

    streak = 0
    for ok in reversed(pairs):
        if not ok:
            break
        streak += 1
        if streak >= required:
            return True
    return False


async def _has_echo_same_len_streak(user_id: str, agent_id: str, at: datetime, *, required: int) -> bool:
    rows = await _day_messages_until(user_id, agent_id, at)
    pairs = []
    for index, row in enumerate(rows):
        if _field(row, "role") != "user":
            continue
        previous_assistant = None
        for previous in reversed(rows[:index]):
            if _field(previous, "role") == "assistant":
                previous_assistant = previous
                break
        if not previous_assistant:
            pairs.append(False)
            continue
        pairs.append(
            count_chars(str(_field(row, "content") or ""))
            == count_chars(str(_field(previous_assistant, "content") or ""))
        )

    streak = 0
    for ok in reversed(pairs):
        if not ok:
            break
        streak += 1
        if streak >= required:
            return True
    return False


async def _check_proactive_response(user_id: str, agent_id: str, workspace_id: str | None, conversation_id: str, message_id: str, at: datetime) -> None:
    rows = await db.query_raw(
        """
        SELECT m.id, m.created_at, m.metadata
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'assistant'
          AND m.created_at < $3::timestamp
          AND m.created_at >= $3::timestamp - INTERVAL '24 hours'
          AND (m.metadata->>'proactive')::boolean = TRUE
        ORDER BY m.created_at DESC
        LIMIT 1
        """,
        user_id,
        agent_id,
        at,
    )
    if not rows:
        return
    proactive = rows[0]
    proactive_at = _field(proactive, "created_at")
    between = await db.query_raw(
        """
        SELECT COUNT(*) AS count
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'user'
          AND m.created_at > $3::timestamp
          AND m.created_at < $4::timestamp
        """,
        user_id,
        agent_id,
        proactive_at,
        at,
    )
    if int(_field(between[0], "count", 0)) > 0:
        return
    metadata = _field(proactive, "metadata") or {}
    trigger = metadata.get("trigger_type") if isinstance(metadata, dict) else None
    await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="proactive_replied", source_id=str(_field(proactive, "id")), metadata={"reply_message_id": message_id, "trigger_type": trigger})
    await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=25)
    count = await _event_count(user_id, agent_id, "proactive_replied")
    if count >= 100:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=89)
    if proactive_at:
        proactive_at = _aware(proactive_at)
    if proactive_at and (at - proactive_at).total_seconds() <= 180:
        await record_event(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, event_type="proactive_replied_3min", source_id=str(_field(proactive, "id")))
        if await _all_proactive_messages_replied_quickly(user_id, agent_id, at, required=100):
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=92)
    if trigger == "special_holiday":
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=61)
    elif trigger == "special_birthday":
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=68)
    elif trigger == "special_combined":
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=78)


async def _all_proactive_messages_replied_quickly(
    user_id: str,
    agent_id: str,
    at: datetime,
    *,
    required: int,
) -> bool:
    proactive_rows = await db.query_raw(
        """
        SELECT COUNT(*) AS count
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'assistant'
          AND m.created_at <= $3::timestamp
          AND (m.metadata->>'proactive')::boolean = TRUE
        """,
        user_id,
        agent_id,
        at,
    )
    quick_rows = await db.query_raw(
        """
        SELECT COUNT(*) AS count
        FROM achievement_events
        WHERE user_id = $1
          AND agent_id = $2
          AND event_type = 'proactive_replied_3min'
          AND occurred_at <= $3::timestamp
        """,
        user_id,
        agent_id,
        at,
    )
    proactive_count = int(_field(proactive_rows[0], "count", 0)) if proactive_rows else 0
    quick_count = int(_field(quick_rows[0], "count", 0)) if quick_rows else 0
    return proactive_count >= required and proactive_count == quick_count


async def _check_daily_chat_day_milestones(user_id: str, agent_id: str, workspace_id: str | None, conversation_id: str) -> None:
    rows = await db.query_raw(
        """
        SELECT COUNT(*) AS count
        FROM (
            SELECT (m.created_at AT TIME ZONE 'Asia/Shanghai')::date AS d
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.user_id = $1
              AND c.agent_id = $2
              AND c.is_deleted = FALSE
              AND m.role = 'user'
            GROUP BY d
            HAVING COUNT(*) >= 30
        ) days
        """,
        user_id,
        agent_id,
    )
    days = int(_field(rows[0], "count", 0)) if rows else 0
    for threshold, achievement_id in ((7, 35), (15, 54), (30, 77), (90, 91), (180, 97)):
        if days >= threshold:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=achievement_id)


async def _check_intimacy(user_id: str, agent_id: str, workspace_id: str | None, conversation_id: str | None) -> None:
    rows = await db.query_raw(
        "SELECT growth_intimacy FROM intimacies WHERE user_id = $1 AND agent_id = $2 LIMIT 1",
        user_id,
        agent_id,
    )
    if not rows:
        return
    value = int(_field(rows[0], "growth_intimacy", 0) or 0)
    for threshold, achievement_id in ((401, 39), (601, 67), (801, 75), (1000, 88)):
        if value >= threshold:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, conversation_id=conversation_id, achievement_id=achievement_id)
