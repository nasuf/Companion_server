"""Persistence helpers for achievement events and unlocks."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from app.db import db
from app.redis_client import get_redis
from app.services.achievements.definitions import ACHIEVEMENT_BY_ID, ACHIEVEMENTS
from app.services.achievements.mode import (
    achievement_alerts_enabled,
    achievement_evaluation_enabled,
)
from app.services.achievements.rule_registry import (
    ACHIEVEMENT_RULES,
    DISABLED_ACHIEVEMENT_IDS,
)
from app.services.achievements.utils import _day_bounds, _field, _json, _local, _now, count_chars
from app.services.runtime.ws_manager import manager

logger = logging.getLogger(__name__)

_UNLOCKED_CACHE_TTL_S = 86400 * 30


def _unlocked_cache_key(user_id: str, agent_id: str) -> str:
    return f"achievements:unlocked:{user_id}:{agent_id}"


async def _is_unlock_cached(user_id: str, agent_id: str, achievement_id: int) -> bool:
    try:
        redis = await get_redis()
        return bool(await redis.sismember(_unlocked_cache_key(user_id, agent_id), str(achievement_id)))
    except Exception as e:
        logger.debug(f"[ACH] unlock cache read skipped id={achievement_id}: {e}")
        return False


async def _cache_unlocked_achievements(user_id: str, agent_id: str, achievement_ids: list[int] | set[int]) -> None:
    if not achievement_ids:
        return
    try:
        redis = await get_redis()
        key = _unlocked_cache_key(user_id, agent_id)
        await redis.sadd(key, *[str(achievement_id) for achievement_id in achievement_ids])
        await redis.expire(key, _UNLOCKED_CACHE_TTL_S)
    except Exception as e:
        logger.debug(f"[ACH] unlock cache write skipped: {e}")


async def record_event(
    *,
    user_id: str,
    agent_id: str,
    event_type: str,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    source_id: str | None = None,
    value_int: int | None = None,
    value_text: str | None = None,
    metadata: dict | None = None,
    occurred_at: datetime | None = None,
) -> bool:
    """Persist a lightweight achievement event. Returns False for duplicate source events."""
    try:
        rows = await db.query_raw(
            """
            INSERT INTO achievement_events (
                user_id, agent_id, workspace_id, conversation_id, event_type,
                source_id, value_int, value_text, metadata, occurred_at
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9::jsonb, COALESCE($10::timestamp, CURRENT_TIMESTAMP)
            )
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            user_id,
            agent_id,
            workspace_id,
            conversation_id,
            event_type,
            source_id,
            value_int,
            value_text,
            _json(metadata),
            occurred_at,
        )
        return bool(rows)
    except Exception as e:
        logger.warning(f"[ACH] record_event failed type={event_type}: {e}")
        return False


async def unlock_achievement(
    *,
    user_id: str,
    agent_id: str,
    achievement_id: int,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
    metadata: dict | None = None,
    notify: bool = True,
) -> bool:
    if not await achievement_evaluation_enabled():
        return False
    definition = ACHIEVEMENT_BY_ID.get(achievement_id)
    if not definition:
        return False
    rule = ACHIEVEMENT_RULES.get(achievement_id)
    if not rule or not rule.enabled:
        return False
    if await _is_unlock_cached(user_id, agent_id, achievement_id):
        return False
    try:
        rows = await db.query_raw(
            """
            INSERT INTO achievement_unlocks (
                user_id, agent_id, workspace_id, conversation_id, achievement_id, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (user_id, agent_id, achievement_id) DO NOTHING
            RETURNING id, unlocked_at
            """,
            user_id,
            agent_id,
            workspace_id,
            conversation_id,
            achievement_id,
            _json(metadata),
        )
    except Exception as e:
        logger.warning(f"[ACH] unlock failed id={achievement_id}: {e}")
        return False
    if not rows:
        await _cache_unlocked_achievements(user_id, agent_id, {achievement_id})
        return False
    await _cache_unlocked_achievements(user_id, agent_id, {achievement_id})
    unlocked_at = _field(rows[0], "unlocked_at") or _now()
    payload = {
        **definition.to_dict(),
        "achievement_id": definition.id,
        "enabled": True,
        "unlocked": True,
        "unlocked_at": unlocked_at.isoformat() if hasattr(unlocked_at, "isoformat") else str(unlocked_at),
    }
    # Silent mode ("H5 静默计算") persists the unlock above but mutes the
    # unlock moments (chat WS popup + APNs push); switching back to "on"
    # must NOT retro-notify.
    if notify and await achievement_alerts_enabled():
        try:
            from app.services.notifications.service import notify_achievement_unlocked
            from app.services.runtime.tasks import fire_background

            fire_background(notify_achievement_unlocked(
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                achievement_id=achievement_id,
                title=definition.name,
            ))
            delivered = False
            if conversation_id:
                delivered = await manager.send_event(conversation_id, "achievement_unlocked", payload)
            if not delivered and workspace_id:
                delivered = bool(await manager.send_to_workspace(workspace_id, "achievement_unlocked", payload))
            if delivered:
                await db.execute_raw(
                    "UPDATE achievement_unlocks SET notified_at = CURRENT_TIMESTAMP WHERE user_id = $1 AND agent_id = $2 AND achievement_id = $3",
                    user_id,
                    agent_id,
                    achievement_id,
                )
        except Exception as e:
            logger.debug(f"[ACH] notify skipped id={achievement_id}: {e}")
    return True


async def list_achievements(user_id: str, agent_id: str) -> dict:
    rows = await db.query_raw(
        """
        SELECT achievement_id, unlocked_at
        FROM achievement_unlocks
        WHERE user_id = $1 AND agent_id = $2
        """,
        user_id,
        agent_id,
    )
    unlocked = {
        achievement_id: _field(row, "unlocked_at")
        for row in rows
        if (
            (achievement_id := int(_field(row, "achievement_id")))
            in ACHIEVEMENT_RULES
            and ACHIEVEMENT_RULES[achievement_id].enabled
        )
    }
    await _cache_unlocked_achievements(user_id, agent_id, set(unlocked.keys()))
    items = []
    score = 0
    for definition in ACHIEVEMENTS:
        unlocked_at = unlocked.get(definition.id)
        rule = ACHIEVEMENT_RULES[definition.id]
        if unlocked_at:
            score += definition.score
        items.append({
            **definition.to_dict(),
            "achievement_id": definition.id,
            "enabled": rule.enabled,
            "unlocked": bool(unlocked_at),
            "unlocked_at": unlocked_at.isoformat() if hasattr(unlocked_at, "isoformat") else unlocked_at,
        })
    return {
        "total": len(items),
        "active_total": sum(1 for rule in ACHIEVEMENT_RULES.values() if rule.enabled),
        "disabled_total": len(DISABLED_ACHIEVEMENT_IDS),
        "unlocked": len(unlocked),
        "score": score,
        "items": items,
    }


async def _recent_user_messages(
    user_id: str,
    agent_id: str,
    *,
    since: datetime | None = None,
    limit: int = 260,
) -> list[dict]:
    where_since = "AND m.created_at >= $3::timestamp" if since else ""
    args: list[Any] = [user_id, agent_id]
    if since:
        args.append(since)
    rows = await db.query_raw(
        f"""
        SELECT m.id, m.content, m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'user'
          {where_since}
        ORDER BY m.created_at ASC
        LIMIT {int(limit)}
        """,
        *args,
    )
    return [dict(row) if isinstance(row, dict) else row.__dict__ for row in rows]


async def _day_user_messages(
    user_id: str,
    agent_id: str,
    at: datetime | None = None,
    *,
    through: datetime | None = None,
) -> list[dict]:
    start, end = _day_bounds(_local(at))
    through_at = through or end
    rows = await db.query_raw(
        """
        SELECT m.id, m.content, m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role = 'user'
          AND m.created_at >= $3::timestamp
          AND m.created_at < $4::timestamp
          AND m.created_at <= $5::timestamp
        ORDER BY m.created_at ASC
        """,
        user_id,
        agent_id,
        start,
        end,
        through_at,
    )
    return [dict(row) if isinstance(row, dict) else row.__dict__ for row in rows]


async def _day_role_char_counts(user_id: str, agent_id: str, at: datetime | None = None) -> tuple[int, int]:
    start, end = _day_bounds(_local(at))
    rows = await db.query_raw(
        """
        SELECT m.role, m.content
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.created_at >= $3::timestamp
          AND m.created_at < $4::timestamp
        """,
        user_id,
        agent_id,
        start,
        end,
    )
    user_chars = 0
    ai_chars = 0
    for row in rows:
        chars = count_chars(str(_field(row, "content") or ""))
        if _field(row, "role") == "user":
            user_chars += chars
        elif _field(row, "role") == "assistant":
            ai_chars += chars
    return user_chars, ai_chars


async def _day_role_message_counts(
    user_id: str,
    agent_id: str,
    at: datetime | None = None,
) -> tuple[int, int]:
    start, end = _day_bounds(_local(at))
    rows = await db.query_raw(
        """
        SELECT m.role, COUNT(*) AS count
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.user_id = $1
          AND c.agent_id = $2
          AND c.is_deleted = FALSE
          AND m.role IN ('user', 'assistant')
          AND m.created_at >= $3::timestamp
          AND m.created_at < $4::timestamp
        GROUP BY m.role
        """,
        user_id,
        agent_id,
        start,
        end,
    )
    counts = {
        str(_field(row, "role") or ""): int(_field(row, "count", 0) or 0)
        for row in rows
    }
    return counts.get("user", 0), counts.get("assistant", 0)


async def _event_count(user_id: str, agent_id: str, event_type: str) -> int:
    rows = await db.query_raw(
        """
        SELECT COUNT(*) AS count
        FROM achievement_events e
        LEFT JOIN conversations c ON c.id = e.conversation_id
        WHERE e.user_id = $1
          AND e.agent_id = $2
          AND e.event_type = $3
          AND (e.conversation_id IS NULL OR c.is_deleted = FALSE)
        """,
        user_id,
        agent_id,
        event_type,
    )
    return int(_field(rows[0], "count", 0)) if rows else 0



async def _memory_count(user_id: str, workspace_id: str | None, main: str, sub: str | None = None) -> int:
    rows = await db.query_raw(
        """
        SELECT COUNT(*) AS count
        FROM memories_user
        WHERE user_id = $1
          AND ($2 IS NULL OR workspace_id = $2)
          AND is_archived = FALSE
          AND main_category = $3
          AND ($4::text IS NULL OR sub_category = $4)
        """,
        user_id,
        workspace_id,
        main,
        sub,
    )
    return int(_field(rows[0], "count", 0)) if rows else 0


async def _birthday_mmdd(user_id: str, workspace_id: str | None, *, source: str) -> tuple[int, int] | None:
    table = "memories_ai" if source == "ai" else "memories_user"
    rows = await db.query_raw(
        f"""
        SELECT content
        FROM {table}
        WHERE user_id = $1
          AND ($2 IS NULL OR workspace_id = $2)
          AND is_archived = FALSE
          AND main_category = '身份'
          AND sub_category = '生日'
        ORDER BY level ASC, created_at ASC
        LIMIT 1
        """,
        user_id,
        workspace_id,
    )
    if not rows:
        return None
    text = str(_field(rows[0], "content") or "")
    match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})", text)
    if not match:
        match = re.search(r"(\d{1,2})[/-](\d{1,2})", text)
    if not match:
        return None
    month, day = int(match.group(1)), int(match.group(2))
    if 1 <= month <= 12 and 1 <= day <= 31:
        return month, day
    return None
