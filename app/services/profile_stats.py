from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import db
from app.models.user import ProfileStatsResponse
from app.services.relationship.intimacy import get_intimacy_data


_STAGE_BANDS = [
    (0, 21, "P1", "初见陪伴"),
    (21, 41, "P2", "熟悉陪伴"),
    (41, 61, "P3", "亲近陪伴"),
    (61, 81, "P4", "稳定陪伴"),
    (81, 101, "P5", "深度羁绊"),
]

_GENDER_LABELS = {
    "female": "女",
    "woman": "女",
    "girl": "女",
    "女": "女",
    "male": "男",
    "man": "男",
    "boy": "男",
    "男": "男",
}


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _companion_days(created_at: Any) -> int:
    created = _ensure_utc(created_at)
    if created is None:
        return 0
    return max(1, (datetime.now(UTC).date() - created.astimezone(UTC).date()).days + 1)


def _date_label(created_at: Any) -> str | None:
    created = _ensure_utc(created_at)
    if created is None:
        return None
    return created.astimezone(UTC).strftime("%Y.%m.%d")


def _profile_stage(topic_intimacy: float) -> tuple[str, str]:
    for lo, hi, code, label in _STAGE_BANDS:
        if lo <= topic_intimacy < hi:
            return code, label
    return "P5", "深度羁绊"


def _intimacy_subtitle(topic_intimacy: float) -> str:
    if topic_intimacy >= 81:
        return "默契很深，靠近得很自然"
    if topic_intimacy >= 61:
        return "稳定陪伴，越来越熟悉"
    if topic_intimacy >= 41:
        return "亲近起来，开始有默契"
    if topic_intimacy >= 21:
        return "慢慢靠近，开始有点熟悉"
    return "初见陪伴，故事刚刚开始"


def _duration_label(minutes: int) -> str:
    if minutes <= 0:
        return "0m"
    hours = minutes // 60
    remainder = minutes % 60
    if hours > 0 and remainder > 0:
        return f"{hours}h{remainder}m"
    if hours > 0:
        return f"{hours}h"
    return f"{remainder}m"


def _duration_subtitle(minutes: int) -> str:
    if minutes <= 0:
        return "还没有累计相处时光"
    movie_count = max(1, round(minutes / 112))
    return f"≈ 一起看了{movie_count}场电影"


def _mbti_type(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = value.get("type")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().upper()
    return None


def _gender_label(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return _GENDER_LABELS.get(raw.lower(), _GENDER_LABELS.get(raw, raw))


def _companion_summary(agent: Any | None) -> str:
    parts = ["唯一伴生对象"]
    if agent is not None:
        gender = _gender_label(getattr(agent, "gender", None))
        mbti = _mbti_type(getattr(agent, "currentMbti", None)) or _mbti_type(
            getattr(agent, "mbti", None)
        )
        if gender:
            parts.append(gender)
        if mbti:
            parts.append(mbti)
    return " · ".join(parts)


async def _workspace_message_stats(
    workspace_id: str, user_id: str
) -> tuple[int, int, int, int]:
    since = datetime.now(UTC) - timedelta(days=7)
    rows = await db.query_raw(
        """
        SELECT
          COALESCE(COUNT(m.id), 0)::int AS message_count,
          COALESCE(COUNT(DISTINCT date_trunc('hour', m.created_at)), 0)::int
            AS active_chat_hours,
          COALESCE(COUNT(DISTINCT date_trunc('minute', m.created_at)), 0)::int
            AS active_chat_minutes,
          COALESCE(
            COUNT(m.id) FILTER (WHERE m.created_at >= $3::timestamp),
            0
          )::int AS recent_7d_message_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.workspace_id = $1
          AND c.user_id = $2
          AND c.is_deleted = FALSE
        """,
        workspace_id,
        user_id,
        since,
    )
    row = rows[0] if rows else {}
    return (
        _as_int(_field(row, "message_count")),
        _as_int(_field(row, "active_chat_hours")),
        _as_int(_field(row, "active_chat_minutes")),
        _as_int(_field(row, "recent_7d_message_count")),
    )


async def _backpack_count(user_id: str) -> int:
    rows = await db.query_raw(
        """
        SELECT COALESCE(SUM(quantity), 0)::int AS count
        FROM user_store_inventory
        WHERE user_id = $1
          AND quantity > 0
        """,
        user_id,
    )
    row = rows[0] if rows else {}
    return _as_int(_field(row, "count"))


async def get_profile_stats_for_workspace(
    *, user_id: str, workspace: Any
) -> ProfileStatsResponse:
    workspace_id = getattr(workspace, "id", None) or _field(workspace, "id", "")
    agent_id = getattr(workspace, "agentId", None) or _field(workspace, "agentId")
    agent = None
    if agent_id:
        agent = await db.aiagent.find_unique(where={"id": agent_id})

    topic_intimacy = 0.0
    if agent_id:
        intimacy = await get_intimacy_data(agent_id, user_id)
        topic_intimacy = float(intimacy.get("topic_intimacy") or 0)
    stage, stage_label = _profile_stage(topic_intimacy)
    message_count, chat_hours, chat_minutes, recent_7d = await _workspace_message_stats(
        workspace_id, user_id
    )
    chat_minutes = max(chat_minutes, chat_hours * 60)
    backpack_count = await _backpack_count(user_id)

    return ProfileStatsResponse(
        workspace_id=workspace_id,
        intimacy_stage=stage,
        intimacy_stage_label=stage_label,
        topic_intimacy=topic_intimacy,
        intimacy_subtitle=_intimacy_subtitle(topic_intimacy),
        companion_days=_companion_days(getattr(workspace, "createdAt", None)),
        companion_started_on=_date_label(getattr(workspace, "createdAt", None)),
        chat_hours=chat_hours,
        chat_minutes=chat_minutes,
        chat_duration_label=_duration_label(chat_minutes),
        chat_duration_subtitle=_duration_subtitle(chat_minutes),
        message_count=message_count,
        recent_7d_message_count=recent_7d,
        recent_7d_message_label=f"近7天 +{recent_7d}条",
        companion_summary=_companion_summary(agent),
        backpack_count=backpack_count,
        member_is_active=False,
        member_expires_on=None,
    )
