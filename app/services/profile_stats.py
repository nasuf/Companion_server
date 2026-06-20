from __future__ import annotations

from datetime import UTC, datetime
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


def _profile_stage(topic_intimacy: float) -> tuple[str, str]:
    for lo, hi, code, label in _STAGE_BANDS:
        if lo <= topic_intimacy < hi:
            return code, label
    return "P5", "深度羁绊"


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


async def _workspace_message_stats(workspace_id: str, user_id: str) -> tuple[int, int]:
    rows = await db.query_raw(
        """
        SELECT
          COALESCE(COUNT(m.id), 0)::int AS message_count,
          COALESCE(COUNT(DISTINCT date_trunc('hour', m.created_at)), 0)::int AS active_chat_hours
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.workspace_id = $1
          AND c.user_id = $2
          AND c.is_deleted = FALSE
        """,
        workspace_id,
        user_id,
    )
    row = rows[0] if rows else {}
    return _as_int(_field(row, "message_count")), _as_int(
        _field(row, "active_chat_hours")
    )


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
    message_count, chat_hours = await _workspace_message_stats(workspace_id, user_id)

    return ProfileStatsResponse(
        workspace_id=workspace_id,
        intimacy_stage=stage,
        intimacy_stage_label=stage_label,
        topic_intimacy=topic_intimacy,
        companion_days=_companion_days(getattr(workspace, "createdAt", None)),
        chat_hours=chat_hours,
        message_count=message_count,
        companion_summary=_companion_summary(agent),
    )
