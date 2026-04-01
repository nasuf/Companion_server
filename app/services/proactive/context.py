"""主动聊天上下文构建。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.services.relationship.emotion import get_ai_emotion
from app.services.relationship.intimacy import get_relationship_stage, get_topic_intimacy
from app.services.memory import memory_repo
from app.services.memory.core_memory import load_core_memory_strings
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status

UTC = timezone.utc

STAGE_CATEGORY_MAP = {
    "cold_start": ["身份", "生活", "偏好"],
    "warming": ["生活", "偏好", "情绪"],
    "intimate": ["情绪", "思维", "生活", "偏好"],
}


async def build_proactive_context(
    *,
    workspace_id: str,
    user_id: str,
    agent_id: str,
    trigger_type: str,
    stage: str,
    exclude_memory_ids: set[str] | None = None,
) -> dict[str, Any]:
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise ValueError(f"Agent not found: {agent_id}")

    schedule = await get_cached_schedule(agent_id)
    schedule_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "status": "idle", "type": "leisure"}
    emotion = await get_ai_emotion(agent_id)
    core_memories = await load_core_memory_strings(user_id=user_id, workspace_id=workspace_id, source="user")
    proactive_memories, used_memory_ids = await _load_proactive_memories(
        user_id=user_id,
        workspace_id=workspace_id,
        stage=stage,
        trigger_type=trigger_type,
        exclude_memory_ids=exclude_memory_ids,
    )
    topic_intimacy = await get_topic_intimacy(agent_id, user_id)
    relationship_stage = get_relationship_stage(topic_intimacy)

    silence_hours = await _compute_silence_hours(workspace_id)
    scene_hint = _build_scene_hint(trigger_type, schedule_status)

    return {
        "agent": agent,
        "schedule_status": schedule_status,
        "emotion": emotion,
        "core_memories": core_memories[:8],
        "proactive_memories": proactive_memories[:6],
        "used_memory_ids": used_memory_ids,
        "relationship_stage": relationship_stage,
        "topic_intimacy": topic_intimacy,
        "silence_hours": silence_hours,
        "scene_hint": scene_hint,
        "trigger_type": trigger_type,
        "stage": stage,
    }


async def _load_proactive_memories(
    *,
    user_id: str,
    workspace_id: str,
    stage: str,
    trigger_type: str,
    exclude_memory_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Load proactive memories with dedup support.

    Returns (texts, memory_ids) for tracking which memories were used.
    """
    rows = await memory_repo.find_many(
        source="user",
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "isArchived": False,
        },
        order={"importance": "desc"},
        take=30,
    )

    exclude = exclude_memory_ids or set()
    allowed_categories = set(STAGE_CATEGORY_MAP.get(stage, ["生活", "偏好"]))
    if trigger_type == "silence_wakeup":
        allowed_categories = {"生活", "偏好", "身份"}
    elif trigger_type == "scheduled_scene":
        allowed_categories = {"生活", "偏好"}

    texts: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.id in exclude:
            continue
        if row.mainCategory and row.mainCategory not in allowed_categories:
            continue
        text = row.summary or row.content
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(f"[{row.mainCategory or '生活'}/{row.subCategory or '其他'}] {text}")
        ids.append(row.id)
    return texts[:6], ids[:6]


def _build_scene_hint(trigger_type: str, schedule_status: dict[str, Any]) -> str:
    activity = str(schedule_status.get("activity") or "自由时间")
    status = str(schedule_status.get("status") or "idle")
    if trigger_type == "scheduled_scene":
        return f"你当前处于{activity}（状态：{status}），适合从此刻生活情景自然发起聊天。"
    if trigger_type == "memory_proactive":
        return "优先从用户过往记忆里选一个具体点切入，不要泛泛问候。"
    return "优先用轻量、低打扰的方式重新建立联系。"


async def _compute_silence_hours(workspace_id: str) -> float:
    rows = await db.query_raw(
        """
        SELECT m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.workspace_id = $1
          AND c.is_deleted = FALSE
        ORDER BY m.created_at DESC
        LIMIT 1
        """,
        workspace_id,
    )
    if not rows:
        return 999.0
    last_created = rows[0].get("created_at")
    if not isinstance(last_created, datetime):
        try:
            last_created = datetime.fromisoformat(str(last_created))
        except ValueError:
            return 999.0
    if last_created.tzinfo is None:
        last_created = last_created.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - last_created.astimezone(UTC)).total_seconds() / 3600.0)
