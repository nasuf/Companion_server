"""Achievement evaluators driven by user memory changelog rows."""

from __future__ import annotations

from app.db import db
from app.services.achievements.repository import _memory_count, unlock_achievement
from app.services.achievements.utils import _field


async def process_memory_changelog(user_id: str, memory_id: str, operation: str, workspace_id: str | None = None) -> None:
    if operation == "access" or not memory_id:
        return
    rows = await db.query_raw(
        """
        SELECT 'user' AS source, user_id, workspace_id, main_category, sub_category, content
        FROM memories_user WHERE id = $1
        UNION ALL
        SELECT 'ai' AS source, user_id, workspace_id, main_category, sub_category, content
        FROM memories_ai WHERE id = $1
        LIMIT 1
        """,
        memory_id,
    )
    if not rows:
        return
    row = rows[0]
    if _field(row, "source") != "user":
        return
    memory_user_id = str(_field(row, "user_id") or "")
    if not memory_user_id or memory_user_id != user_id:
        return
    user_id = memory_user_id
    workspace_id = workspace_id or _field(row, "workspace_id")
    ws_rows = await db.query_raw(
        "SELECT agent_id FROM chat_workspaces WHERE id = $1 LIMIT 1",
        workspace_id,
    ) if workspace_id else []
    if not ws_rows:
        return
    agent_id = str(_field(ws_rows[0], "agent_id"))
    main = str(_field(row, "main_category") or "")
    sub = str(_field(row, "sub_category") or "")
    mapping = {
        ("身份", "姓名"): 29,
        ("身份", "年龄"): 10,
        ("身份", "性别"): 11,
        ("身份", "现居地"): 12,
        ("身份", "出生地"): 12,
        ("身份", "成长地"): 12,
        ("身份", "职业/与经济"): 13,
        ("情绪", "高兴"): 16,
        ("情绪", "悲伤"): 23,
        ("思维", "理想与目标"): 49,
    }
    achievement_id = mapping.get((main, sub))
    if achievement_id:
        await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, achievement_id=achievement_id)
    if main == "偏好":
        count = await _memory_count(user_id, workspace_id, main)
        if count >= 20:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, achievement_id=65)
    if main == "情绪" and sub == "恐惧":
        count = await _memory_count(user_id, workspace_id, main, sub)
        if count >= 10:
            await unlock_achievement(user_id=user_id, agent_id=agent_id, workspace_id=workspace_id, achievement_id=66)
