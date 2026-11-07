"""推荐过的线下活动, 直接从推荐记录读 (VOICE01.10)。

用户问"你推荐过我什么地方/我们去过哪"时, 记忆里没有这些推荐: AI 写的活动文案按设计不进
抽取 (chat_sources.excluded_from_chat_extraction, 防止被抽成第一人称人设事实), "接受了某个
推荐"这种可撤销的打算也不进长期记忆 (40cc0d69)。模型查不到只能编。

这里把推荐记录本身当回答这类问题的来源: 标题、地点、推荐日期和当前状态都以表里为准, 能完整
列出全部推荐 (记忆检索只取最相关的几条)。用户回应过的活动另有一对随状态改写的记忆
(offline/activity_memory.py), 用于聊别的话题时自然想起来; 从没回应过的推荐只在这里。
文字聊天只在用户这句话像在问推荐/去过哪时注入 (见 asks_about_recommendations),
语音通话在开场时注入一次 (数据很小, 不必每轮检索)。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.db import db

# 注入的条数上限: 生产上单个用户最多 21 条推荐, 其中大多是没回应过的
LIMIT = 10
_TZ = ZoneInfo(settings.schedule_timezone)

# 像在问"你推荐过/我们去过"的说法。宽一点没关系: 段落规定对方没问起时不主动列举,
# 多注入一次只多几十个 token; 漏了就会编造地点。
_RECALL_CUE = re.compile(
    r"推荐过|介绍过|你推荐的|你给我推荐|给我推荐的|去过哪|去过什么|去过的地方|好玩的地方|好去处"
    r"|那个活动|上次.{0,4}活动|活动推荐|待出行|一起去过|带我去过"
)

_STATUS = {
    "pending": "推荐了，对方还没回应",
    "ignored": "对方暂时不考虑",
    "cancelled": "已取消，对方没去",
    "expired": "已过期，对方没去",
    "completed": "已完成",
}


def asks_about_recommendations(message: str) -> bool:
    return bool(_RECALL_CUE.search(message or ""))


async def load_recent(user_id: str, workspace_id: str | None, *, limit: int = LIMIT) -> list[dict[str, Any]]:
    """该用户在这个工作区最近的推荐 (新的在前), 含已取消/过期的: 状态本身就是要说清的事实。"""
    rows = await db.query_raw(
        """
        SELECT title, location_name, city, status, reached, created_at
        FROM offline_activity_recommendations
        WHERE user_id = $1 AND ($2::text IS NULL OR workspace_id = $2)
        ORDER BY created_at DESC
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    return [dict(r) for r in rows or []]


def _status(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    if status == "accepted":
        return "对方到过现场" if row.get("reached") else "对方放进了待出行，还没去"
    label = _STATUS.get(status, status or "状态未知")
    if status == "completed" and row.get("reached"):
        return "对方去过，已完成"
    return label


def _day(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ""
    if not isinstance(value, datetime):
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(_TZ).strftime("%Y-%m-%d")


def render_lines(rows: list[dict[str, Any]]) -> list[str]:
    """每条一行: 日期、标题、地点、当前状态。不含任务/拍摄物品等不能披露的字段。"""
    lines = []
    for row in rows:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        place = str(row.get("location_name") or "").strip()
        city = str(row.get("city") or "").strip()
        if city and city not in place:
            place = f"{city}·{place}" if place else city
        day = _day(row.get("created_at"))
        lines.append(f"{day + ' ' if day else ''}推荐「{title}」{f'（{place}）' if place else ''}：{_status(row)}")
    return lines
