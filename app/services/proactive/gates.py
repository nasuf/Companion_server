"""主动交流的发送门槛 —— 「现在能不能主动说话」只在这里判断。

两套门槛, 对应两种主动消息:

- `check_window_gates`: A 模式 (原主动回复, window 1-4)。用户已经离开 ≥30min
  才考虑开新话题, 所以要求 30 分钟内没有用户活动。
- `check_followup_gates`: B 模式 (话题未完结·温柔追问, 判定窗)。用户 5 分钟前
  刚说过话, 「30 分钟内有用户活动」恒为真, 不能套用 A 的门槛; 状态行仍处于
  running 本身就证明了 AI 说完之后用户没再发消息。

全局冷却 (spec「A/B 任意一条主动消息发送成功, 立即开启全局冷却」): 任何经
emit_proactive_message 发出的消息都带 metadata.proactive=true, 30 分钟内存在
这样的消息即处于冷却期。以消息表为准而不是另存一把锁 —— 所有主动出口
(A/B/特殊日期/线下活动/提醒) 天然覆盖, 不会漏记也不会因 Redis 丢失而失效。
用户主动设的提醒不受冷却约束 (triggers._handle_reminder_trigger 有意豁免)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from app.db import db
from app.services.proactive.state import (
    ProactiveStateRecord,
    has_recent_game_activity,
    has_recent_proactive_or_reminder,
    has_recent_user_activity,
)
from app.services.schedule_domain.time_service import _TZ
from app.services.topic import detect_topic_fatigue

logger = logging.getLogger(__name__)

UTC = timezone.utc

# spec §1.2: 主动交流仅在每日 8:00-22:00 之间发送
ACTIVE_HOUR_START = 8
ACTIVE_HOUR_END = 22
COOLDOWN_MINUTES = 30
USER_ACTIVE_MINUTES = 30
# B 追问时对局只要在最近这么久内有动静就算"人在玩", 不插话
FOLLOWUP_GAME_WINDOW_MINUTES = 10
# 提醒豁免冷却: 追问后一分钟又响提醒 = 连收两条, B 让路
FOLLOWUP_REMINDER_LOOKAHEAD_MINUTES = 15

GateAction = Literal["stop", "defer", "miss"]


@dataclass(frozen=True)
class Gate:
    """一次拦截: stop=停止本轮 (workspace 失效), defer/miss=推进到下一窗口."""

    action: GateAction
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)


def is_in_active_hours(now: datetime) -> bool:
    local = now.astimezone(_TZ)
    return ACTIVE_HOUR_START <= local.hour < ACTIVE_HOUR_END


async def check_workspace(state: ProactiveStateRecord) -> Gate | None:
    rows = await db.query_raw(
        "SELECT status FROM chat_workspaces WHERE id = $1 LIMIT 1",
        state.workspace_id,
    )
    if not rows:
        return Gate("stop", "workspace_missing")
    if str(rows[0].get("status") or "archived") != "active":
        return Gate("stop", "workspace_inactive")
    return None


async def offline_activity_owns_channel(state: ProactiveStateRecord) -> bool:
    """到达中的线下活动接管主动通道; 普通聊天照常回复。可选互斥, 失败放行。"""
    try:
        from app.services.offline.activity_companion import can_take_over_proactive_channel

        return await can_take_over_proactive_channel(state.user_id, state.workspace_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PROACTIVE] offline activity mutex failed: %s", exc)
        return False


async def _topic_fatigued(workspace_id: str) -> bool:
    rows = await db.query_raw(
        """
        SELECT m.content
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.workspace_id = $1
          AND c.is_deleted = FALSE
          AND m.role = 'user'
        ORDER BY m.created_at DESC
        LIMIT 10
        """,
        workspace_id,
    )
    texts = [str(r.get("content", "")) for r in (rows or [])]
    texts.reverse()  # chronological
    return detect_topic_fatigue({}, texts)


async def check_window_gates(
    state: ProactiveStateRecord,
    *,
    now: datetime | None = None,
) -> Gate | None:
    """A 模式窗口门槛, 按代价从低到高排列。None = 放行."""
    now_ts = now or datetime.now(UTC)
    gate = await check_workspace(state)
    if gate:
        return gate
    if await offline_activity_owns_channel(state):
        return Gate("defer", "offline_activity_active")
    if await has_recent_user_activity(
        state.workspace_id, now=now, window_minutes=USER_ACTIVE_MINUTES,
    ):
        return Gate("defer", "recent_user_activity")
    # 全局冷却。两个独立调度器 (trigger_scan 15s + 本扫描 1min) 不互相协调时,
    # 没有这道门会出现 reminder + scheduled_scene 同分钟双发 (生产 2026-05-03 14:00)。
    if await has_recent_proactive_or_reminder(
        state.workspace_id, now=now, window_minutes=COOLDOWN_MINUTES,
    ):
        return Gate("defer", "recent_proactive_activity")
    if await _topic_fatigued(state.workspace_id):
        return Gate("defer", "topic_fatigue")
    if not is_in_active_hours(now_ts):
        return Gate("miss", "off_hours", {"local_hour": now_ts.astimezone(_TZ).hour})
    return None


async def _patience_is_normal(state: ProactiveStateRecord) -> bool:
    """AI 还在生气 (耐心 <70) 时不追问 —— 那时候追一句只会显得别扭。读失败放行."""
    try:
        from app.services.interaction.boundary import PATIENCE_NORMAL_MIN, get_patience

        return await get_patience(state.agent_id, state.user_id) >= PATIENCE_NORMAL_MIN
    except Exception as exc:  # noqa: BLE001
        logger.debug("[FOLLOWUP] patience read failed, allow: %s", exc)
        return True


async def _reminder_due_soon(state: ProactiveStateRecord, now: datetime) -> bool:
    try:
        row = await db.timetrigger.find_first(where={
            "aiAgentId": state.agent_id,
            "userId": state.user_id,
            "actionType": "reminder",
            "isActive": True,
            "triggerTime": {
                "gte": now,
                "lte": now + timedelta(minutes=FOLLOWUP_REMINDER_LOOKAHEAD_MINUTES),
            },
        })
    except Exception as exc:  # noqa: BLE001
        logger.debug("[FOLLOWUP] reminder lookahead failed, allow: %s", exc)
        return False
    return row is not None


async def check_followup_gates(
    state: ProactiveStateRecord,
    *,
    now: datetime | None = None,
) -> str | None:
    """B 模式追问门槛; 返回拦截原因, None = 放行。

    B 刻意**不**计入每日 3 条上限和疲劳分: 它是一次仍在进行中的对话的自然
    延续, 不是冷启动搭话; 单会话仅一次已经把频率压住了。
    """
    now_ts = now or datetime.now(UTC)
    if not is_in_active_hours(now_ts):
        return "off_hours"
    if await offline_activity_owns_channel(state):
        return "offline_activity_active"
    if await has_recent_game_activity(
        state.workspace_id, now=now, window_minutes=FOLLOWUP_GAME_WINDOW_MINUTES,
    ):
        return "game_in_progress"
    if await has_recent_proactive_or_reminder(
        state.workspace_id, now=now, window_minutes=COOLDOWN_MINUTES,
    ):
        return "cooldown"
    if not await _patience_is_normal(state):
        return "patience_low"
    if await _reminder_due_soon(state, now_ts):
        return "reminder_due_soon"
    return None
