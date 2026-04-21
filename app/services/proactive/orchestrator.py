"""主动聊天编排器。

第一阶段只做状态推进，不直接生成主动消息：
- 扫描到期 proactive state
- 执行基础互斥检查
- 记录命中窗口
- 推进到下一时间区间

后续阶段将在这里加入：
- 区间概率命中
- 类型选择（沉默唤醒 / 记忆主动 / 定时情景）
- 统一发送链路
- n 衰减状态机
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.db import db
from app.services.proactive.policy import (
    fallback_trigger_type,
    scene_candidate_available,
    select_trigger_type,
    should_hit_window,
)
from app.services.proactive.sender import generate_and_send_proactive
from app.services.proactive.state import (
    advance_to_next_window,
    claim_due_proactive_state,
    claim_waiting_timeout_state,
    escalate_waiting_state,
    has_recent_user_activity,
    list_due_proactive_states,
    list_waiting_timeout_states,
    log_proactive_event,
    stop_proactive_state,
)
from app.services.relationship.intimacy import get_topic_intimacy
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status
from app.services.schedule_domain.time_service import _TZ
from app.services.topic import detect_topic_fatigue

logger = logging.getLogger(__name__)


# spec §1.2: 主动交流仅在每日 8:00-22:00 之间发送.
# 22:00-次日 8:00 的窗口照常推进+概率计数, 但命中后也不发送.
PROACTIVE_ACTIVE_HOUR_START = 8
PROACTIVE_ACTIVE_HOUR_END = 22


def _is_in_active_hours(now: datetime) -> bool:
    local = now.astimezone(_TZ)
    return PROACTIVE_ACTIVE_HOUR_START <= local.hour < PROACTIVE_ACTIVE_HOUR_END


async def scan_proactive_states(now: datetime | None = None) -> None:
    states = await list_due_proactive_states(now=now)
    waiting_states = await list_waiting_timeout_states(now=now)
    if not states and not waiting_states:
        return

    for state in states:
        try:
            claimed = await claim_due_proactive_state(state.id, now=now)
            if not claimed:
                continue
            await _process_due_state(claimed, now=now)
        except Exception as e:
            logger.warning(f"Proactive state scan failed for workspace={state.workspace_id}: {e}")

    for state in waiting_states:
        try:
            claimed = await claim_waiting_timeout_state(state.id, now=now)
            if not claimed:
                continue
            await _process_waiting_timeout(claimed, now=now)
        except Exception as e:
            logger.warning(f"Proactive waiting timeout failed for workspace={state.workspace_id}: {e}")


async def _process_due_state(state, now: datetime | None = None) -> None:
    # --- Mutex: workspace active (永久性条件，停止合理) ---
    workspace_rows = await db.query_raw(
        """
        SELECT status
        FROM chat_workspaces
        WHERE id = $1
        LIMIT 1
        """,
        state.workspace_id,
    )
    if not workspace_rows:
        await stop_proactive_state(state, reason="workspace_missing", now=now)
        return

    workspace_status = str(workspace_rows[0].get("status") or "archived")
    if workspace_status != "active":
        await stop_proactive_state(state, reason="workspace_inactive", now=now)
        return

    # --- Mutex: 30分钟内有用户活动 (临时性条件，推迟到下一窗口) ---
    if await has_recent_user_activity(state.workspace_id, now=now, window_minutes=30):
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_deferred",
            payload={"reason": "recent_user_activity"},
        )
        return

    # --- Mutex: 话题疲劳 (临时性条件，推迟到下一窗口) ---
    recent_user_msgs = await db.query_raw(
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
        state.workspace_id,
    )
    recent_texts = [str(r.get("content", "")) for r in (recent_user_msgs or [])]
    recent_texts.reverse()  # chronological order
    if detect_topic_fatigue({}, recent_texts):
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_deferred",
            payload={"reason": "topic_fatigue"},
        )
        return

    # --- Gate: 亲密度前置检查 (永久性条件，停止合理) ---
    topic_intimacy = await get_topic_intimacy(state.agent_id, state.user_id)
    if state.stage == "cold_start" and topic_intimacy < 10:
        await stop_proactive_state(state, reason="intimacy_too_low", now=now)
        return

    # --- Gate: spec §1.2 22:00-8:00 不发送, 照常推进窗口 ---
    from datetime import datetime as _dt, timezone as _tz
    _now_dt = now or _dt.now(_tz.utc)
    if not _is_in_active_hours(_now_dt):
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_missed",
            payload={"reason": "off_hours", "local_hour": _now_dt.astimezone(_TZ).hour},
        )
        return

    # --- 强制计划 vs 正常概率分支 (BUG 1) ---
    is_forced_plan = state.followup_plan_type in ("seven_day_sparse", "thirty_day_final")

    if is_forced_plan:
        # 流程图: "直接触发(跳过概率)"
        trigger_type = fallback_trigger_type()

        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="forced_trigger",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={
                "stage": state.stage,
                "followup_plan_type": state.followup_plan_type,
                "remaining": state.remaining_forced_triggers,
            },
        )

        sent = await generate_and_send_proactive(state, trigger_type=trigger_type, now=now)
        if not sent:
            # 强制计划生成失败，推进到下一窗口等待重试
            await advance_to_next_window(
                state,
                now=now,
                event_type="window_missed",
                payload={"reason": "forced_send_failed", "trigger_type": trigger_type},
            )
    else:
        # 正常概率流程
        hit, final_rate = should_hit_window(state)
        if not hit:
            await advance_to_next_window(
                state,
                now=now,
                event_type="window_missed",
                payload={"reason": "probability_miss", "final_rate": final_rate},
            )
            return

        schedule = await get_cached_schedule(state.agent_id)
        schedule_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "status": "idle", "type": "leisure"}
        scene_available = scene_candidate_available(state, schedule_status, now=now)
        trigger_type = select_trigger_type(state, scene_available=scene_available)
        if trigger_type == "scheduled_scene" and not scene_available:
            trigger_type = fallback_trigger_type()

        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="window_due",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={"stage": state.stage, "final_rate": final_rate},
        )

        sent = await generate_and_send_proactive(state, trigger_type=trigger_type, now=now)
        if not sent:
            await advance_to_next_window(
                state,
                now=now,
                event_type="window_missed",
                payload={"reason": "send_skipped", "trigger_type": trigger_type},
            )


async def _process_waiting_timeout(state, now: datetime | None = None) -> None:
    workspace_rows = await db.query_raw(
        """
        SELECT status
        FROM chat_workspaces
        WHERE id = $1
        LIMIT 1
        """,
        state.workspace_id,
    )
    if not workspace_rows:
        await stop_proactive_state(state, reason="workspace_missing", now=now)
        return

    workspace_status = str(workspace_rows[0].get("status") or "archived")
    if workspace_status != "active":
        await stop_proactive_state(state, reason="workspace_inactive", now=now)
        return

    if await has_recent_user_activity(state.workspace_id, now=now, window_minutes=30):
        await stop_proactive_state(state, reason="recent_user_activity", now=now)
        return

    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=state.conversation_id,
        event_type="reply_timeout",
        trigger_type=state.followup_plan_type,
        payload={
            "silence_level_n": state.silence_level_n,
            "followup_plan_type": state.followup_plan_type,
        },
    )
    await escalate_waiting_state(state, now=now)
