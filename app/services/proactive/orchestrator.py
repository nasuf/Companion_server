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
from app.observability import bind_context
from app.observability.events import (
    EVT_PROACTIVE_DEFERRED,
    EVT_PROACTIVE_FALLBACK,
)
from app.services.proactive.policy import (
    fallback_trigger_type,
    scene_candidate_available,
    select_trigger_type,
    should_hit_window,
)
from app.services.proactive.history import get_proactive_rhythm_adjustment
from app.services.proactive.sender import generate_and_send_proactive
from app.services.proactive.state import (
    advance_to_next_window,
    claim_due_proactive_state,
    claim_waiting_timeout_state,
    escalate_waiting_state,
    has_recent_proactive_or_reminder,
    has_recent_user_activity,
    list_due_proactive_states,
    list_waiting_timeout_states,
    log_proactive_event,
    stop_proactive_state,
)
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
            # 绑 log 上下文 — claimed 含 agent_id / user_id / workspace_id;
            # agent_name 走一次 DB lookup, 整个 process 链路复用
            agent = await db.aiagent.find_unique(where={"id": claimed.agent_id})
            with bind_context(
                agent_id=claimed.agent_id,
                agent_name=agent.name if agent else None,
                user_id=claimed.user_id,
                workspace_id=claimed.workspace_id,
                conversation_id=claimed.conversation_id,
            ):
                await _process_due_state(claimed, now=now)
        except Exception as e:
            logger.warning(f"Proactive state scan failed for workspace={state.workspace_id}: {e}")

    for state in waiting_states:
        try:
            claimed = await claim_waiting_timeout_state(state.id, now=now)
            if not claimed:
                continue
            agent = await db.aiagent.find_unique(where={"id": claimed.agent_id})
            with bind_context(
                agent_id=claimed.agent_id,
                agent_name=agent.name if agent else None,
                user_id=claimed.user_id,
                workspace_id=claimed.workspace_id,
                conversation_id=claimed.conversation_id,
            ):
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
        logger.info(
            "[PROACTIVE] deferred: recent_user_activity",
            extra={"event": EVT_PROACTIVE_DEFERRED, "reason": "recent_user_activity",
                   "stage": state.stage},
        )
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_deferred",
            payload={"reason": "recent_user_activity"},
        )
        return

    # --- Mutex: 30分钟内 AI 已发过 proactive (含 reminder fire). 防止
    # reminder + scheduled_scene 同分钟双发 (生产 bug 复现 2026-05-03 14:00).
    # has_recent_user_activity 只看 user 消息, 漏了"AI 自己刚响完 reminder
    # 还要再 fire 一条 proactive scheduled_scene" 的场景. 两个独立调度器
    # (trigger_scan 15s + proactive_orchestrator 1min) 没互相协调时必撞.
    if await has_recent_proactive_or_reminder(state.workspace_id, now=now, window_minutes=30):
        logger.info(
            "[PROACTIVE] deferred: recent_proactive_activity (reminder/proactive 30min 内已发)",
            extra={"event": EVT_PROACTIVE_DEFERRED, "reason": "recent_proactive_activity",
                   "stage": state.stage},
        )
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_deferred",
            payload={"reason": "recent_proactive_activity"},
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
        logger.info(
            "[PROACTIVE] deferred: topic_fatigue",
            extra={"event": EVT_PROACTIVE_DEFERRED, "reason": "topic_fatigue",
                   "stage": state.stage, "n_recent_msgs": len(recent_texts)},
        )
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_deferred",
            payload={"reason": "topic_fatigue"},
        )
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
        rhythm = await get_proactive_rhythm_adjustment(
            state.agent_id,
            state.user_id,
            workspace_id=state.workspace_id,
            now=_now_dt,
        )
        hit, final_rate = should_hit_window(
            state,
            rate_multiplier=float(rhythm.get("multiplier") or 1.0),
        )
        if not hit:
            await advance_to_next_window(
                state,
                now=now,
                event_type="window_missed",
                payload={
                    "reason": "probability_miss",
                    "final_rate": final_rate,
                    "rhythm": rhythm,
                },
            )
            return

        # spec §1.3: 先 30/30/40 抽签, 抽中 scene 不可用才 50/50 fallback.
        # 预校验 scene 会折损 30/30 配比.
        trigger_type = select_trigger_type()
        if trigger_type == "scheduled_scene":
            schedule = await get_cached_schedule(state.agent_id)
            schedule_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "status": "idle", "type": "leisure"}
            if not scene_candidate_available(state, schedule_status, now=now):
                fallback = fallback_trigger_type()
                logger.info(
                    f"[PROACTIVE] trigger fallback: scheduled_scene → {fallback} "
                    f"(scene unavailable: {schedule_status.get('activity')})",
                    extra={
                        "event": EVT_PROACTIVE_FALLBACK,
                        "from_trigger": "scheduled_scene",
                        "to_trigger": fallback,
                        "schedule_activity": schedule_status.get("activity"),
                    },
                )
                trigger_type = fallback

        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="window_due",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={"stage": state.stage, "final_rate": final_rate, "rhythm": rhythm},
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
