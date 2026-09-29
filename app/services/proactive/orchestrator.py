"""主动交流编排器: 每分钟扫描到期的状态行, 按所处阶段分派。

- window 0 (判定窗, AI 说完 5 分钟): followup.process_followup_window
  → 话题完结判定 + B 模式追问
- window 1-4 (A 模式): gates.check_window_gates → 强制计划直发 / 概率命中
  → 抽触发类型 → sender.generate_and_send_proactive
- waiting_user 超过回复期限: n+1 衰减 (state.escalate_waiting_state)

状态转换与 CAS 语义见 state.py 顶部。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from app.db import db
from app.observability import bind_context
from app.observability.events import (
    EVT_PROACTIVE_DEFERRED,
    EVT_PROACTIVE_FALLBACK,
)
from app.services.proactive.followup import process_followup_window
from app.services.proactive.gates import Gate, check_window_gates, check_workspace
from app.services.proactive.history import get_proactive_rhythm_adjustment
from app.services.proactive.policy import (
    fallback_trigger_type,
    scene_candidate_available,
    select_trigger_type,
    should_hit_window,
)
from app.services.proactive.sender import generate_and_send_proactive
from app.services.proactive.state import (
    JUDGE_WINDOW_INDEX,
    ProactiveStateRecord,
    advance_to_next_window,
    claim_due_proactive_state,
    claim_waiting_timeout_state,
    escalate_waiting_state,
    has_recent_user_activity,
    list_due_proactive_states,
    list_waiting_timeout_states,
    log_proactive_event,
    reclaim_stale_processing_states,
    stop_proactive_state,
)
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status

logger = logging.getLogger(__name__)

# 判定窗让每次 AI 回合结束都会产生一个到期行, 串行扫描时一次慢 LLM 就能拖住
# 整分钟。并发上限低于后台 LLM 配额 (llm_background_max_concurrency), 不挤前台。
_SCAN_CONCURRENCY = 8

_Claim = Callable[..., Awaitable[ProactiveStateRecord | None]]
_Process = Callable[..., Awaitable[None]]


async def scan_proactive_states(now: datetime | None = None) -> None:
    # 先回收僵死在 processing* 的行 (实例在发送途中崩溃), 再列出到期行
    await reclaim_stale_processing_states(now=now)

    states = await list_due_proactive_states(now=now)
    waiting_states = await list_waiting_timeout_states(now=now)
    if not states and not waiting_states:
        return

    sem = asyncio.Semaphore(_SCAN_CONCURRENCY)
    await asyncio.gather(
        *(_claim_and_process(s, claim_due_proactive_state, _process_due_state, sem, now)
          for s in states),
        *(_claim_and_process(s, claim_waiting_timeout_state, _process_waiting_timeout, sem, now)
          for s in waiting_states),
    )


async def _claim_and_process(
    state: ProactiveStateRecord,
    claim: _Claim,
    process: _Process,
    sem: asyncio.Semaphore,
    now: datetime | None,
) -> None:
    async with sem:
        try:
            claimed = await claim(state.id, now=now)
            if not claimed:
                return
            # 绑 log 上下文; agent_name 查一次, 整个处理链路复用
            agent = await db.aiagent.find_unique(where={"id": claimed.agent_id})
            with bind_context(
                agent_id=claimed.agent_id,
                agent_name=agent.name if agent else None,
                user_id=claimed.user_id,
                workspace_id=claimed.workspace_id,
                conversation_id=claimed.conversation_id,
            ):
                await process(claimed, now=now)
        except Exception as e:
            logger.warning(f"Proactive scan failed for workspace={state.workspace_id}: {e}")


async def _apply_gate(state: ProactiveStateRecord, gate: Gate, now: datetime | None) -> None:
    if gate.action == "stop":
        await stop_proactive_state(state, reason=gate.reason, now=now)
        return
    if gate.action == "defer":
        logger.info(
            f"[PROACTIVE] deferred: {gate.reason}",
            extra={"event": EVT_PROACTIVE_DEFERRED, "reason": gate.reason, "stage": state.stage},
        )
    await advance_to_next_window(
        state,
        now=now,
        event_type="window_deferred" if gate.action == "defer" else "window_missed",
        payload={"reason": gate.reason, **gate.payload},
    )


async def _process_due_state(state: ProactiveStateRecord, now: datetime | None = None) -> None:
    if state.current_window_index == JUDGE_WINDOW_INDEX:
        await process_followup_window(state, now=now)
        return

    gate = await check_window_gates(state, now=now)
    if gate is not None:
        await _apply_gate(state, gate, now)
        return

    if state.followup_plan_type in ("seven_day_sparse", "thirty_day_final"):
        await _send_forced(state, now)
    else:
        await _send_by_probability(state, now)


async def _send_forced(state: ProactiveStateRecord, now: datetime | None) -> None:
    """衰减第二/三阶段: 直接触发 (跳过概率)."""
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
    if not await generate_and_send_proactive(state, trigger_type=trigger_type, now=now):
        # 生成失败 → 推进到下一窗口等待重试 (用户中途回来时 CAS 落空, 保持 idle)
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_missed",
            payload={"reason": "forced_send_failed", "trigger_type": trigger_type},
        )


async def _send_by_probability(state: ProactiveStateRecord, now: datetime | None) -> None:
    now_ts = now or datetime.now(timezone.utc)
    rhythm = await get_proactive_rhythm_adjustment(
        state.agent_id,
        state.user_id,
        workspace_id=state.workspace_id,
        now=now_ts,
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
            payload={"reason": "probability_miss", "final_rate": final_rate, "rhythm": rhythm},
        )
        return

    # spec §1.3: 先 30/30/40 抽签, 抽中 scene 不可用才 50/50 fallback.
    # 预校验 scene 会折损 30/30 配比.
    trigger_type = select_trigger_type()
    if trigger_type == "scheduled_scene":
        schedule = await get_cached_schedule(state.agent_id)
        schedule_status = (
            get_current_status(schedule)
            if schedule
            else {"activity": "自由时间", "status": "idle", "type": "leisure"}
        )
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
    if not await generate_and_send_proactive(state, trigger_type=trigger_type, now=now):
        await advance_to_next_window(
            state,
            now=now,
            event_type="window_missed",
            payload={"reason": "send_skipped", "trigger_type": trigger_type},
        )


async def _process_waiting_timeout(state: ProactiveStateRecord, now: datetime | None = None) -> None:
    gate = await check_workspace(state)
    if gate is not None:
        await stop_proactive_state(state, reason=gate.reason, now=now)
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
