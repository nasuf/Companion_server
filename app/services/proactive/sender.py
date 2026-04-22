"""主动消息生成与发送入口.

按职责拆分:
  _check_send_eligibility   日限/二日限/workspace/conversation 检查
  _resolve_conversation_id  从 state 推出最终 conversation_id
  _apply_memory_cooldown    spec §9 -1/+50 冷却语义
  _generate_message         按 (trigger_type, source, decay_final) 分发 7 个 prompt
  _persist_proactive_state  调 mark_proactive_sent + save_last_reply_timestamp
  generate_and_send_proactive  主流程编排 (上述 5 段 + emit + bg AI 自我记忆 pipeline)

公共持久化与 WS 广播在 emit.py.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db import db
from app.services.llm.models import get_chat_model, invoke_text
from app.services.memory.recording.pipeline import process_memory_pipeline
from app.services.proactive.emit import emit_proactive_message
from app.services.proactive.history import (
    can_send_proactive, can_send_proactive_2day,
    increment_proactive_count, increment_proactive_2day_count,
)
from app.services.proactive.context import build_proactive_context
from app.services.proactive.policy import select_topic_source, select_topic_theme
from app.services.proactive.state import (
    ProactiveStateRecord,
    ensure_proactive_state_for_workspace,
    get_active_workspace_context,
    log_proactive_event,
    mark_proactive_sent,
)
from app.services.prompting.store import get_prompt_text
from app.services.interaction.reply_context import save_last_reply_timestamp

logger = logging.getLogger(__name__)

UTC = timezone.utc
SENDABLE_PROACTIVE_STATUSES = {"idle"}

_MEMORY_SOURCES = frozenset({"ai_l1", "ai_l2", "user_l1", "user_l2"})


# ────────────────────────────────────────────────────────────────────
# Eligibility checks
# ────────────────────────────────────────────────────────────────────

async def _log_skip(
    state: ProactiveStateRecord,
    trigger_type: str,
    reason: str,
    *,
    conversation_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"reason": reason}
    if extra:
        payload.update(extra)
    await log_proactive_event(
        state_id=state.id,
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        conversation_id=conversation_id or state.conversation_id,
        event_type="send_skipped",
        window_index=state.current_window_index,
        trigger_type=trigger_type,
        payload=payload,
    )


@dataclass
class _SendPrep:
    conversation_id: str
    cooldown: dict[str, int]
    exclude_memory_ids: set[str]


async def _check_send_eligibility(
    state: ProactiveStateRecord,
    trigger_type: str,
) -> _SendPrep | None:
    """spec §9 互斥: 检查日限/二日限/workspace/conversation. 失败返回 None."""
    if not await can_send_proactive(state.agent_id, state.user_id):
        await _log_skip(state, trigger_type, "daily_limit")
        return None
    if not await can_send_proactive_2day(state.agent_id, state.user_id):
        await _log_skip(state, trigger_type, "2day_limit")
        return None

    workspace_context = await get_active_workspace_context(state.workspace_id)
    if not workspace_context:
        await _log_skip(state, trigger_type, "workspace_missing")
        return None

    conversation_id = str(
        workspace_context.get("conversation_id") or state.conversation_id or ""
    )
    if not conversation_id:
        await _log_skip(state, trigger_type, "conversation_missing")
        return None

    cooldown, exclude = _apply_memory_cooldown(state, trigger_type)
    return _SendPrep(
        conversation_id=conversation_id,
        cooldown=cooldown,
        exclude_memory_ids=exclude,
    )


# ────────────────────────────────────────────────────────────────────
# spec §9 记忆冷却 (-1 / +50)
# ────────────────────────────────────────────────────────────────────

def _apply_memory_cooldown(
    state: ProactiveStateRecord,
    trigger_type: str,
) -> tuple[dict[str, int], set[str]]:
    """spec §9 记忆去重规则.

    - metadata["memory_cooldown"] = {memory_id: int}
    - 只在 memory_proactive 候选检索时 -1
    - 抽中后置 50 (在 _persist_proactive_state 处理)
    - 兼容旧 used_memory_ids 列表 → 一次性迁移为冷却 50
    """
    metadata = state.metadata or {}
    cooldown: dict[str, int] = dict(metadata.get("memory_cooldown") or {})
    if not cooldown and metadata.get("used_memory_ids"):
        cooldown = {mid: 50 for mid in (metadata.get("used_memory_ids") or [])}
    if trigger_type == "memory_proactive":
        cooldown = {mid: cd - 1 for mid, cd in cooldown.items() if cd - 1 > 0}
    exclude = {mid for mid, cd in cooldown.items() if cd > 0}
    return cooldown, exclude


# ────────────────────────────────────────────────────────────────────
# Personality brief & prompt dispatch
# ────────────────────────────────────────────────────────────────────

def _build_personality_brief(agent) -> str:
    """从 agent 7 维性格导出简短描述, 给 prompt 用."""
    try:
        p = getattr(agent, "personality", None) or {}
        if not isinstance(p, dict) or not p:
            return "温和友善"
        parts: list[str] = []
        if p.get("liveliness", 50) >= 70:
            parts.append("活泼")
        elif p.get("liveliness", 50) <= 30:
            parts.append("安静")
        if p.get("humor", 50) >= 70:
            parts.append("幽默")
        if p.get("rationality", 50) >= 70:
            parts.append("理性")
        if p.get("sensitivity", 50) >= 70:
            parts.append("感性")
        if p.get("planning", 50) >= 70:
            parts.append("计划性强")
        if p.get("spontaneity", 50) >= 70:
            parts.append("随性")
        if p.get("imagination", 50) >= 70:
            parts.append("脑洞大")
        return "、".join(parts) if parts else "温和友善"
    except Exception:
        return "温和友善"


# (trigger_type, source) → prompt key
_PROMPT_KEY_BY_SOURCE: dict[tuple[str, str], str] = {
    ("silence_wakeup", "ai_l1"): "proactive.silence_ai_memory",
    ("silence_wakeup", "ai_l2"): "proactive.silence_ai_memory",
    ("silence_wakeup", "user_l1"): "proactive.silence_user_memory",
    ("silence_wakeup", "user_l2"): "proactive.silence_user_memory",
    ("silence_wakeup", "ai_schedule"): "proactive.silence_schedule",
    ("silence_wakeup", "greeting"): "proactive.silence_plain",
    ("memory_proactive", "ai_l1"): "proactive.memory_ai",
    ("memory_proactive", "ai_l2"): "proactive.memory_ai",
    ("memory_proactive", "user_l1"): "proactive.memory_user",
    ("memory_proactive", "user_l2"): "proactive.memory_user",
    ("scheduled_scene", "ai_schedule"): "proactive.scheduled_scene",
}


def _format_prompt(key: str, ctx: dict, personality_brief: str) -> str | None:
    """按 prompt key 选定填充字段."""
    topic = ctx.get("topic_theme") or "日常"
    memories = ctx.get("proactive_memories") or []
    schedule_status = ctx.get("schedule_status") or {}
    activity = str(schedule_status.get("activity") or "自由时间")
    status = str(schedule_status.get("status") or "idle")
    memory_text = "\n".join(f"- {m}" for m in memories) if memories else "（暂无）"

    # Spec §4.1 step 4：沉默唤醒参考信息含 话题主题/用户画像/近期对话
    user_portrait = ctx.get("user_portrait") or "(未知)"
    recent_context = ctx.get("recent_context") or "(无)"
    silence_shared = {
        "topic": topic,
        "user_portrait": user_portrait,
        "recent_context": recent_context,
    }
    fields_by_key: dict[str, dict[str, Any]] = {
        "proactive.silence_plain": {
            "personality_brief": personality_brief,
            **silence_shared,
        },
        "proactive.silence_ai_memory": {
            "personality_brief": personality_brief,
            "ai_memory": memory_text,
            **silence_shared,
        },
        "proactive.silence_user_memory": {
            "personality_brief": personality_brief,
            "user_memory": memory_text,
            **silence_shared,
        },
        "proactive.silence_schedule": {
            "personality_brief": personality_brief,
            "current_activity": f"{activity}({status})",
            **silence_shared,
        },
        # Spec §4.2 + 指令模版 P24-25：仅 3 项（性格/记忆/话题主题）
        "proactive.memory_ai": {
            "personality_brief": personality_brief,
            "ai_memory": memory_text,
            "topic": topic,
        },
        "proactive.memory_user": {
            "personality_brief": personality_brief,
            "user_memory": memory_text,
            "topic": topic,
        },
        "proactive.scheduled_scene": {
            "personality_brief": personality_brief,
            "time": datetime.now(UTC).astimezone().strftime("%H:%M"),
            "activity": activity,
            "status": status,
        },
    }
    fields = fields_by_key.get(key)
    if fields is None:
        return None
    try:
        # tpl is fetched async by caller (this is a sync helper for clarity)
        tpl = ctx["__tpl"]
        return tpl.format(**fields)
    except (KeyError, ValueError) as e:
        logger.warning(f"Prompt format failed key={key}: {e}")
        return None


async def _generate_message(ctx: dict) -> str | None:
    """spec §4 按 (trigger_type, source) 分发到 7 个专属 prompt;
    spec §8.5 衰减最后一次优先 decay_final.
    """
    agent = ctx["agent"]
    trigger_type = ctx["trigger_type"]
    source = ctx.get("source") or "greeting"
    personality_brief = _build_personality_brief(agent)

    if ctx.get("is_decay_final"):
        tpl = await get_prompt_text("proactive.decay_final")
        prompt = tpl.format(personality_brief=personality_brief)
    else:
        key = _PROMPT_KEY_BY_SOURCE.get(
            (trigger_type, source), "proactive.silence_plain"
        )
        tpl = await get_prompt_text(key)
        ctx["__tpl"] = tpl
        prompt = _format_prompt(key, ctx, personality_brief)
        if not prompt:
            return None

    response = (await invoke_text(get_chat_model(), prompt)).strip()
    if response == "SKIP" or len(response) < 4:
        return None
    return response


# ────────────────────────────────────────────────────────────────────
# Persistence wrapping (state + cooldown commit)
# ────────────────────────────────────────────────────────────────────

async def _persist_proactive_state(
    state: ProactiveStateRecord,
    *,
    trigger_type: str,
    message: str,
    assistant_message_id: str,
    cooldown: dict[str, int],
    new_used_ids: set[str],
    now_ts: datetime,
) -> None:
    """spec §9 抽中的 mid 置 50 + mark_proactive_sent + last_reply_timestamp."""
    for mid in new_used_ids:
        cooldown[mid] = 50
    await mark_proactive_sent(
        state,
        trigger_type=trigger_type,
        message=message,
        assistant_message_id=assistant_message_id,
        now=now_ts,
        mark_daily_scene=(trigger_type == "scheduled_scene"),
        extra_metadata={
            "memory_cooldown": cooldown,
            "used_memory_ids": list(cooldown.keys()),
        },
    )
    await save_last_reply_timestamp(state.agent_id, state.user_id, when=now_ts)


# ────────────────────────────────────────────────────────────────────
# Main entry: generate_and_send_proactive
# ────────────────────────────────────────────────────────────────────

async def generate_and_send_proactive(
    state: ProactiveStateRecord,
    *,
    trigger_type: str,
    now: datetime | None = None,
) -> bool:
    now_ts = now or datetime.now(UTC)

    prep = await _check_send_eligibility(state, trigger_type)
    if prep is None:
        return False

    # spec §3.2 话题方向 + §4.1/§4.2 来源概率表
    topic_theme = select_topic_theme(state.stage)
    source = select_topic_source(state.stage, trigger_type)

    ctx = await build_proactive_context(
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        trigger_type=trigger_type,
        stage=state.stage,
        exclude_memory_ids=prep.exclude_memory_ids,
        source=source,
        topic_theme=topic_theme,
    )

    # spec §4.1 沉默唤醒兜底; §4.2 记忆主动失败时取消
    if source in _MEMORY_SOURCES and not ctx.get("proactive_memories"):
        if trigger_type == "silence_wakeup":
            source = "greeting"
            ctx["source"] = "greeting"
            ctx["scene_hint"] = "优先用轻量、低打扰的方式重新建立联系。"
        else:
            await _log_skip(
                state, trigger_type, "memory_source_empty",
                conversation_id=prep.conversation_id,
                extra={"source": source},
            )
            return False

    # spec §8.5 衰减最后一次
    ctx["is_decay_final"] = state.followup_plan_type == "thirty_day_final"

    message = await _generate_message(ctx)
    if not message:
        await _log_skip(
            state, trigger_type, "empty_or_skip",
            conversation_id=prep.conversation_id,
        )
        return False

    assistant_message_id = await emit_proactive_message(
        conversation_id=prep.conversation_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        workspace_id=state.workspace_id,
        message=message,
        trigger_type=trigger_type,
        extra_metadata={"stage": state.stage},
    )

    await increment_proactive_count(state.agent_id, state.user_id)
    await increment_proactive_2day_count(state.agent_id, state.user_id)

    await _persist_proactive_state(
        state,
        trigger_type=trigger_type,
        message=message,
        assistant_message_id=assistant_message_id,
        cooldown=prep.cooldown,
        new_used_ids=set(ctx.get("used_memory_ids", [])),
        now_ts=now_ts,
    )

    asyncio.create_task(_bg_proactive_ai_memory(state.user_id, message))
    return True


# ────────────────────────────────────────────────────────────────────
# Manual / triggered entry
# ────────────────────────────────────────────────────────────────────

async def send_manual_or_triggered_proactive(
    *,
    workspace_id: str,
    trigger_type: str,
    now: datetime | None = None,
) -> dict[str, str | bool | None]:
    state = await ensure_proactive_state_for_workspace(
        workspace_id, now=now, reason="manual_or_triggered",
    )
    if not state:
        return {"ok": False, "reason": "workspace_or_state_missing", "message": None}
    if state.status not in SENDABLE_PROACTIVE_STATUSES:
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="send_skipped",
            trigger_type=trigger_type,
            payload={"reason": "state_not_sendable", "status": state.status},
        )
        return {"ok": False, "reason": f"state_not_sendable:{state.status}", "message": None}

    sent = await generate_and_send_proactive(state, trigger_type=trigger_type, now=now)
    if not sent:
        return {"ok": False, "reason": "generation_or_limit_blocked", "message": None}

    rows = await db.query_raw(
        """
        SELECT message
        FROM proactive_chat_logs
        WHERE workspace_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        state.workspace_id,
    )
    latest_message = str(rows[0]["message"]) if rows else None
    return {"ok": True, "reason": None, "message": latest_message}


# ────────────────────────────────────────────────────────────────────
# spec §12 开场主动第一句话
# ────────────────────────────────────────────────────────────────────

async def send_first_greeting(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    workspace_id: str | None = None,
) -> bool:
    """spec §12: 用户首次进入聊天 (对话消息数=0) 时 AI 主动发送第一句.

    不走时间窗概率/不计入每日 3 次上限; 计入衰减 n=1 由 state 常规路径处理.
    """
    count = await db.message.count(where={"conversationId": conversation_id})
    if count > 0:
        return False

    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return False

    try:
        tpl = await get_prompt_text("proactive.first_greeting")
        prompt = tpl.format(
            ai_name=agent.name,
            personality_brief=_build_personality_brief(agent),
            occupation=getattr(agent, "occupation", None) or "普通人",
        )
        message = (await invoke_text(get_chat_model(), prompt)).strip()
        if not message or len(message) < 4:
            return False

        await emit_proactive_message(
            conversation_id=conversation_id,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            message=message,
            trigger_type="first_greeting",
            skip_post_process=True,
        )
        return True
    except Exception as e:
        logger.warning(f"send_first_greeting failed: {e}")
        return False


# ────────────────────────────────────────────────────────────────────
# 后台任务
# ────────────────────────────────────────────────────────────────────

async def _bg_proactive_ai_memory(user_id: str, message: str) -> None:
    """Spec §2.2：把刚发出的主动消息送进 per-message AI 自我记忆 pipeline。"""
    try:
        await process_memory_pipeline(
            user_id=user_id,
            conversation_text=f"assistant: {message}",
            side="ai",
        )
    except Exception as e:
        logger.warning(f"Proactive AI memory pipeline failed: {e}")
