"""主动消息统一发送入口。"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from prisma import Json

from app.db import db
from app.services.llm.models import get_chat_model, invoke_text
from app.services.memory.self_memory import generate_daily_self_memories
from app.services.proactive.history import (
    can_send_proactive, can_send_proactive_2day,
    increment_proactive_count, increment_proactive_2day_count,
)
from app.services.proactive.context import build_proactive_context
from app.services.proactive.state import (
    ProactiveStateRecord,
    ensure_proactive_state_for_workspace,
    get_active_workspace_context,
    log_proactive_event,
    mark_proactive_sent,
)
from app.services.prompting.store import get_prompt_text
from app.services.interaction.reply_context import save_last_reply_timestamp
from app.services.runtime.ws_manager import manager

logger = logging.getLogger(__name__)

UTC = timezone.utc
SENDABLE_PROACTIVE_STATUSES = {"idle"}


async def generate_and_send_proactive(
    state: ProactiveStateRecord,
    *,
    trigger_type: str,
    now: datetime | None = None,
) -> bool:
    now_ts = now or datetime.now(UTC)
    if not await can_send_proactive(state.agent_id, state.user_id):
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="send_skipped",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={"reason": "daily_limit"},
        )
        return False

    if not await can_send_proactive_2day(state.agent_id, state.user_id):
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="send_skipped",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={"reason": "2day_limit"},
        )
        return False

    workspace_context = await get_active_workspace_context(state.workspace_id)
    if not workspace_context:
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="send_skipped",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={"reason": "workspace_missing"},
        )
        return False

    conversation_id = str(workspace_context.get("conversation_id") or state.conversation_id or "")
    if not conversation_id:
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=state.conversation_id,
            event_type="send_skipped",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={"reason": "conversation_missing"},
        )
        return False

    # 记忆去重: 排除上次主动已用的记忆 ID
    exclude_memory_ids = set((state.metadata or {}).get("used_memory_ids", []))

    ctx = await build_proactive_context(
        workspace_id=state.workspace_id,
        user_id=state.user_id,
        agent_id=state.agent_id,
        trigger_type=trigger_type,
        stage=state.stage,
        exclude_memory_ids=exclude_memory_ids,
    )

    # 如果是记忆主动但去重后无可用记忆，fallback 到沉默唤醒
    if trigger_type == "memory_proactive" and not ctx.get("proactive_memories"):
        trigger_type = "silence_wakeup"
        ctx["trigger_type"] = trigger_type
        ctx["scene_hint"] = "优先用轻量、低打扰的方式重新建立联系。"

    message = await _generate_message(ctx)
    if not message:
        await log_proactive_event(
            state_id=state.id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            agent_id=state.agent_id,
            conversation_id=conversation_id,
            event_type="generation_skipped",
            window_index=state.current_window_index,
            trigger_type=trigger_type,
            payload={"reason": "empty_or_skip"},
        )
        return False

    created = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "assistant",
            "content": message,
            "metadata": Json({
                "proactive": True,
                "trigger_type": trigger_type,
                "stage": state.stage,
            }),
        }
    )

    await increment_proactive_count(state.agent_id, state.user_id)
    await increment_proactive_2day_count(state.agent_id, state.user_id)
    try:
        await db.proactivechatlog.create(
            data={
                "agent": {"connect": {"id": state.agent_id}},
                "userId": state.user_id,
                "workspaceId": state.workspace_id,
                "conversationId": conversation_id,
                "message": message,
                "eventType": trigger_type,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to write proactive chat log: {e}")

    # 持久化已用记忆 ID 到 state metadata，供下次去重
    used_memory_ids = list(exclude_memory_ids | set(ctx.get("used_memory_ids", [])))
    await mark_proactive_sent(
        state,
        trigger_type=trigger_type,
        message=message,
        assistant_message_id=created.id,
        now=now_ts,
        mark_daily_scene=(trigger_type == "scheduled_scene"),
        extra_metadata={"used_memory_ids": used_memory_ids},
    )
    await save_last_reply_timestamp(state.agent_id, state.user_id, when=now_ts)

    await manager.send_to_user(
        state.user_id,
        "proactive",
        {
            "text": message,
            "agent_id": state.agent_id,
            "assistant_message_id": created.id,
            "trigger_type": trigger_type,
        },
    )

    asyncio.create_task(_bg_self_memory(state.agent_id, state.user_id, message))
    return True


async def send_manual_or_triggered_proactive(
    *,
    workspace_id: str,
    trigger_type: str,
    now: datetime | None = None,
) -> dict[str, str | bool | None]:
    state = await ensure_proactive_state_for_workspace(
        workspace_id,
        now=now,
        reason="manual_or_triggered",
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


async def _generate_message(ctx: dict) -> str | None:
    agent = ctx["agent"]
    emotion = ctx["emotion"] or {}
    pleasure = float(emotion.get("pleasure", 0.0))
    mood = "不错" if pleasure > 0.2 else ("有点低落" if pleasure < -0.2 else "平静")
    memories = ctx.get("proactive_memories") or ctx.get("core_memories") or []
    prompt = (await get_prompt_text("proactive.message")).format(
        ai_name=agent.name,
        mood=mood,
        pleasure=emotion.get("pleasure", 0),
        arousal=emotion.get("arousal", 0),
        dominance=emotion.get("dominance", 0),
        memories="\n".join(f"- {m}" for m in memories) or "暂无记忆。",
    )
    prompt += (
        f"\n\n补充上下文：\n"
        f"- 当前主动类型：{ctx['trigger_type']}\n"
        f"- 当前关系阶段：{ctx['stage']} / {ctx['relationship_stage']}\n"
        f"- 距离上次聊天约：{ctx['silence_hours']:.1f} 小时\n"
        f"- 当前情景提示：{ctx['scene_hint']}\n"
        "额外规则：\n"
        "- 语气自然，不要像系统提醒。\n"
        "- 只发一条消息，1-2句话。\n"
        "- 不要用“我突然想起”“很久没联系”这种刻意铺垫。\n"
        "- 如果是记忆主动，优先抓具体记忆点，不要空泛寒暄。\n"
        "- 如果当前上下文不适合主动发起，返回 SKIP。\n"
    )

    model = get_chat_model()
    response = (await invoke_text(model, prompt)).strip()
    if response == "SKIP" or len(response) < 4:
        return None
    return response


async def _bg_self_memory(agent_id: str, user_id: str, message: str) -> None:
    try:
        await generate_daily_self_memories(
            agent_id=agent_id,
            user_id=user_id,
            dialogue_summary=f"AI主动对用户说：{message}",
        )
    except Exception as e:
        logger.warning(f"Proactive self-memory generation failed: {e}")
