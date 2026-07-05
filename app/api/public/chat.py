import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from prisma import Json
from sse_starlette.sse import EventSourceResponse

from app.api.deps import require_redis
from app.api.jwt_auth import require_user
from app.api.ownership import require_user_self
from app.db import db
from app.models.message import ChatRequest
from app.services.interaction.delayed_queue import enqueue_or_append_delayed
from app.services.relationship.emotion import quick_emotion_estimate
from app.services.interaction.reply_context import build_reply_timing_context
from app.services.interaction.user_turn_aggregation import (
    enqueue_planned_user_message,
    plan_user_message_aggregation,
)
from app.services.schedule_domain.schedule import generate_daily_schedule, get_cached_schedule, get_current_status
from app.services.mbti import get_mbti
from app.services.proactive import get_proactive_history
from app.services.proactive.sender import send_manual_or_triggered_proactive
from app.services.proactive.state import mark_user_replied_for_conversation
from app.services.workspace.workspaces import resolve_workspace_id

router = APIRouter(prefix="/chat", tags=["chat"])


async def _empty_stream() -> AsyncGenerator[dict, None]:
    """空SSE流：用户消息已进入聚合窗口，暂无AI回复。"""
    yield {"event": "pending", "data": json.dumps({"status": "aggregating"})}
    yield {"event": "done", "data": json.dumps({"message_id": "pending"})}


async def _queued_stream(delay_seconds: float) -> AsyncGenerator[dict, None]:
    """SSE queue acknowledgement stream for delayed delivery."""
    if delay_seconds > 5:
        yield {"event": "delay", "data": json.dumps({"duration": delay_seconds})}
    yield {"event": "pending", "data": json.dumps({"status": "queued", "delay": delay_seconds})}
    yield {"event": "done", "data": json.dumps({"message_id": "queued"})}


async def _persist_user_message(
    conversation_id: str,
    text: str,
    *,
    metadata: dict | None = None,
) -> str:
    saved = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "user",
            "content": text,
            **({"metadata": Json(metadata)} if metadata else {}),
        }
    )
    await mark_user_replied_for_conversation(conversation_id)
    return saved.id


@router.post("/{conversation_id}", dependencies=[Depends(require_redis)])
async def chat(
    conversation_id: str,
    data: ChatRequest,
    user: dict = Depends(require_user),
):
    conv = await db.conversation.find_unique(
        where={"id": conversation_id},
        include={"agent": True},
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.isDeleted:
        raise HTTPException(status_code=410, detail="Conversation deleted")
    # Ownership: conversation_id is not a capability token (admins bypass).
    if user.get("role") != "admin" and conv.userId != user.get("sub"):
        raise HTTPException(status_code=403, detail="Not your conversation")

    # 阻断: agent 还在初始化中（人生经历生成未完成）
    if conv.agent and conv.agent.status == "provisioning":
        raise HTTPException(
            status_code=503,
            detail="AI 正在初始化中，请稍等...",
        )

    user_id = conv.userId

    # --- 用户回合聚合：fragment/turn 策略由 user_turn_aggregation 统一决定 ---
    schedule = await get_cached_schedule(conv.agent.id)
    if not schedule:
        schedule = await generate_daily_schedule(
            conv.agent.id, conv.agent.name, get_mbti(conv.agent), user_id=user_id,
        )
    received_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "type": "leisure", "status": "idle"}
    current_context = await build_reply_timing_context(
        agent_id=conv.agent.id,
        user_id=user_id,
        received_status=received_status,
        user_emotion=quick_emotion_estimate(data.message),
    )

    plan = await plan_user_message_aggregation(
        agent_id=conv.agent.id,
        user_id=user_id,
        conversation_id=conversation_id,
        text=data.message,
        reply_context=current_context,
    )
    message_id = await _persist_user_message(
        conversation_id,
        data.message,
        metadata=plan.metadata,
    )
    if plan.should_wait:
        pushed = await enqueue_planned_user_message(plan, message_id=message_id)
        if pushed:
            return EventSourceResponse(_empty_stream())
        # Redis 挂 → 聚合失败, 当作完整消息入延迟队列 (跳聚合, 不合并)
        delay_seconds = float((plan.fallback_context or {}).get("delay_seconds", 0.0) or 0.0)
        await enqueue_or_append_delayed(
            conversation_id,
            {
                "conversation_id": conversation_id,
                "agent_id": conv.agent.id,
                "user_id": user_id,
                "message": plan.fallback_message,
                "message_id": message_id,
                "reply_context": plan.fallback_context,
            },
            delay_seconds,
        )
        return EventSourceResponse(_queued_stream(delay_seconds))

    delay_seconds = float((plan.final_context or {}).get("delay_seconds", 0.0) or 0.0)
    # 原子入队：若已有待处理消息则追加（不延长等待），否则新建
    await enqueue_or_append_delayed(
        conversation_id,
        {
            "conversation_id": conversation_id,
            "agent_id": conv.agent.id,
            "user_id": user_id,
            "message": plan.final_message,
            "message_id": message_id,
            "reply_context": plan.final_context,
        },
        delay_seconds,
    )

    return EventSourceResponse(_queued_stream(delay_seconds))


@router.post("/proactive/{agent_id}", dependencies=[Depends(require_redis)])
async def trigger_proactive(
    agent_id: str, user_id: str, _user=Depends(require_user_self),
):
    """触发AI主动消息。"""
    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    if not workspace_id:
        return {"message": None, "reason": "workspace_not_found"}
    result = await send_manual_or_triggered_proactive(
        workspace_id=workspace_id,
        trigger_type="manual_trigger",
    )
    if not result["ok"]:
        return {"message": None, "reason": "no_content_or_limit_reached"}
    return {"message": result["message"]}


@router.get("/proactive/{agent_id}/history")
async def proactive_history(
    agent_id: str, user_id: str, limit: int = 10, _user=Depends(require_user_self),
):
    """获取主动消息历史。"""
    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    history = await get_proactive_history(agent_id, user_id, limit, workspace_id=workspace_id)
    return {"history": history}
