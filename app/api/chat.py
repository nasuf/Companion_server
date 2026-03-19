import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from prisma import Json
from sse_starlette.sse import EventSourceResponse

from app.db import db
from app.models.message import ChatRequest
from app.services.aggregation import is_short_message, push_pending, flush_pending
from app.services.delayed_queue import enqueue_delayed_message
from app.services.emotion import quick_emotion_estimate
from app.services.reply_context import build_reply_timing_context, merge_reply_contexts
from app.services.schedule import generate_daily_schedule, get_cached_schedule, get_current_status
from app.services.trait_model import get_seven_dim
from app.services.proactive import generate_proactive_message, get_proactive_history

router = APIRouter(prefix="/chat", tags=["chat"])


async def _empty_stream() -> AsyncGenerator[dict, None]:
    """空SSE流：碎片消息已入队，暂无AI回复。"""
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
    return saved.id


@router.post("/{conversation_id}")
async def chat(conversation_id: str, data: ChatRequest):
    conv = await db.conversation.find_unique(
        where={"id": conversation_id},
        include={"agent": True},
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.isDeleted:
        raise HTTPException(status_code=410, detail="Conversation deleted")

    user_id = conv.userId

    # --- 12E: 碎片化消息聚合 (PRD §3.4) ---
    # 先检查是否有待聚合碎片（被当前消息打断）
    schedule = await get_cached_schedule(conv.agent.id)
    if not schedule:
        schedule = await generate_daily_schedule(
            conv.agent.id, conv.agent.name, get_seven_dim(conv.agent), user_id=user_id,
        )
    received_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "type": "leisure", "status": "idle"}
    current_context = await build_reply_timing_context(
        agent_id=conv.agent.id,
        user_id=user_id,
        received_status=received_status,
        user_emotion=quick_emotion_estimate(data.message),
    )

    pending_text, _, pending_context, _ = await flush_pending(user_id)

    if is_short_message(data.message) and not pending_text:
        # 短消息，加入聚合队列，暂不触发AI回复
        message_id = await _persist_user_message(
            conversation_id,
            data.message,
            metadata={"fragment": True},
        )
        await push_pending(user_id, conversation_id, data.message, current_context, message_id)
        # 保存碎片到DB（用户能看到自己发的消息）
        return EventSourceResponse(_empty_stream())

    # 有聚合文本：拼接后处理
    final_message = data.message
    final_context = current_context
    if pending_text:
        final_message = " ".join(part for part in [pending_text, data.message] if part)
        final_context = merge_reply_contexts(pending_context, current_context)

    message_id = await _persist_user_message(
        conversation_id,
        data.message,
        metadata={"queued": True},
    )
    delay_seconds = float((final_context or {}).get("delay_seconds", 0.0) or 0.0)
    await enqueue_delayed_message(
        conversation_id,
        {
            "conversation_id": conversation_id,
            "agent_id": conv.agent.id,
            "user_id": user_id,
            "message": final_message,
            "message_id": message_id,
            "reply_context": final_context,
        },
        delay_seconds,
    )
    return EventSourceResponse(_queued_stream(delay_seconds))


@router.post("/proactive/{agent_id}")
async def trigger_proactive(agent_id: str, user_id: str):
    """触发AI主动消息。"""
    message = await generate_proactive_message(user_id, agent_id)
    if not message:
        return {"message": None, "reason": "no_content_or_limit_reached"}
    return {"message": message}


@router.get("/proactive/{agent_id}/history")
async def proactive_history(agent_id: str, user_id: str, limit: int = 10):
    """获取主动消息历史。"""
    history = await get_proactive_history(agent_id, user_id, limit)
    return {"history": history}
