import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from prisma import Json
from sse_starlette.sse import EventSourceResponse

from app.db import db
from app.models.message import ChatRequest
from app.services.chat_service import stream_chat_response
from app.services.aggregation import is_short_message, push_pending, flush_pending
from app.services.proactive import generate_proactive_message, get_proactive_history

router = APIRouter(prefix="/chat", tags=["chat"])


async def _empty_stream() -> AsyncGenerator[dict, None]:
    """空SSE流：碎片消息已入队，暂无AI回复。"""
    yield {"event": "done", "data": json.dumps({"message_id": "pending"})}


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
    pending_text, _ = await flush_pending(user_id)

    if is_short_message(data.message) and not pending_text:
        # 短消息，加入聚合队列，暂不触发AI回复
        await push_pending(user_id, conversation_id, data.message)
        # 保存碎片到DB（用户能看到自己发的消息）
        await db.message.create(
            data={
                "conversation": {"connect": {"id": conversation_id}},
                "role": "user",
                "content": data.message,
                "metadata": Json({"fragment": True}),
            }
        )
        return EventSourceResponse(_empty_stream())

    # 有聚合文本：拼接后处理
    final_message = data.message
    if pending_text:
        final_message = pending_text + " " + data.message

    return EventSourceResponse(
        stream_chat_response(
            conversation_id=conversation_id,
            user_message=final_message,
            agent=conv.agent,
            user_id=user_id,
        )
    )


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
