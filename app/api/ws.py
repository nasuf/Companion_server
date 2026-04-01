"""WebSocket 聊天端点。

替代 SSE 的持久双向连接，支持碎片聚合推送和主动消息推送。
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from prisma import Json

from app.db import db
from app.services.aggregation import is_short_message, push_pending, flush_pending
from app.services.chat_service import stream_chat_response
from app.services.delayed_queue import enqueue_delayed_message
from app.services.emotion import quick_emotion_estimate
from app.services.reply_context import build_reply_timing_context, merge_reply_contexts
from app.services.schedule import generate_daily_schedule, get_cached_schedule, get_current_status
from app.services.trait_model import get_seven_dim
from app.services.proactive_state import mark_user_replied_for_conversation
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

_IDLE_TIMEOUT = 90.0


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


async def _queue_reply(
    ws: WebSocket,
    *,
    conversation_id: str,
    agent_id: str,
    user_id: str,
    user_message: str,
    user_message_id: str | None,
    reply_context: dict | None,
) -> None:
    delay_seconds = float((reply_context or {}).get("delay_seconds", 0.0) or 0.0)
    await enqueue_delayed_message(
        conversation_id,
        {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "user_id": user_id,
            "message": user_message,
            "message_id": user_message_id,
            "reply_context": reply_context,
        },
        delay_seconds,
    )
    if delay_seconds > 5:
        await ws.send_json({"type": "delay", "data": {"duration": delay_seconds}})
    await ws.send_json({"type": "pending", "data": {"status": "queued", "delay": delay_seconds}})


@router.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket 聊天连接。"""
    conv = await db.conversation.find_unique(
        where={"id": conversation_id},
        include={"agent": True},
    )
    if not conv or conv.isDeleted or not conv.agent:
        await websocket.close(code=4004, reason="Conversation not found")
        return

    user_id = conv.userId
    agent = conv.agent

    await websocket.accept()
    await manager.connect(conversation_id, user_id, websocket)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=_IDLE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                await websocket.close(code=4008, reason="Idle timeout")
                break

            msg_type = raw.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "message":
                text = (raw.get("data") or {}).get("message", "").strip()
                if not text:
                    continue
                await _handle_message(
                    websocket, conversation_id, user_id, agent, text
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WS error conv={conversation_id[:8]}: {e}")
        try:
            await websocket.send_json(
                {"type": "error", "data": {"message": str(e)}}
            )
        except Exception:
            pass
    finally:
        await manager.disconnect(conversation_id)


async def _handle_message(
    ws: WebSocket,
    conversation_id: str,
    user_id: str,
    agent,
    text: str,
) -> None:
    """处理用户消息：聚合检查 → 生成回复 → 推送。"""
    schedule = await get_cached_schedule(agent.id)
    if not schedule:
        schedule = await generate_daily_schedule(
            agent.id, agent.name, get_seven_dim(agent), user_id=user_id,
        )
    received_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "type": "leisure", "status": "idle"}
    current_context = await build_reply_timing_context(
        agent_id=agent.id,
        user_id=user_id,
        received_status=received_status,
        user_emotion=quick_emotion_estimate(text),
    )

    pending_text, _, pending_context, _ = await flush_pending(user_id)

    if is_short_message(text) and not pending_text:
        message_id = await _persist_user_message(
            conversation_id,
            text,
            metadata={"fragment": True},
        )
        await push_pending(user_id, conversation_id, text, current_context, message_id)
        await ws.send_json({"type": "pending", "data": {"status": "aggregating"}})
        return

    final_message = text
    final_context = current_context
    user_message_id: str | None = None
    if pending_text:
        final_message = " ".join(part for part in [pending_text, text] if part)
        final_context = merge_reply_contexts(pending_context, current_context)
    user_message_id = await _persist_user_message(
        conversation_id,
        text,
        metadata={"queued": True},
    )

    try:
        await _queue_reply(
            ws,
            conversation_id=conversation_id,
            agent_id=agent.id,
            user_id=user_id,
            user_message=final_message,
            user_message_id=user_message_id,
            reply_context=final_context,
        )
    except Exception as e:
        logger.error(f"Chat queue failed for conv={conversation_id[:8]}: {e}")
        await ws.send_json({"type": "error", "data": {"message": "消息入队失败"}})


async def stream_to_ws(ws: WebSocket, generator) -> None:
    """将 stream_chat_response() 的 yield 转为 WS send_json。"""
    async for event in generator:
        event_type = event.get("event", "")
        data_str = event.get("data") or "{}"
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            data = {"raw": data_str}
        await ws.send_json({"type": event_type, "data": data})
