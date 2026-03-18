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
from app.services.ws_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

_IDLE_TIMEOUT = 90.0


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
    pending_text, _ = await flush_pending(user_id)

    if is_short_message(text) and not pending_text:
        await push_pending(user_id, conversation_id, text)
        await db.message.create(
            data={
                "conversation": {"connect": {"id": conversation_id}},
                "role": "user",
                "content": text,
                "metadata": Json({"fragment": True}),
            }
        )
        await ws.send_json({"type": "pending", "data": {"status": "aggregating"}})
        return

    final_message = text
    if pending_text:
        final_message = pending_text + " " + text

    try:
        await stream_to_ws(
            ws,
            stream_chat_response(
                conversation_id=conversation_id,
                user_message=final_message,
                agent=agent,
                user_id=user_id,
            ),
        )
    except Exception as e:
        logger.error(f"Chat stream failed for conv={conversation_id[:8]}: {e}")
        await ws.send_json({"type": "error", "data": {"message": "回复生成失败"}})


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
