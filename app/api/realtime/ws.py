"""WebSocket 聊天端点。

替代 SSE 的持久双向连接，支持碎片聚合推送和主动消息推送。
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from prisma import Json

from app.db import db
from app.services.interaction.aggregation import is_short_message, push_pending, flush_pending
from app.services.chat.orchestrator import stream_chat_response
from app.services.interaction.delayed_queue import enqueue_delayed_message
from app.services.relationship.emotion import quick_emotion_estimate, get_ai_emotion
from app.services.interaction.reply_context import build_reply_timing_context, merge_reply_contexts
from app.services.schedule_domain.schedule import generate_daily_schedule, get_cached_schedule, get_current_status
from app.services.mbti import get_mbti
from app.services.proactive.state import mark_user_replied_for_conversation
from app.services.proactive.sender import send_first_greeting
from app.services.runtime.ws_manager import manager

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
    from app.redis_client import is_redis_healthy

    if not is_redis_healthy():
        # readonly mode: 无 Redis 无法跑聚合 / 延迟队列 / 计数, 拒绝新连接
        # code=1011: Internal Server Error (WebSocket 协议语义)
        await websocket.close(code=1011, reason="redis_unavailable")
        return

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

    # spec §12 开场主动第一句话: 只在首次进入 (0 消息) 时触发
    try:
        asyncio.create_task(
            send_first_greeting(
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id=agent.id,
                workspace_id=getattr(conv, "workspaceId", None),
            )
        )
    except Exception as e:
        logger.warning(f"first_greeting dispatch failed conv={conversation_id[:8]}: {e}")

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
            agent.id, agent.name, get_mbti(agent), user_id=user_id,
        )
    received_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "type": "leisure", "status": "idle"}
    # Spec §6.2 延迟计算用缓存 PAD（消息入队前，在 orchestrator compute_ai_pad 之前）
    ai_emotion = await get_ai_emotion(agent.id)
    current_context = await build_reply_timing_context(
        agent_id=agent.id,
        user_id=user_id,
        received_status=received_status,
        user_emotion=quick_emotion_estimate(text),
        ai_emotion=ai_emotion,
    )

    # spec §1.3 窗口管理规则
    # - 碎片 + 无 pending → 新建窗口
    # - 碎片 + 已有 pending → 追加 + 刷新窗口（push_pending 里的 zadd 自动刷新 due_at）
    # - 非碎片 + 已有 pending → 打断触发（flush 合并后处理）
    # - 非碎片 + 无 pending → 直接处理
    is_fragment = is_short_message(text)

    if is_fragment:
        message_id = await _persist_user_message(
            conversation_id,
            text,
            metadata={"fragment": True},
        )
        # push_pending 内部 zadd 刷新 due_at = now + 5，同时覆盖最新 reply_context
        await push_pending(
            agent_id=agent.id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            reply_context=current_context,
            message_id=message_id,
        )
        await ws.send_json({"type": "pending", "data": {"status": "aggregating"}})
        return

    # 非碎片：若有 pending，先打断触发合并；否则直接处理当前消息
    pending_text, _, pending_context, _ = await flush_pending(
        agent_id=agent.id, user_id=user_id,
    )

    final_message = text
    final_context = current_context
    if pending_text:
        # spec §1.5: 按原始顺序直接连接（中文不加空格），当前非碎片作为最后一条
        final_message = "".join(part for part in [pending_text, text] if part)
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


async def stream_to_ws(ws: WebSocket, generator, conversation_id: str | None = None) -> None:
    """将 stream_chat_response() 的 yield 转为 WS send_json。

    每次推送前重新从 manager 获取当前活跃 WS。这对应前端重连场景：
    LLM 流式生成往往持续几十秒，期间原 WS 可能被新连接替换。
    如果只保留最初的 ws 引用，后续 send_json 会推给已关闭的连接，
    消息丢失，前端永远等不到 reply。
    """
    async for event in generator:
        event_type = event.get("event", "")
        data_str = event.get("data") or "{}"
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            data = {"raw": data_str}

        # Prefer the current active WS for the conversation (handles reconnect).
        target = manager.get(conversation_id) if conversation_id else None
        if target is None:
            target = ws
        try:
            if target.client_state == WebSocketState.CONNECTED:
                await target.send_json({"type": event_type, "data": data})
            else:
                logger.debug(
                    f"WS not connected, dropped event type={event_type} "
                    f"for conv={(conversation_id or '')[:8]}"
                )
        except Exception as e:
            logger.warning(
                f"WS send failed for conv={(conversation_id or '')[:8]} "
                f"type={event_type}: {e}"
            )
