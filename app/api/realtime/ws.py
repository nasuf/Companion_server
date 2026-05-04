"""WebSocket 聊天端点。

替代 SSE 的持久双向连接，支持碎片聚合推送和主动消息推送。
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from prisma import Json

from app.db import db
from app.observability import bind_context
from app.observability.events import EVT_WS_CONNECT, EVT_WS_DISCONNECT, EVT_WS_MESSAGE_RECV
from app.redis_client import is_redis_healthy
from app.services.interaction.aggregation import is_short_message, push_pending, flush_pending
from app.services.interaction.delayed_queue import enqueue_or_append_delayed
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
    agent,
    user_id: str,
    user_message: str,
    user_message_id: str | None,
    reply_context: dict | None,
) -> None:
    delay_seconds = float((reply_context or {}).get("delay_seconds", 0.0) or 0.0)

    # delay=0 同步快路径: 跳过 delayed queue + scheduler 1s 调度延迟, 直接走 orchestrator
    # 流式回复. settings.reply_delay_enabled=False (默认) 时所有消息走这里; 即便开启
    # 延迟, 偶尔某次随机出 0 也走快路径. 不发 "queued" pending 事件 (前端不显示"已排队").
    if delay_seconds <= 0.0:
        from app.services.chat.orchestrator import stream_chat_response
        gen = stream_chat_response(
            conversation_id=conversation_id,
            user_message=user_message,
            agent=agent,
            user_id=user_id,
            reply_context=reply_context,
            save_user_message=False,
            user_message_id=user_message_id,
            delivered_from_queue=True,
        )
        await stream_to_ws(gen, conversation_id)
        return

    # 用户连发非碎片：若已有 pending payload，append 到同一 due_at（沿用 spec §6.3
    # 时间戳沿用语义），scheduler flush 时一次拿到合并处理，避免双发。
    appended = await enqueue_or_append_delayed(
        conversation_id,
        {
            "conversation_id": conversation_id,
            "agent_id": agent.id,
            "user_id": user_id,
            "message": user_message,
            "message_id": user_message_id,
            "reply_context": reply_context,
        },
        delay_seconds,
    )
    if delay_seconds > 5 and not appended:
        await ws.send_json({"type": "delay", "data": {"duration": delay_seconds}})
    await ws.send_json({"type": "pending", "data": {"status": "queued", "delay": delay_seconds}})


@router.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket 聊天连接。"""
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
    workspace_id = getattr(conv, "workspaceId", None)
    # username 一次性查询缓存 — 整个 WS 生命周期复用, 避免每条 message 查 DB
    user_record = await db.user.find_unique(where={"id": user_id})
    cached_username = user_record.username if user_record else None

    # 绑定整个 WS 连接生命周期的 context — 内层 message handler / 派生的所有
    # asyncio.create_task / fire_background 自动继承
    with bind_context(
        conversation_id=conversation_id,
        workspace_id=workspace_id,
        agent_id=agent.id,
        agent_name=agent.name,
        user_id=user_id,
        username=cached_username,
    ):
        await websocket.accept()
        await manager.connect(
            conversation_id, user_id, websocket, workspace_id=workspace_id,
        )
        logger.info("ws connected", extra={"event": EVT_WS_CONNECT})

        # spec §12 开场主动第一句话: 只在首次进入 (0 消息) 时触发
        try:
            asyncio.create_task(
                send_first_greeting(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=agent.id,
                    workspace_id=workspace_id,
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
                    payload = raw.get("data") or {}
                    text = (payload.get("message") or "").strip()
                    if not text:
                        continue
                    # client_id (optional, 但前端推荐传) — 让 ack 事件带回供前端 reconcile.
                    # 不传时 ack 仅含 message_id (DB id), 前端按时间顺序匹配.
                    client_id = payload.get("client_id")
                    logger.info(
                        "ws message received",
                        extra={"event": EVT_WS_MESSAGE_RECV, "msg_len": len(text)},
                    )
                    await _handle_message(
                        websocket, conversation_id, user_id, agent, text,
                        client_id=client_id,
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
            logger.info("ws disconnected", extra={"event": EVT_WS_DISCONNECT})
            await manager.disconnect(conversation_id)


async def _send_ack(
    ws: WebSocket, *, message_id: str, client_id: str | None,
) -> None:
    """spec 之外的工程扩展: persist 落库后立刻发"已读" ack 给前端.

    用户体感"我说出去的话 AI 看到了" — 之前从用户发到 AI 实际开始回复中间
    这段无任何反馈 (1-5s 在延迟队列), 气泡像石沉大海. ack 让前端能在气泡
    旁加 ✓✓ 标记.

    `client_id`: 前端发消息时塞的 UUID, 后端原样回. 让前端在快速连发多条时
    精确对应 ack 跟具体气泡, 不用按时间猜. 没传时仅返 message_id (DB id).
    """
    from app.services.schedule_domain.time_service import _now_corrected
    try:
        await ws.send_json({
            "type": "ack",
            "data": {
                "message_id": message_id,
                "client_id": client_id,
                "received_at": _now_corrected().isoformat(),
            },
        })
    except Exception as e:
        # ack 失败不影响主流程 (回复仍会发) — 前端最终拿到 reply 也能反推消息已收到
        logger.warning(f"[WS] send_ack failed: {e}")


async def _handle_message(
    ws: WebSocket,
    conversation_id: str,
    user_id: str,
    agent,
    text: str,
    *,
    client_id: str | None = None,
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
        await _send_ack(ws, message_id=message_id, client_id=client_id)
        # push_pending 内部 zadd 刷新 due_at = now + 5，同时覆盖最新 reply_context
        pushed = await push_pending(
            agent_id=agent.id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            reply_context=current_context,
            message_id=message_id,
        )
        if pushed:
            await ws.send_json({"type": "pending", "data": {"status": "aggregating"}})
            return
        # Redis 挂 → 聚合失败, 消息已持久化 (fragment=True); 直接入延迟队列走同步回复
        logger.warning(
            f"[WS] push_pending failed conv={conversation_id[:8]}, fallback to sync queue"
        )
        try:
            await _queue_reply(
                ws,
                conversation_id=conversation_id,
                agent=agent,
                user_id=user_id,
                user_message=text,
                user_message_id=message_id,
                reply_context=current_context,
            )
        except Exception as e:
            logger.error(f"Chat queue failed for conv={conversation_id[:8]}: {e}")
            await ws.send_json({"type": "error", "data": {"message": "消息入队失败"}})
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
    await _send_ack(ws, message_id=user_message_id, client_id=client_id)

    try:
        await _queue_reply(
            ws,
            conversation_id=conversation_id,
            agent=agent,
            user_id=user_id,
            user_message=final_message,
            user_message_id=user_message_id,
            reply_context=final_context,
        )
    except Exception as e:
        logger.error(f"Chat queue failed for conv={conversation_id[:8]}: {e}")
        await ws.send_json({"type": "error", "data": {"message": "消息入队失败"}})


async def stream_to_ws(generator, conversation_id: str) -> None:
    """将 stream_chat_response() 的 yield 转为 WS 推送 (跨进程兼容).

    通过 manager.send_event 路由到持有 conversation_id 连接的 worker:
    - 同进程: fast path 本地直送
    - 跨进程 (multi-worker / scheduler 拆容器): publish 到 ws:conv:{conv_id}
    - 前端断连重连切到别的 worker: 新 worker 接管, publish 自动路由过去
    """
    async for event in generator:
        event_type = event.get("event", "")
        data_str = event.get("data") or "{}"
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            data = {"raw": data_str}
        await manager.send_event(conversation_id, event_type, data)
