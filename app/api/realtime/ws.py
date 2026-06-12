"""WebSocket 聊天端点。

替代 SSE 的持久双向连接，支持用户回合聚合推送和主动消息推送。
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
from app.services.interaction.delayed_queue import enqueue_or_append_delayed
from app.services.interaction.reply_context import build_reply_timing_context
from app.services.interaction.user_turn_aggregation import (
    enqueue_planned_user_message,
    plan_user_message_aggregation,
)
from app.services.schedule_domain.schedule import generate_daily_schedule, get_cached_schedule, get_current_status
from app.services.mbti import get_mbti
from app.services.proactive.state import mark_user_replied_for_conversation
from app.services.proactive.sender import send_first_greeting
from app.services.relationship.emotion import quick_emotion_estimate
from app.services.runtime.ws_manager import manager
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

_IDLE_TIMEOUT = 90.0


def _message_metadata(
    base: dict | None = None,
    *,
    client_id: str | None = None,
    component_card: dict | None = None,
) -> dict | None:
    metadata = dict(base or {})
    if client_id:
        metadata["client_id"] = client_id
    if component_card:
        metadata["component_card"] = component_card
    return metadata or None


def _sanitize_component_card(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    card_type = raw.get("type")
    if card_type not in {
        "time_capsule",
        "weather",
        "checkin_reminder",
        "checkin_habit",
        "music_track",
    }:
        return None
    payload = _sanitize_component_card_payload(card_type, raw.get("payload"))
    card: dict = {
        "version": 1,
        "type": card_type,
        "title": str(raw.get("title") or "")[:80],
        "subtitle": str(raw.get("subtitle") or "")[:120],
        "body": str(raw.get("body") or "")[:1000],
        "footer": str(raw.get("footer") or "")[:120],
        "accent": str(raw.get("accent") or "#7C3CFF")[:16],
    }
    if payload is not None:
        card["payload"] = payload
    return card


def _truncate_payload_value(value: object, limit: int) -> str:
    return str(value or "")[:limit]


def _sanitize_component_card_payload(card_type: object, raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    if card_type == "time_capsule":
        payload: dict = {}
        capsule_id = _truncate_payload_value(raw.get("capsule_id"), 80).strip()
        if capsule_id:
            payload["capsule_id"] = capsule_id
        for key in ("created_date", "open_date"):
            value = _truncate_payload_value(raw.get(key), 32).strip()
            if value:
                payload[key] = value
        content = _truncate_payload_value(raw.get("content"), 1000)
        if content:
            payload["content"] = content
        return payload or None
    if card_type == "weather":
        payload = {}
        for key in ("location", "date", "condition", "temperature", "unit"):
            value = _truncate_payload_value(raw.get(key), 80).strip()
            if value:
                payload[key] = value
        return payload or None
    if card_type in {"checkin_reminder", "checkin_habit"}:
        payload = {}
        for key in (
            "trigger_id",
            "reminder_id",
            "habit_id",
            "summary",
            "recurrence",
            "trigger_time",
        ):
            value = _truncate_payload_value(raw.get(key), 120).strip()
            if value:
                payload[key] = value
        weekdays = _sanitize_weekdays(raw.get("habit_weekdays"))
        if weekdays:
            payload["habit_weekdays"] = weekdays
        sent_to_ai = raw.get("sent_to_ai")
        if isinstance(sent_to_ai, bool):
            payload["sent_to_ai"] = sent_to_ai
        return payload or None
    if card_type == "music_track":
        intent = _truncate_payload_value(raw.get("intent"), 40).strip()
        if intent not in {"invite", "recommend"}:
            intent = "invite"
        source = _truncate_payload_value(raw.get("source"), 80).strip() or "music_page"
        raw_track = raw.get("track")
        if not isinstance(raw_track, dict):
            return None
        track = _sanitize_music_track_payload(raw_track)
        if track is None:
            return None
        payload = {"intent": intent, "source": source, "track": track}
        for key, limit in (
            ("mode", 40),
            ("library", 120),
            ("library_title", 120),
        ):
            value = _truncate_payload_value(raw.get(key), limit).strip()
            if value:
                payload[key] = value
        return payload
    return None


def _sanitize_music_track_payload(raw: dict) -> dict | None:
    track_id = _truncate_payload_value(raw.get("id"), 160).strip()
    title = _truncate_payload_value(raw.get("title"), 240).strip()
    if not track_id or not title:
        return None
    metadata = raw.get("metadata")
    safe_metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "id": track_id,
        "title": title,
        "artist": _truncate_payload_value(raw.get("artist") or "Jamendo", 160),
        "album": _truncate_payload_value(raw.get("album") or "Jamendo Library", 240),
        "library": _truncate_payload_value(raw.get("library") or "focus", 120),
        "url": _truncate_payload_value(raw.get("url"), 2000),
        "duration_sec": _safe_int(raw.get("duration_sec"), min_value=0, max_value=24 * 60 * 60),
        "cover_key": _truncate_payload_value(raw.get("cover_key") or "music-cover-01.jpg", 160),
        "accent_a": _truncate_payload_value(raw.get("accent_a") or "#1f6fff", 32),
        "accent_b": _truncate_payload_value(raw.get("accent_b") or "#18c6c0", 32),
        "source": _truncate_payload_value(raw.get("source") or "jamendo", 80),
        "metadata": safe_metadata,
    }


def _safe_int(value: object, *, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        parsed = 0
    return max(min_value, min(max_value, parsed))


def _sanitize_weekdays(raw: object) -> list[int]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.strip("[]").split(",")]
    else:
        return []
    days: list[int] = []
    for value in values:
        try:
            day = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= day <= 7 and day not in days:
            days.append(day)
    return sorted(days)


def _component_card_reply_message(text: str, component_card: dict | None) -> str | None:
    if not component_card:
        return None
    card_type = component_card.get("type")
    if card_type not in {"checkin_reminder", "checkin_habit"}:
        return None

    payload = component_card.get("payload") if isinstance(component_card.get("payload"), dict) else {}
    summary = str(payload.get("summary") or component_card.get("body") or text).strip()
    subtitle = str(component_card.get("subtitle") or "").strip()
    recurrence = str(payload.get("recurrence") or "").strip()
    weekdays = payload.get("habit_weekdays")
    weekday_text = ""
    if isinstance(weekdays, list) and weekdays:
        labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        picked = [labels[day - 1] for day in weekdays if isinstance(day, int) and 1 <= day <= 7]
        if picked:
            weekday_text = "每" + "、".join(picked)

    kind = "周期习惯打卡" if card_type == "checkin_habit" else "单次打卡提醒"
    # `trigger_time` is a machine ISO timestamp. Feeding it into the chat hot path
    # makes the message look like a normal explicit-time query and can trigger
    # time-memory retrieval. The user-visible subtitle already carries the date/time.
    details = [part for part in (subtitle, weekday_text) if part]
    detail_text = "；".join(dict.fromkeys(details))
    return (
        f"用户发送了一张已创建的{kind}卡片。"
        f"事项：{summary[:200]}。"
        f"{f'时间设置：{detail_text}。' if detail_text else ''}"
        "这张卡片已经由打卡系统保存并同步给你；不要重新创建提醒、不要反问时间，"
        "只需要基于卡片里的事项、周期和时间自然确认你会按这个设置提醒用户。"
    )


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
    metadata_keys = sorted((metadata or {}).keys())
    logger.debug(
        "ws user message persisted "
        f"message_id={saved.id[:8]} metadata_keys={metadata_keys} "
        f"has_component_card={bool((metadata or {}).get('component_card'))} "
        f"has_client_id={bool((metadata or {}).get('client_id'))}",
        extra={
            "message_id": saved.id,
            "metadata_keys": metadata_keys,
            "has_component_card": bool((metadata or {}).get("component_card")),
            "has_client_id": bool((metadata or {}).get("client_id")),
        },
    )
    await mark_user_replied_for_conversation(conversation_id)
    return saved.id


async def _persist_assistant_message(
    conversation_id: str,
    text: str,
    *,
    metadata: dict | None = None,
) -> str:
    created = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "assistant",
            "content": text,
            **({"metadata": Json(metadata)} if metadata else {}),
        }
    )
    return created.id


async def _handle_music_component_card(
    ws: WebSocket,
    *,
    conversation_id: str,
    user_id: str,
    agent,
    workspace_id: str | None,
    user_name: str | None,
    user_message_id: str,
    component_card: dict,
    received_status: dict,
) -> bool:
    payload = component_card.get("payload")
    if not isinstance(payload, dict):
        return False
    track_payload = payload.get("track")
    if not isinstance(track_payload, dict):
        return False

    from app.models.music import MusicTrackPayload
    from app.services import music
    from app.services.music_chat import render_music_reply
    from app.services.music_status import persist_and_emit_music_status
    from app.services.runtime.tasks import fire_background

    try:
        track = MusicTrackPayload(**track_payload)
    except Exception:
        return False

    status = str((received_status or {}).get("status") or "idle")
    activity = str((received_status or {}).get("activity") or (received_status or {}).get("event") or "处理自己的事")
    accepted = status not in {"sleep", "busy", "very_busy"}
    session_status = "active" if accepted else "pending_agent"
    initiated_by = "user_joined" if accepted else "user_pending"
    current_session = await music.get_open_co_listening(conversation_id=conversation_id)
    agent_was_waiting = (
        current_session is not None
        and current_session.status == "agent_waiting_user"
    )
    already_co_listening = (
        current_session is not None
        and current_session.status == "active"
        and current_session.initiated_by != "user_pending"
        and accepted
    )
    try:
        await music.start_co_listening(
            user_id=user_id,
            agent_id=agent.id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            payload=track,
            initiated_by=initiated_by,
            status=session_status,
            is_playing=True,
        )
    except ValueError:
        logger.warning("music co-listening start skipped: invalid ownership")
        return False

    if status == "sleep":
        prompt_key = "music.sleep_reject"
    elif status in {"busy", "very_busy"}:
        prompt_key = "music.busy_reject"
    elif already_co_listening:
        prompt_key = "music.switch_track"
    else:
        prompt_key = "music.accept_invite"

    try:
        reply = await render_music_reply(
            prompt_key,
            user_name=user_name or "你",
            ai_name=getattr(agent, "name", "") or "我",
            track=track,
            activity=activity,
        )
    except Exception as exc:
        logger.warning("music reply generation failed: %s", exc)
        reply = "我先把这首歌记下，等会儿好好听。"

    metadata = {
        "music_co_listening": accepted,
        "music_prompt_key": prompt_key,
        "reply_index": 0,
    }
    assistant_message_id = await _persist_assistant_message(
        conversation_id,
        reply,
        metadata=metadata,
    )
    try:
        from app.services.chat.post_process import _bg_memory_pipeline

        fire_background(_bg_memory_pipeline(
            user_id,
            [
                {
                    "role": "user",
                    "content": f"我给你推荐了《{track.title}》- {track.artist}，想和你一起听。",
                },
                {"role": "assistant", "content": reply},
            ],
            conversation_id=conversation_id,
            workspace_id=workspace_id,
        ))
    except Exception as memory_err:
        logger.debug(f"[MUSIC] memory pipeline skipped: {memory_err}")

    await ws.send_json({
        "type": "reply",
        "data": {
            "text": reply,
            "assistant_message_id": assistant_message_id,
            "music_co_listening": accepted,
        },
    })
    if not already_co_listening:
        await persist_and_emit_music_status(
            conversation_id=conversation_id,
            status="started",
            track=track,
            actor="user",
        )
    if accepted and not agent_was_waiting and not already_co_listening:
        await persist_and_emit_music_status(
            conversation_id=conversation_id,
            status="started",
            track=track,
            actor="agent",
            actor_name=getattr(agent, "name", "") or "我",
        )
    await ws.send_json({"type": "done", "data": {"message_id": user_message_id}})
    return True


async def _queue_reply(
    ws: WebSocket,
    *,
    conversation_id: str,
    agent,
    user_id: str,
    user_message: str,
    user_message_id: str | None,
    reply_context: dict | None,
    forced_intent=None,
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
            forced_intent=forced_intent,
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
            "forced_intent": getattr(forced_intent, "value", None),
        },
        delay_seconds,
    )
    if delay_seconds > 5 and not appended:
        await ws.send_json({"type": "delay", "data": {"duration": delay_seconds}})
    await ws.send_json({"type": "pending", "data": {"status": "queued", "delay": delay_seconds}})


async def _queue_reply_or_error(
    ws: WebSocket,
    *,
    conversation_id: str,
    agent,
    user_id: str,
    user_message: str,
    user_message_id: str | None,
    reply_context: dict | None,
    forced_intent=None,
) -> None:
    """Queue or stream a reply and surface failures to the active websocket."""
    try:
        await _queue_reply(
            ws,
            conversation_id=conversation_id,
            agent=agent,
            user_id=user_id,
            user_message=user_message,
            user_message_id=user_message_id,
            reply_context=reply_context,
            forced_intent=forced_intent,
        )
    except Exception as e:
        logger.error(f"Chat queue failed for conv={conversation_id[:8]}: {e}")
        await ws.send_json({"type": "error", "data": {"message": "消息入队失败"}})


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
                    # client_id (optional, 但前端推荐传) — 让 ack 事件带回供前端 reconcile.
                    # 不传时 ack 仅含 message_id (DB id), 前端按时间顺序匹配.
                    client_id = payload.get("client_id")
                    raw_component_card = payload.get("component_card")
                    component_card = _sanitize_component_card(raw_component_card)
                    if not text and component_card is None:
                        continue
                    client_id_present = isinstance(client_id, str) and bool(client_id)
                    component_card_type = (
                        raw_component_card.get("type")
                        if isinstance(raw_component_card, dict)
                        else None
                    )
                    logger.debug(
                        "ws message received "
                        f"len={len(text)} client_id_present={client_id_present} "
                        f"component_card_present={raw_component_card is not None} "
                        f"component_card_type={component_card_type} "
                        f"component_card_sanitized={component_card is not None}",
                        extra={
                            "event": EVT_WS_MESSAGE_RECV,
                            "msg_len": len(text),
                            "client_id_present": client_id_present,
                            "component_card_present": raw_component_card is not None,
                            "component_card_type": component_card_type,
                            "component_card_sanitized": component_card is not None,
                        },
                    )
                    await _handle_message(
                        websocket, conversation_id, user_id, agent, text,
                        workspace_id=workspace_id,
                        client_id=client_id,
                        component_card=component_card,
                        user_name=cached_username,
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
            try:
                from app.services.music_status import end_if_disconnected_after_timeout

                fire_background(
                    end_if_disconnected_after_timeout(
                        user_id=user_id,
                        agent_id=agent.id,
                        conversation_id=conversation_id,
                    )
                )
            except Exception as exc:
                logger.debug("music disconnect timeout scheduling skipped: %s", exc)


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
    workspace_id: str | None = None,
    client_id: str | None = None,
    component_card: dict | None = None,
    user_name: str | None = None,
) -> None:
    """处理用户消息：聚合检查 → 生成回复 → 推送。"""
    schedule = await get_cached_schedule(agent.id)
    if not schedule:
        schedule = await generate_daily_schedule(
            agent.id, agent.name, get_mbti(agent), user_id=user_id,
        )
    received_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "type": "leisure", "status": "idle"}
    current_context = await build_reply_timing_context(
        agent_id=agent.id,
        user_id=user_id,
        received_status=received_status,
        user_emotion=quick_emotion_estimate(text),
    )

    plan = await plan_user_message_aggregation(
        agent_id=agent.id,
        user_id=user_id,
        conversation_id=conversation_id,
        text=text,
        reply_context=current_context,
    )
    user_message_id = await _persist_user_message(
        conversation_id,
        text,
        metadata=_message_metadata(
            plan.metadata,
            client_id=client_id,
            component_card=component_card,
        ),
    )
    await _send_ack(ws, message_id=user_message_id, client_id=client_id)
    try:
        from app.services.achievements.service import handle_user_message_event

        fire_background(handle_user_message_event(
            user_id=user_id,
            agent_id=agent.id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            message_id=user_message_id,
            text=text,
            agent_name=getattr(agent, "name", None),
            reply_context=current_context,
            aggregation_route=plan.route,
            component_card=component_card,
        ))
    except Exception as achievement_err:
        logger.debug(f"[ACH] user message hook skipped: {achievement_err}")

    if component_card and component_card.get("type") == "music_track":
        handled = await _handle_music_component_card(
            ws,
            conversation_id=conversation_id,
            user_id=user_id,
            agent=agent,
            workspace_id=workspace_id,
            user_name=user_name,
            user_message_id=user_message_id,
            component_card=component_card,
            received_status=received_status,
        )
        if handled:
            return

    card_reply_message = _component_card_reply_message(plan.final_message, component_card)
    if card_reply_message:
        from app.services.chat.intent_dispatcher import IntentType

        card_context = dict(plan.final_context or {})
        card_context["delay_seconds"] = 0.0
        card_context["component_card_reply"] = True
        card_context["skip_time_memory_lookup"] = True
        await _queue_reply_or_error(
            ws,
            conversation_id=conversation_id,
            agent=agent,
            user_id=user_id,
            user_message=card_reply_message,
            user_message_id=user_message_id,
            reply_context=card_context,
            forced_intent=IntentType.NONE,
        )
        return

    if plan.should_wait:
        pushed = await enqueue_planned_user_message(plan, message_id=user_message_id)
        if pushed:
            await ws.send_json({"type": "pending", "data": {"status": "aggregating"}})
            return
        logger.warning(
            f"[WS] aggregation enqueue failed route={plan.route} "
            f"conv={conversation_id[:8]}, fallback to sync queue"
        )
        await _queue_reply_or_error(
            ws,
            conversation_id=conversation_id,
            agent=agent,
            user_id=user_id,
            user_message=plan.fallback_message,
            user_message_id=user_message_id,
            reply_context=plan.fallback_context,
        )
        return

    if plan.metadata.get("append_delayed"):
        delay_seconds = float((plan.final_context or {}).get("delay_seconds", 0.0) or 0.0)
        await enqueue_or_append_delayed(
            conversation_id,
            {
                "conversation_id": conversation_id,
                "agent_id": agent.id,
                "user_id": user_id,
                "message": plan.final_message,
                "message_id": user_message_id,
                "reply_context": plan.final_context,
            },
            delay_seconds,
        )
        await ws.send_json({
            "type": "pending",
            "data": {"status": "queued", "delay": delay_seconds},
        })
        return

    await _queue_reply_or_error(
        ws,
        conversation_id=conversation_id,
        agent=agent,
        user_id=user_id,
        user_message=plan.final_message,
        user_message_id=user_message_id,
        reply_context=plan.final_context,
    )


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
