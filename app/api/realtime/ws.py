"""WebSocket 聊天端点。

替代 SSE 的持久双向连接，支持用户回合聚合推送和主动消息推送。
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from prisma import Json

from app.db import db
from app.observability import bind_context
from app.observability.events import EVT_WS_CONNECT, EVT_WS_DISCONNECT, EVT_WS_MESSAGE_RECV
from app.redis_client import is_redis_healthy
from app.services.interaction.delayed_queue import (
    clear_reply_inflight,
    enqueue_or_append_delayed,
    mark_reply_inflight,
)
from app.services.interaction.reply_context import build_reply_timing_context
from app.services.interaction.user_turn_aggregation import (
    enqueue_planned_user_message,
    plan_user_message_aggregation,
)
from app.services.schedule_domain.schedule import (
    generate_daily_schedule,
    get_cached_schedule,
    get_current_status,
    get_life_overview,
)
from app.services.mbti import get_mbti
from app.services.notifications.presence import record_ws_online, remove_ws_online
from app.services.proactive.state import mark_user_replied_for_conversation
from app.services.proactive.sender import send_first_greeting
from app.services.relationship.emotion import quick_emotion_estimate
from app.services.runtime.ws_manager import manager
from app.services.runtime.tasks import fire_background
from app.services.chat_media import repo as chat_media_repo
from app.services.chat_media.prompt import render_user_message_with_attachments
from app.services.chat_media.vision import ensure_vision_summaries
from app.services.chat_links import (
    bind_link_card_to_message,
    component_card_for_link,
    create_or_update_link_card,
    extract_first_url,
    extract_link_metadata,
    extract_urls,
    find_link_card,
    metadata_for_link_card,
)
from app.services.chat_links.prompt import render_user_message_with_link
from app.services import wallet
from app.services.vip import chat_quota

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

_IDLE_TIMEOUT = 90.0


def _message_metadata(
    base: dict | None = None,
    *,
    client_id: str | None = None,
    component_card: dict | None = None,
    attachments: list[dict] | None = None,
    link_card: dict | None = None,
) -> dict | None:
    metadata = dict(base or {})
    if client_id:
        metadata["client_id"] = client_id
    if component_card:
        metadata["component_card"] = component_card
    if attachments:
        metadata["attachments"] = attachments
    if link_card:
        metadata["link_card"] = link_card
    return metadata or None


def _sanitize_attachment_ids(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw[:3]:
        if isinstance(item, dict):
            raw_id = item.get("id")
        else:
            raw_id = item
        value = str(raw_id or "").strip()
        if value and len(value) <= 80:
            ids.append(value)
    return list(dict.fromkeys(ids))


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
        "external_link",
        "offline_activity",
        "offline_gift",
        "meal_voucher",
        "red_packet",
        "gift",
    }:
        return None
    payload = _sanitize_component_card_payload(card_type, raw.get("payload"))
    # Red packets are server-issued; a card without offering_id must not
    # enter _handle_message as an empty bubble.
    if card_type in {"red_packet", "gift"} and not payload:
        return None
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
    if card_type == "external_link":
        payload = {}
        for key, limit in (
            ("link_id", 80),
            ("app_url", 2000),
            ("source_url", 2000),
            ("final_url", 2000),
            ("platform", 40),
            ("author", 120),
            ("image_url", 2000),
            ("summary", 1000),
            ("status", 40),
            ("error", 300),
        ):
            value = _truncate_payload_value(raw.get(key), limit).strip()
            if value:
                payload[key] = value
        if not payload.get("link_id") and not payload.get("source_url") and not payload.get("final_url"):
            return None
        return payload
    if card_type in {"offline_activity", "offline_gift"}:
        payload = {}
        for key, limit in (
            ("activity_id", 80),
            ("gift_id", 80),
            ("status", 40),
            ("status_label", 40),
            ("location_name", 160),
            ("gift_name", 160),
            ("image_url", 2000),
            ("real_world_type", 40),
        ):
            value = _truncate_payload_value(raw.get(key), limit).strip()
            if value:
                payload[key] = value
        required_key = "activity_id" if card_type == "offline_activity" else "gift_id"
        if not payload.get(required_key):
            return None
        return payload
    if card_type == "meal_voucher":
        payload = {}
        for key, limit in (
            ("target_tab", 40),
            ("target_section", 80),
            ("fallback_text", 240),
            ("native_status", 40),
            ("campaign_ends_at", 80),
            ("native_message", 300),
        ):
            value = _truncate_payload_value(raw.get(key), limit).strip()
            if value:
                payload[key] = value
        return payload or None
    if card_type == "red_packet":
        offering_id = _truncate_payload_value(raw.get("offering_id"), 80).strip()
        if not offering_id:
            return None
        payload = {"offering_id": offering_id, "kind": "red_packet"}
        for key, limit in (
            ("status", 40),
            ("status_label", 40),
            ("created_at", 80),
            ("received_at", 80),
            ("agent_id", 80),
        ):
            value = _truncate_payload_value(raw.get(key), limit).strip()
            if value:
                payload[key] = value
        amount = _safe_int(raw.get("ticket_amount"), min_value=0, max_value=1_000_000)
        if amount > 0:
            payload["ticket_amount"] = amount
        yuan = _safe_int(raw.get("agent_value_yuan"), min_value=0, max_value=1_000_000)
        if yuan > 0:
            payload["agent_value_yuan"] = yuan
        return payload
    if card_type == "gift":
        offering_id = _truncate_payload_value(raw.get("offering_id"), 80).strip()
        if not offering_id:
            return None
        payload = {"offering_id": offering_id, "kind": "gift"}
        for key, limit in (
            ("status", 40),
            ("status_label", 40),
            ("created_at", 80),
            ("received_at", 80),
            ("agent_id", 80),
            ("product_kind", 80),
            ("product_title", 80),
            ("product_subcategory", 40),
            ("product_asset_key", 40),
        ):
            value = _truncate_payload_value(raw.get(key), limit).strip()
            if value:
                payload[key] = value
        amount = _safe_int(raw.get("ticket_amount"), min_value=0, max_value=1_000_000)
        if amount > 0:
            payload["ticket_amount"] = amount
        yuan = _safe_int(raw.get("agent_value_yuan"), min_value=0, max_value=1_000_000)
        if yuan > 0:
            payload["agent_value_yuan"] = yuan
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


async def _delete_unbound_user_message(message_id: str) -> None:
    """Drop a just-persisted user row that never claimed its red packet."""
    try:
        await db.message.delete(where={"id": message_id})
    except Exception:
        logger.warning("failed to drop unbound red-packet message %s", message_id[:8])


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
    from app.services.music_status import persist_and_emit_music_status, reconcile_co_listening_for_status
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
    user_was_joined = (
        current_session is not None
        and current_session.status in {"active", "pending_agent"}
        and current_session.initiated_by not in {"agent", "agent_auto"}
    )
    agent_was_present = (
        current_session is not None
        and current_session.status in {"active", "agent_waiting_user"}
        and current_session.initiated_by != "user_pending"
    )
    agent_join_was_announced = (
        current_session is not None
        and agent_was_present
        and current_session.initiated_by != "agent_auto"
    )
    already_co_listening = (
        current_session is not None
        and current_session.status == "active"
        and user_was_joined
        and agent_was_present
        and accepted
    )
    previously_joined = (
        current_session is not None
        and current_session.status in {"active", "agent_waiting_user"}
        and current_session.initiated_by
        not in {"agent", "agent_auto", "user_pending"}
        and agent_was_present
    )
    if previously_joined and not accepted:
        await reconcile_co_listening_for_status(
            user_id=user_id,
            agent_id=agent.id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            status_code=status,
            activity=activity,
            ai_name=getattr(agent, "name", "") or "我",
            user_name=user_name or "你",
        )
        await ws.send_json({"type": "done", "data": {"message_id": user_message_id}})
        return True
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

    # trace 包装 (测试手册 §4 缺口): 共听邀请响应带 Trace 按钮,
    # music.accept_invite / busy_reject / sleep_reject / switch_track 可面板调试.
    from app.services.llm.usage_tracker import traced_usage_session

    async with traced_usage_session(
        name=f"[music:{prompt_key}]", scope="music",
        conversation_id=conversation_id,
        agent_id=getattr(agent, "id", None), user_id=user_id,
    ) as tracer:
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
        if tracer.safe_trace_id:
            metadata["trace_id"] = tracer.safe_trace_id
        assistant_message_id = await _persist_assistant_message(
            conversation_id,
            reply,
            metadata=metadata,
        )
    try:
        from app.services.chat.post_process import _bg_memory_pipeline
        from app.services.notifications.service import notify_agent_message_created

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
        fire_background(notify_agent_message_created(
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            text=reply,
            metadata=metadata,
            user_id=user_id,
            agent_id=agent.id,
            workspace_id=workspace_id,
            agent_name=getattr(agent, "name", None),
        ))
    except Exception as memory_err:
        logger.debug(f"[MUSIC] background hooks skipped: {memory_err}")

    await ws.send_json({
        "type": "reply",
        "data": {
            "text": reply,
            "assistant_message_id": assistant_message_id,
            "music_co_listening": accepted,
        },
    })
    if not user_was_joined:
        await persist_and_emit_music_status(
            conversation_id=conversation_id,
            status="started",
            track=track,
            actor="user",
        )
    if accepted and not agent_join_was_announced:
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
        # Mark generation in-flight so a message arriving mid-reply coalesces
        # into the delayed queue instead of racing a duplicate parallel reply.
        await mark_reply_inflight(conversation_id)
        try:
            await stream_to_ws(gen, conversation_id)
        finally:
            await clear_reply_inflight(conversation_id)
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


async def _warm_daily_schedule(agent, user_id: str) -> None:
    """连接后台把当日作息生成好, 让消息路径大概率直接命中缓存.

    整条链路全程吞异常: 预热失败只是回到「消息路径现场生成」这个原有行为, 不该
    影响连接本身。
    """
    try:
        if await get_cached_schedule(agent.id):
            return
        await generate_daily_schedule(
            agent.id, agent.name, get_mbti(agent), user_id=user_id,
            life_overview=await get_life_overview(agent.id),
        )
    except Exception as e:
        logger.warning(f"schedule warmup failed agent={str(agent.id)[:8]}: {e}")


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
    client_supports_voice = (
        websocket.query_params.get("client", "").strip().lower() == "flutter"
    )
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
        # 实时在线 (连接数语义): 每条 WS 用唯一 conn_id 入 WS 池, 断开时摘除 →
        # 离开聊天页瞬时下线. App / H5 同一路径, 一视同仁.
        ws_conn_id = uuid.uuid4().hex
        await record_ws_online(user_id, ws_conn_id)
        logger.info("ws connected", extra={"event": EVT_WS_CONNECT})

        # spec §12 开场主动第一句话: 只在首次进入 (0 消息) 时触发
        try:
            asyncio.create_task(
                send_first_greeting(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=agent.id,
                    workspace_id=workspace_id,
                    voice_eligible=client_supports_voice,
                )
            )
        except Exception as e:
            logger.warning(f"first_greeting dispatch failed conv={conversation_id[:8]}: {e}")

        # 预热当日作息。带 LLM 的夜间任务只覆盖近 7 天活跃的 agent (见
        # scheduler.LLM_CRON_ACTIVE_WINDOW_DAYS), 休眠用户回来时缓存必然是空的,
        # 而消息处理路径上的生成是 await 的 —— 会让"久违的第一句话"多等约 6 秒。
        # 放在连接后台跑, 用户打字的时间足够覆盖掉它; 消息路径的 await 保留作兜底。
        fire_background(_warm_daily_schedule(agent, user_id))

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

                # 任意帧 (ping/消息) 都续期在线状态 (前端每 25s ping, TTL 90s 兜底).
                await record_ws_online(user_id, ws_conn_id)

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
                    attachment_ids = _sanitize_attachment_ids(payload.get("attachments"))
                    paid_confirmed = bool(payload.get("paid_confirmed", False))
                    if not text and component_card is None and not attachment_ids:
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
                        attachment_ids=attachment_ids,
                        user_name=cached_username,
                        client_supports_voice=client_supports_voice,
                        paid_confirmed=paid_confirmed,
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
            # 立即摘除该连接 → 实时在线瞬时反映离开.
            await remove_ws_online(user_id, ws_conn_id)
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


async def _bind_offering_and_queue_reply(
    ws: WebSocket,
    *,
    conversation_id: str,
    agent,
    user_id: str,
    client_id: str | None,
    offering: dict,
    user_message_id: str,
    plan,
    current_context: dict,
    context_key: str,
    error_message: str,
    get_existing,
) -> None:
    """Bind a sent offering to this user message, then queue the companion reply.

    Bind is the exclusive claim. Ack only after it succeeds so a losing
    concurrent persist is not marked delivered, then deleted.
    """
    from app.services import offerings as offerings_svc
    from app.services.chat.intent_dispatcher import IntentType

    try:
        bound = await offerings_svc.bind_offering_message(
            offering_id=str(offering["id"]),
            message_id=user_message_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except ValueError:
        await _delete_unbound_user_message(user_message_id)
        existing_id = None
        try:
            existing = await get_existing(
                offering_id=str(offering["id"]),
                user_id=user_id,
            )
            existing_id = (existing.get("offering") or {}).get("message_id")
        except ValueError:
            existing_id = None
        if existing_id:
            await _send_ack(ws, message_id=str(existing_id), client_id=client_id)
            return
        await ws.send_json({"type": "error", "data": {"message": error_message}})
        return
    await _send_ack(ws, message_id=user_message_id, client_id=client_id)
    reply_message = await offerings_svc.build_offering_user_message(bound)
    card_context = dict(plan.final_context or current_context)
    card_context["delay_seconds"] = 0.0
    card_context["component_card_reply"] = True
    card_context["skip_time_memory_lookup"] = True
    card_context[context_key] = offerings_svc.reply_context_payload(bound)
    await _queue_reply_or_error(
        ws,
        conversation_id=conversation_id,
        agent=agent,
        user_id=user_id,
        user_message=reply_message,
        user_message_id=user_message_id,
        reply_context=card_context,
        forced_intent=IntentType.NONE,
    )


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


async def _ensure_link_card_for_turn(
    *,
    user_id: str,
    conversation_id: str,
    text: str,
    component_card: dict | None,
) -> tuple[dict | None, dict | None]:
    """Return (component_card, model-visible link metadata) for this user turn."""
    link = None
    if component_card and component_card.get("type") == "external_link":
        payload = component_card.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        link_id = str(payload.get("link_id") or "").strip()
        if link_id:
            link = await find_link_card(
                link_id=link_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if link and _link_card_needs_refresh(link):
                source_url = str(link.source_url or link.final_url or "").strip()
                shared_text = "\n".join(
                    part
                    for part in (
                        text,
                        str(component_card.get("title") or ""),
                        str(component_card.get("body") or ""),
                        source_url,
                    )
                    if part.strip()
                )
                if source_url or extract_first_url(shared_text):
                    metadata = await extract_link_metadata(
                        url=source_url or None,
                        shared_text=shared_text,
                    )
                    link = await create_or_update_link_card(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        metadata=metadata,
                        role="user",
                        source_app="component_card_refresh",
                        extra_metadata={"refreshed_from_link_id": link.id},
                    )
        if link is None:
            source_url = str(payload.get("source_url") or payload.get("final_url") or "").strip()
            shared_text = "\n".join(
                part
                for part in (
                    text,
                    str(component_card.get("title") or ""),
                    str(component_card.get("body") or ""),
                    source_url,
                )
                if part.strip()
            )
            if source_url or extract_first_url(shared_text):
                metadata = await extract_link_metadata(
                    url=source_url or None,
                    shared_text=shared_text,
                )
                link = await create_or_update_link_card(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    metadata=metadata,
                    role="user",
                    source_app=str(payload.get("source_app") or "component_card"),
                )
    elif extract_first_url(text):
        metadata = await extract_link_metadata(url=None, shared_text=text)
        link = await create_or_update_link_card(
            user_id=user_id,
            conversation_id=conversation_id,
            metadata=metadata,
            role="user",
            source_app="chat_text",
        )

    if link is None:
        return component_card, None
    return component_card_for_link(link), metadata_for_link_card(link)


def _link_card_needs_refresh(link) -> bool:
    if str(getattr(link, "platform", "") or "") != "微博":
        return False
    fields = (
        getattr(link, "title", ""),
        getattr(link, "description", ""),
        getattr(link, "summary", ""),
        getattr(link, "content_text", ""),
    )
    text = " ".join(str(field or "") for field in fields)
    if "Sina Visitor System" in text:
        return True
    body = " ".join(str(field or "") for field in fields[1:]).strip()
    url_count = len(extract_urls(body))
    return url_count > 0 and len(body.replace(" ", "")) <= 120 + url_count * 80


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
    attachment_ids: list[str] | None = None,
    user_name: str | None = None,
    client_supports_voice: bool = False,
    paid_confirmed: bool = False,
) -> None:
    """处理用户消息：额度闸门 → 聚合检查 → 生成回复 → 推送。"""
    # CLAUDE.md 权益项 1 只约束"发消息聊天"这件事。红包/礼物走的是自己的
    # 钞票/积分支付流程 (offerings.py)，跟对话额度是两套互不相关的计费口径
    # —— 不豁免的话，一个聊天额度用尽、没钞票的用户会连"花钞票发红包"这个
    # 本该独立于聊天额度的动作都发不出去，等于用一个功能的限制卡死另一个。
    is_red_packet_or_gift = (
        isinstance(component_card, dict)
        and component_card.get("type") in {"red_packet", "gift"}
    )
    if not is_red_packet_or_gift:
        is_vip = await wallet.is_vip(user_id)
        quota_result = await chat_quota.consume_one(
            user_id, is_vip=is_vip, paid_confirmed=paid_confirmed
        )
        if not quota_result["allowed"]:
            # 未确认付费 / 余额不足 —— 消息不入库、不计数、不扣费，前端据此
            # 弹"是否继续扣费"或"是否订阅VIP"，取消则文本保留。
            await ws.send_json({
                "type": "quota_blocked",
                "data": {
                    "reason": quota_result["reason"],
                    "per_msg_cost": quota_result["per_msg_cost"],
                    "spendable_tickets": quota_result["spendable_tickets"],
                    # 带上 client_id 让前端能精确摘掉被拒的那条草稿, 而不是
                    # "摘最后一条待发消息" —— 用户手快连发两条时后者会摘错。
                    "client_id": client_id,
                },
            })
            return
    attachments = await chat_media_repo.get_message_attachments(
        attachment_ids=attachment_ids or [],
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if attachment_ids and len(attachments) != len(set(attachment_ids)):
        await ws.send_json({"type": "error", "data": {"message": "附件无效或已发送"}})
        return
    red_packet_offering = None
    gift_offering = None
    if component_card and component_card.get("type") in {"red_packet", "gift"}:
        from app.services import offerings as offerings_svc

        card_type = component_card.get("type")
        try:
            if card_type == "gift":
                component_card = await offerings_svc.authorize_gift_card(
                    component_card,
                    user_id=user_id,
                    agent_id=agent.id,
                    conversation_id=conversation_id,
                )
            else:
                component_card = await offerings_svc.authorize_red_packet_card(
                    component_card,
                    user_id=user_id,
                    agent_id=agent.id,
                    conversation_id=conversation_id,
                )
        except ValueError:
            message = "礼物无效或已发送" if card_type == "gift" else "红包无效或已发送"
            await ws.send_json({"type": "error", "data": {"message": message}})
            return
        if isinstance(component_card, dict):
            bound_offering = component_card.pop("_offering", None)
            if card_type == "gift":
                gift_offering = bound_offering
            else:
                red_packet_offering = bound_offering
    # Attachments are analysed before the chat turn opens its usage session,
    # so wrap them in their own one — otherwise the vision tokens are billed
    # by Ark but invisible in the admin cost dashboard.
    from app.services.llm.usage_tracker import usage_session

    async with usage_session(
        scope="chat_media",
        conversation_id=conversation_id,
        agent_id=agent.id,
        user_id=user_id,
    ):
        attachment_metadata = await ensure_vision_summaries(
            attachments,
            user_text=text,
        )
    prompt_text = render_user_message_with_attachments(text, attachment_metadata)
    component_card, link_card_metadata = await _ensure_link_card_for_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        text=text,
        component_card=component_card,
    )
    prompt_text = render_user_message_with_link(prompt_text, link_card_metadata)

    schedule = await get_cached_schedule(agent.id)
    if not schedule:
        # 必须带上生活画像。不带的话 generate_daily_schedule 走的是通用模板分支,
        # 生成一份跟这个 agent 的职业/性格无关的作息 —— 而缓存未命中恰恰发生在
        # 「久未上线的用户回来说第一句话」这种时刻, 那天的 AI 会显得完全换了个人。
        schedule = await generate_daily_schedule(
            agent.id, agent.name, get_mbti(agent), user_id=user_id,
            life_overview=await get_life_overview(agent.id),
        )
    received_status = get_current_status(schedule) if schedule else {"activity": "自由时间", "type": "leisure", "status": "idle"}
    current_context = await build_reply_timing_context(
        agent_id=agent.id,
        user_id=user_id,
        received_status=received_status,
        user_emotion=quick_emotion_estimate(prompt_text),
    )
    current_context["client_supports_voice"] = client_supports_voice

    if red_packet_offering or gift_offering:
        from app.services.interaction.user_turn_aggregation import (
            UserMessageAggregationPlan,
        )

        # Empty bubble + card would otherwise look like a 0-char fragment.
        offering_meta = {"queued": True}
        if red_packet_offering:
            offering_meta["red_packet"] = True
        if gift_offering:
            offering_meta["gift"] = True
        plan = UserMessageAggregationPlan(
            route="immediate",
            agent_id=agent.id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            metadata=offering_meta,
            final_message=text,
            final_context=current_context,
            fallback_message=text,
            fallback_context=current_context,
        )
    else:
        plan = await plan_user_message_aggregation(
            agent_id=agent.id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=prompt_text,
            reply_context=current_context,
        )
    user_message_id = await _persist_user_message(
        conversation_id,
        text,
        metadata=_message_metadata(
            plan.metadata,
            client_id=client_id,
            component_card=component_card,
            attachments=attachment_metadata,
            link_card=link_card_metadata,
        ),
    )
    await chat_media_repo.bind_attachments_to_message(
        attachment_ids=[item.id for item in attachments],
        message_id=user_message_id,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if link_card_metadata and link_card_metadata.get("id"):
        await bind_link_card_to_message(
            link_id=str(link_card_metadata["id"]),
            message_id=user_message_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    if red_packet_offering:
        from app.services import offerings as offerings_svc

        await _bind_offering_and_queue_reply(
            ws,
            conversation_id=conversation_id,
            agent=agent,
            user_id=user_id,
            client_id=client_id,
            offering=red_packet_offering,
            user_message_id=user_message_id,
            plan=plan,
            current_context=current_context,
            context_key="red_packet",
            error_message="红包无效或已发送",
            get_existing=offerings_svc.get_red_packet,
        )
        return
    if gift_offering:
        from app.services import offerings as offerings_svc

        await _bind_offering_and_queue_reply(
            ws,
            conversation_id=conversation_id,
            agent=agent,
            user_id=user_id,
            client_id=client_id,
            offering=gift_offering,
            user_message_id=user_message_id,
            plan=plan,
            current_context=current_context,
            context_key="gift",
            error_message="礼物无效或已发送",
            get_existing=offerings_svc.get_gift,
        )
        return
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
