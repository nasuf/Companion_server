from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

from prisma import Json

from app.db import db
from app.services.offline.gift_status import MYSTERY_GIFT_NAME, is_revealed, status_label as gift_status_label
from app.services.proactive.emit import emit_proactive_message
from app.services.runtime.ws_manager import manager


@asynccontextmanager
async def offline_trace(
    kind: str,
    *,
    conversation_id: str | None,
    agent_id: str | None,
    user_id: str | None,
):
    """offline 礼物/活动消息的 trace 包装（对齐 proactive/sender 模式）。

    包住「LLM 生成 + 发射」整段：LLM run 进 LangSmith trace、prompt 渲染
    组件被记录、emit 时把 tracer.safe_trace_id 挂进消息 metadata → 前端
    Trace 按钮可点，运维可在面板内调试 offline.* prompt（测试手册 §4 缺口）。
    """
    from app.services.llm.usage_tracker import traced_usage_session

    async with traced_usage_session(
        name=f"[offline:{kind}]", scope="offline",
        conversation_id=conversation_id, agent_id=agent_id, user_id=user_id,
    ) as tracer:
        yield tracer


async def emit_assistant(
    *,
    conversation_id: str | None,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    message: str,
    real_world_type: str,
    source_id: str,
    trigger_type: str,
    extra_metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
    guard_delivery_key: str | None = None,
    arrival_delivery: tuple[str, int] | None = None,
) -> str | None:
    if not conversation_id or not message.strip():
        return None
    metadata = {
        "real_world_type": real_world_type,
        "source_id": source_id,
        **(extra_metadata or {}),
    }
    message_id = await emit_proactive_message(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        message=message.strip(),
        trigger_type=trigger_type,
        extra_metadata=metadata,
        trace_id=trace_id,
        **({'ws_payload_extra': {'human_takeover': True}} if metadata.get('human_takeover') else {}),
        voice_eligible=guard_delivery_key is None,
        guard_activity_id=source_id if guard_delivery_key else None,
        guard_delivery_key=guard_delivery_key,
        **({"arrival_delivery": arrival_delivery} if arrival_delivery else {}),
    )
    await _notify_takeover_transcript(conversation_id, message_id)
    return message_id


def build_activity_component_card(activity: dict[str, Any], *, status_label: str) -> dict[str, Any]:
    location = (
        activity.get("location_name")
        or activity.get("address")
        or activity.get("city")
        or "线下活动"
    )
    image_urls = activity.get("image_urls") or []
    image_url = image_urls[0] if isinstance(image_urls, list) and image_urls else None
    return {
        "version": 1,
        "type": "offline_activity",
        "title": activity.get("title") or "线下活动",
        "subtitle": f"{status_label} · {location}",
        "body": activity.get("summary") or activity.get("description") or "",
        "footer": "点击查看活动详情",
        "accent": "#5EC7DE",
        "payload": {
            "activity_id": activity.get("id"),
            "status": activity.get("status"),
            "status_label": status_label,
            "location_name": location,
            "image_url": image_url,
            "image_urls": image_urls,
            "kind": activity.get('kind', 'place'),
            "schedule_label": activity.get('schedule_label'),
            "real_world_type": "activity",
        },
    }


def single_bubble_text(text: str | None) -> str:
    """卡片正文是一整段：模型按聊天习惯输出的 || 分句与系统标记都要清掉。"""
    from app.services.chat.reply_formatting import strip_system_markers

    value = (text or "").strip()
    if "||" in value:
        value = " ".join(part.strip() for part in value.split("||") if part.strip())
    return (strip_system_markers(value) or "").strip()


def build_gift_component_card(
    gift: dict[str, Any],
    *,
    status_label: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """礼物聊天卡片。卡片随消息永久落库，所以签收前就不能写入礼物名称/图片/寄语。"""
    status_label = status_label or gift_status_label(gift.get("status"))
    message = single_bubble_text(message) or None
    revealed = is_revealed(gift)
    subtitle_parts = [status_label]
    if gift.get("logistics_provider") and gift.get("logistics_provider") != "mock":
        subtitle_parts.append(str(gift["logistics_provider"]))
    elif gift.get("tracking_number"):
        subtitle_parts.append("物流更新中")
    if revealed:
        title = gift.get("gift_name") or "一份小礼物"
        image_url = gift.get("product_image_url") or gift.get("image_url") or gift.get("cover_url")
        body = message or gift.get("gift_note") or gift.get("gift_reason") or "这份小心意已经送到你手上了。"
        footer = "点击查看礼物详情"
    else:
        title = MYSTERY_GIFT_NAME
        image_url = None
        body = message or "有一份神秘礼物正在路上，签收后揭晓。"
        footer = "点击查看物流"
    return {
        "version": 1,
        "type": "offline_gift",
        "title": title,
        "subtitle": " · ".join(part for part in subtitle_parts if part),
        "body": body,
        "footer": footer,
        "accent": "#F6A64B",
        "payload": {
            "gift_id": gift.get("id"),
            "status": gift.get("status"),
            "status_label": status_label,
            "gift_name": title if revealed else None,
            "image_url": image_url,
            "revealed": revealed,
            "real_world_type": "gift",
        },
    }


def build_gift_user_card(gift: dict[str, Any], *, kind: str, status_label: str, body: str) -> dict[str, Any]:
    """用户发给 AI 的礼物卡片（签收 / 道谢）。只在签收后出现，封面是礼物实物图。"""
    return {
        "version": 1,
        "type": "offline_gift",
        "title": gift.get("gift_name") or "一份小礼物",
        "subtitle": status_label,
        "body": single_bubble_text(body),
        "footer": "点击查看礼物详情",
        "accent": "#F6A64B",
        "payload": {
            "gift_id": gift.get("id"),
            "status": gift.get("status"),
            "status_label": status_label,
            "gift_name": gift.get("gift_name"),
            "image_url": gift.get("product_image_url"),
            "revealed": True,
            "kind": kind,
            "real_world_type": "gift",
        },
    }


async def emit_activity_card(
    *,
    conversation_id: str | None,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    activity: dict[str, Any],
    trigger_type: str,
    status_label: str,
    extra_metadata: dict[str, Any] | None = None,
) -> str | None:
    if not conversation_id:
        return None
    card = build_activity_component_card(activity, status_label=status_label)
    metadata = {
        **(extra_metadata or {}),
        "real_world_type": "activity",
        "source_id": activity.get("id"),
        "component_card": card,
    }
    message_id = await emit_proactive_message(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        message=f"活动卡片：{activity.get('title') or '线下活动'}",
        trigger_type=trigger_type,
        extra_metadata=metadata,
        ws_payload_extra={"component_card": card, **(extra_metadata or {})},
    )
    await _notify_takeover_transcript(conversation_id, message_id)
    return message_id


async def emit_thought_fragment(
    *,
    conversation_id: str | None,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    activity_id: str,
    fragment: dict[str, Any],
    source_message_id: str | None,
    trace_id: str | None = None,
) -> str | None:
    """识图命中产出碎片后推「思绪气泡」。metadata.offline_fragment 让前端定位触发照片
    并按等级渲染；skip_post_process + voice 关闭，避免被当普通回复二次加工/朗读。"""
    if not conversation_id:
        return None
    fragment_meta = {
        "fragment_id": fragment.get("id"),
        "tier": fragment.get("tier"),
        "lead_in": fragment.get("lead_in") or "",
        "source_message_id": source_message_id,
    }
    message_id = await emit_proactive_message(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        message=fragment.get("text") or "",
        trigger_type="offline_thought_fragment",
        extra_metadata={
            "real_world_type": "activity",
            "source_id": activity_id,
            "offline_fragment": fragment_meta,
        },
        ws_payload_extra={"offline_fragment": fragment_meta},
        skip_post_process=True,
        voice_eligible=False,
        trace_id=trace_id,
    )
    await _notify_takeover_transcript(conversation_id, message_id)
    return message_id


async def emit_gift_card(
    *,
    conversation_id: str | None,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
    trigger_type: str,
    status_label: str | None = None,
    message: str | None = None,
    trace_id: str | None = None,
) -> str | None:
    if not conversation_id:
        return None
    message = single_bubble_text(message) or None
    card = build_gift_component_card(gift, status_label=status_label, message=message)
    metadata = {
        "real_world_type": "gift",
        "source_id": gift.get("id"),
        "component_card": card,
    }
    return await emit_proactive_message(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        message=message or f"礼物卡片：{card['title']}",
        trigger_type=trigger_type,
        extra_metadata=metadata,
        ws_payload_extra={"component_card": card},
        trace_id=trace_id,
    )


async def insert_user_activity_card(
    *,
    conversation_id: str | None,
    workspace_id: str | None,
    activity: dict[str, Any],
    trigger_type: str,
    status_label: str,
    arrival_delivery: tuple[str, int] | None = None,
) -> str | None:
    if not conversation_id:
        return None
    card = build_activity_component_card(activity, status_label=status_label)
    return await insert_user_component_message(
        conversation_id=conversation_id,
        workspace_id=workspace_id,
        **({"arrival_delivery": arrival_delivery} if arrival_delivery else {}),
        content=f"活动卡片：{activity.get('title') or '线下活动'}",
        metadata={
            "real_world_type": "activity",
            "source_id": activity.get("id"),
            "trigger_type": trigger_type,
            "component_card": card,
        },
    )


async def insert_user_component_message(
    *,
    conversation_id: str | None,
    workspace_id: str | None,
    content: str,
    metadata: dict[str, Any],
    client_id: str | None = None,
    arrival_delivery: tuple[str, int] | None = None,
) -> str | None:
    if not conversation_id or not content.strip():
        return None
    if client_id:
        metadata = {**metadata, "client_id": client_id}
    if arrival_delivery:
        from app.services.offline.arrival_followups import persist_delivery
        row = await persist_delivery(activity_id=arrival_delivery[0], generation=arrival_delivery[1],
            role='user', conversation_id=conversation_id, content=content.strip(), metadata=metadata)
        if row is None:
            return None
        created_id, created_at = row['id'], row['created_at']
        content, metadata = row['content'], row['metadata']
    else:
        created = await db.message.create(
            data={
                "conversation": {"connect": {"id": conversation_id}},
                "role": "user",
                "content": content.strip(),
                "metadata": Json(metadata),
            }
        )
        created_id, created_at = created.id, created.createdAt
    await manager.send_to_workspace(
        workspace_id,
        "message",
        {
            "message_id": created_id,
            "conversation_id": conversation_id,
            "role": "user",
            "text": content.strip(),
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
            **metadata,
        },
    )
    await _notify_takeover_transcript(conversation_id, created_id)
    return created_id


async def _notify_takeover_transcript(conversation_id: str | None, message_id: str | None) -> None:
    if not conversation_id or not message_id:
        return
    try:
        from app.services.admin_takeover.service import notify_takeover_transcript

        await notify_takeover_transcript(conversation_id, message_id)
    except Exception as exc:
        logger.warning("takeover activity transcript notify failed: %s", type(exc).__name__)
