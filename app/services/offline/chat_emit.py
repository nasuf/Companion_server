from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from prisma import Json

from app.db import db
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
) -> str | None:
    if not conversation_id or not message.strip():
        return None
    metadata = {
        "real_world_type": real_world_type,
        "source_id": source_id,
        **(extra_metadata or {}),
    }
    return await emit_proactive_message(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        message=message.strip(),
        trigger_type=trigger_type,
        extra_metadata=metadata,
        trace_id=trace_id,
    )


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
            "real_world_type": "activity",
        },
    }


def build_gift_component_card(
    gift: dict[str, Any],
    *,
    status_label: str,
    message: str | None = None,
) -> dict[str, Any]:
    image_url = (
        gift.get("product_image_url")
        or gift.get("image_url")
        or gift.get("cover_url")
    )
    subtitle_parts = [status_label]
    if gift.get("logistics_provider"):
        subtitle_parts.append(str(gift["logistics_provider"]))
    elif gift.get("tracking_number"):
        subtitle_parts.append("物流更新中")
    body = (
        message
        or gift.get("gift_note")
        or gift.get("gift_reason")
        or "有一份小心意正在路上。"
    )
    return {
        "version": 1,
        "type": "offline_gift",
        "title": gift.get("gift_name") or "一份小礼物",
        "subtitle": " · ".join(part for part in subtitle_parts if part),
        "body": body,
        "footer": "点击查看礼物详情",
        "accent": "#F6A64B",
        "payload": {
            "gift_id": gift.get("id"),
            "status": gift.get("status"),
            "status_label": status_label,
            "gift_name": gift.get("gift_name"),
            "image_url": image_url,
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
) -> str | None:
    if not conversation_id:
        return None
    card = build_activity_component_card(activity, status_label=status_label)
    metadata = {
        "real_world_type": "activity",
        "source_id": activity.get("id"),
        "component_card": card,
    }
    return await emit_proactive_message(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        message=f"活动卡片：{activity.get('title') or '线下活动'}",
        trigger_type=trigger_type,
        extra_metadata=metadata,
        ws_payload_extra={"component_card": card},
    )


async def emit_gift_card(
    *,
    conversation_id: str | None,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    gift: dict[str, Any],
    trigger_type: str,
    status_label: str,
    message: str | None = None,
    trace_id: str | None = None,
) -> str | None:
    if not conversation_id:
        return None
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
        message=message or f"礼物卡片：{gift.get('gift_name') or '一份小礼物'}",
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
) -> str | None:
    if not conversation_id:
        return None
    card = build_activity_component_card(activity, status_label=status_label)
    return await insert_user_component_message(
        conversation_id=conversation_id,
        workspace_id=workspace_id,
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
) -> str | None:
    if not conversation_id or not content.strip():
        return None
    if client_id:
        metadata = {**metadata, "client_id": client_id}
    created = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "user",
            "content": content.strip(),
            "metadata": Json(metadata),
        }
    )
    await manager.send_to_workspace(
        workspace_id,
        "message",
        {
            "message_id": created.id,
            "conversation_id": conversation_id,
            "role": "user",
            "text": content.strip(),
            "created_at": created.createdAt.isoformat()
            if getattr(created, "createdAt", None)
            else None,
            **metadata,
        },
    )
    return created.id
