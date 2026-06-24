from __future__ import annotations

from typing import Any

from prisma import Json

from app.db import db
from app.services.proactive.emit import emit_proactive_message
from app.services.runtime.ws_manager import manager


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


async def insert_user_component_message(
    *,
    conversation_id: str | None,
    workspace_id: str | None,
    content: str,
    metadata: dict[str, Any],
) -> str | None:
    if not conversation_id or not content.strip():
        return None
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
            "role": "user",
            "text": content.strip(),
            **metadata,
        },
    )
    return created.id
