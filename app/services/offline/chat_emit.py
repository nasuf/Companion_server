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
