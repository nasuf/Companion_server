from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_user
from app.services.notifications.devices import disable_device_for_user, register_device
from app.services.notifications.presence import update_presence
from app.services.notifications.service import enqueue_notification

router = APIRouter(prefix="/notifications", tags=["notifications"])


class PushDeviceRegister(BaseModel):
    platform: str = Field(pattern="^ios$")
    token: str = Field(min_length=8)
    environment: str = "sandbox"
    bundle_id: str | None = None
    device_id: str | None = None
    app_version: str | None = None


class PushDeviceDisable(BaseModel):
    token: str = Field(min_length=8)


class PushPresenceUpdate(BaseModel):
    device_id: str = Field(min_length=1, max_length=120)
    foreground: bool
    workspace_id: str | None = None
    conversation_id: str | None = None


class SystemNotificationCreate(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    body: str = Field(min_length=1, max_length=160)
    type: str = Field(default="system_custom", max_length=80)
    payload: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str | None = Field(default=None, max_length=160)
    scheduled_for: datetime | None = None
    agent_id: str | None = None
    workspace_id: str | None = None
    conversation_id: str | None = None


@router.post("/devices")
async def register_push_device(
    data: PushDeviceRegister,
    user: dict = Depends(require_user),
):
    try:
        device_id = await register_device(
            user_id=user["sub"],
            platform=data.platform,
            token=data.token,
            environment=data.environment,
            bundle_id=data.bundle_id,
            device_id=data.device_id,
            app_version=data.app_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "device_id": device_id}


@router.delete("/devices")
async def disable_push_device(
    data: PushDeviceDisable,
    user: dict = Depends(require_user),
):
    await disable_device_for_user(user_id=user["sub"], token=data.token)
    return {"ok": True}


@router.post("/presence")
async def update_push_presence(
    data: PushPresenceUpdate,
    user: dict = Depends(require_user),
):
    await update_presence(
        user_id=user["sub"],
        device_id=data.device_id,
        foreground=data.foreground,
        workspace_id=data.workspace_id,
        conversation_id=data.conversation_id,
    )
    return {"ok": True}


@router.post("/system")
async def create_system_notification(
    data: SystemNotificationCreate,
    user: dict = Depends(require_user),
):
    if data.type in {"agent_message", "achievement_unlocked", "capsule_ready"}:
        raise HTTPException(status_code=400, detail="Reserved notification type")
    dedupe_key = data.dedupe_key or f"system:{user['sub']}:{datetime.utcnow().timestamp()}"
    event_id = await enqueue_notification(
        user_id=user["sub"],
        agent_id=data.agent_id,
        workspace_id=data.workspace_id,
        conversation_id=data.conversation_id,
        type=data.type or "system_custom",
        title=data.title,
        body=data.body,
        payload={"type": data.type or "system_custom", **data.payload},
        dedupe_key=dedupe_key,
        scheduled_for=data.scheduled_for,
        dispatch_now=data.scheduled_for is None,
    )
    return {"ok": True, "notification_id": event_id}
