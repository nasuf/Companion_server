"""Reminder HTTP endpoints.

业务逻辑在 app.services.reminder.checkin；这里仅保留 HTTP 参数、鉴权依赖和
response_model 绑定，避免路由文件继续膨胀。
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_user
from app.api.ownership import require_user_self
from app.services.reminder import checkin as checkin_service
from app.services.reminder.checkin import (
    DlqItem,
    ReminderCreate,
    ReminderItem,
    RemindersResponse,
    ReminderUpdate,
)

router = APIRouter(prefix="/reminders", tags=["reminders"])


@router.get("", response_model=RemindersResponse)
async def list_reminders(
    user_id: str,
    agent_id: str | None = None,
    status: Literal["active", "fired", "cancelled", "all", "open"] = "all",
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    _user=Depends(require_user_self),
):
    return await checkin_service.list_reminders_for_user(
        user_id=user_id,
        agent_id=agent_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=ReminderItem)
async def create_reminder(
    data: ReminderCreate,
    user: dict = Depends(require_user),
):
    return await checkin_service.create_reminder_for_user(data, user)


@router.patch("/{trigger_id}", response_model=ReminderItem)
async def update_reminder(
    trigger_id: str,
    data: ReminderUpdate,
    user: dict = Depends(require_user),
):
    return await checkin_service.update_reminder_for_user(trigger_id, data, user)


@router.post("/{trigger_id}/complete", response_model=ReminderItem)
async def complete_reminder(
    trigger_id: str,
    conversation_id: str | None = None,
    occurrence_date: str | None = None,
    user: dict = Depends(require_user),
):
    return await checkin_service.complete_reminder_for_user(
        trigger_id,
        conversation_id=conversation_id,
        occurrence_date=occurrence_date,
        user=user,
    )


@router.delete("/{trigger_id}", status_code=204)
async def delete_reminder(
    trigger_id: str,
    conversation_id: str | None = None,
    user: dict = Depends(require_user),
):
    await checkin_service.delete_reminder_for_user(
        trigger_id,
        conversation_id=conversation_id,
        user=user,
    )
    return None


@router.get("/dlq", response_model=list[DlqItem])
async def list_reminder_dlq(
    user_id: str,
    limit: int = Query(default=100, le=500),
    _user=Depends(require_user_self),
):
    return await checkin_service.list_reminder_dlq_for_user(
        user_id=user_id,
        limit=limit,
    )
