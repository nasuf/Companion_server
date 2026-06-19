from __future__ import annotations

from pydantic import BaseModel


class DailySharePhoto(BaseModel):
    id: str
    message_id: str
    conversation_id: str
    name: str | None = None
    mime: str
    size: int
    width: int | None = None
    height: int | None = None
    url: str
    vision_summary: str | None = None
    created_at: str | None = None


class DailySharePhotoGroup(BaseModel):
    id: str
    title: str
    subtitle: str
    count: int
    photos: list[DailySharePhoto]


class DailySharePhotosResponse(BaseModel):
    total: int
    groups: list[DailySharePhotoGroup]
