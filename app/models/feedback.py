from __future__ import annotations

from pydantic import BaseModel, Field


class UserFeedbackCreateResponse(BaseModel):
    id: str
    created_at: str


class UserFeedbackItem(BaseModel):
    id: str
    user_id: str
    username: str | None = None
    display_name: str | None = None
    content: str
    contact: str
    occurred_at: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    status: str
    app_version: str | None = None
    platform: str | None = None
    created_at: str
    updated_at: str


class UserFeedbackListResponse(BaseModel):
    items: list[UserFeedbackItem]
    total: int


class UserFeedbackStatusUpdate(BaseModel):
    status: str
