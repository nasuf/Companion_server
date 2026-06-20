from __future__ import annotations

from pydantic import BaseModel, Field


class ChatLinkAnalyzeRequest(BaseModel):
    conversation_id: str
    url: str | None = None
    shared_text: str | None = Field(default=None, validation_alias="shared_text")
    source_app: str | None = Field(default=None, validation_alias="source_app")


class ChatLinkCardResponse(BaseModel):
    id: str
    conversation_id: str
    message_id: str | None = None
    role: str = "user"
    source_app: str | None = None
    source_url: str
    final_url: str
    platform: str
    title: str
    description: str = ""
    author: str | None = None
    image_url: str | None = None
    content_text: str = ""
    summary: str = ""
    status: str = "ready"
    error: str | None = None
    created_at: str | None = None
    component_card: dict


class DailyShareLink(BaseModel):
    id: str
    message_id: str | None = None
    conversation_id: str
    role: str
    source_app: str | None = None
    source_url: str
    final_url: str
    platform: str
    title: str
    description: str = ""
    author: str | None = None
    image_url: str | None = None
    summary: str = ""
    created_at: str | None = None
    component_card: dict


class DailyShareLinkGroup(BaseModel):
    id: str
    title: str
    subtitle: str
    count: int
    links: list[DailyShareLink]


class DailyShareLinksResponse(BaseModel):
    total: int
    groups: list[DailyShareLinkGroup]
