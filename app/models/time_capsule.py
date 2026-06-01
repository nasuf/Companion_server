from datetime import date

from pydantic import BaseModel, Field


class TimeCapsuleCreate(BaseModel):
    agent_id: str | None = None
    workspace_id: str | None = None
    title: str | None = None
    content: str = Field(min_length=1, max_length=4000)
    media: dict | None = None
    skin: str = "paper"
    open_date: date | None = None
    status: str = "draft"


class TimeCapsuleUpdate(BaseModel):
    title: str | None = None
    content: str | None = Field(default=None, min_length=1, max_length=4000)
    media: dict | None = None
    skin: str | None = None
    open_date: date | None = None
    status: str | None = None


class TimeCapsuleMediaUpload(BaseModel):
    kind: str
    name: str | None = None
    mime: str | None = None
    size: int | None = None
    duration_seconds: float | None = None
    base64: str = Field(min_length=1)


class TimeCapsuleResponse(BaseModel):
    id: str
    user_id: str
    agent_id: str | None = None
    workspace_id: str | None = None
    title: str | None = None
    content: str
    media: dict | None = None
    skin: str
    open_date: str | None = None
    status: str
    state: str
    sealed_at: str | None = None
    opened_at: str | None = None
    created_at: str
    updated_at: str
