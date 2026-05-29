from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class LastWillContact(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    email: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=40)

    @field_validator("name", "email", "phone", mode="before")
    @classmethod
    def _strip_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _require_channel(self) -> "LastWillContact":
        if not (self.email or self.phone):
            raise ValueError("联系人需要填写邮箱或电话")
        return self


class LastWillCreate(BaseModel):
    agent_id: str
    workspace_id: str | None = None
    content: str = Field(default="", max_length=8000)
    inactivity_days: int = Field(default=30, ge=5, le=365)
    contacts: list[LastWillContact] = Field(default_factory=list, max_length=3)
    status: str = "draft"

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        return value.strip()


class LastWillUpdate(BaseModel):
    content: str | None = Field(default=None, max_length=8000)
    inactivity_days: int | None = Field(default=None, ge=5, le=365)
    contacts: list[LastWillContact] | None = Field(default=None, max_length=3)
    status: str | None = None

    @field_validator("content")
    @classmethod
    def _strip_optional_content(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class LastWillResponse(BaseModel):
    id: str
    user_id: str
    agent_id: str
    workspace_id: str | None = None
    content: str
    inactivity_days: int
    contacts: list[LastWillContact]
    status: str
    last_seen_at: str | None = None
    started_at: str | None = None
    triggered_at: str | None = None
    delivered_at: str | None = None
    created_at: str
    updated_at: str


class LastWillDelivery(BaseModel):
    id: str
    last_will_id: str
    channel: str
    contact: LastWillContact
    status: str
    error: str | None = None
    created_at: str
    updated_at: str
