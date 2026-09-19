"""Pydantic models for agent name-library management."""

from pydantic import BaseModel, Field


class NameCreateRequest(BaseModel):
    name: str
    nickname: str = ""
    gender: str
    sort_order: int = 0


class NameUpdateRequest(BaseModel):
    name: str | None = None
    nickname: str | None = None
    gender: str | None = None
    status: str | None = None
    sort_order: int | None = None


class NameResponse(BaseModel):
    id: str
    name: str
    nickname: str
    gender: str
    status: str
    sort_order: int
    created_at: str
    updated_at: str


class NameListResponse(BaseModel):
    items: list[NameResponse]
    total: int
    limit: int
    offset: int
    counts: dict[str, int] = Field(default_factory=dict)
