"""Pydantic models for career template management."""

from pydantic import BaseModel


class CareerCreateRequest(BaseModel):
    title: str
    duties: str
    social_value: str
    clients: str
    sort_order: int = 0


class CareerUpdateRequest(BaseModel):
    title: str | None = None
    duties: str | None = None
    social_value: str | None = None
    clients: str | None = None
    sort_order: int | None = None


class CareerResponse(BaseModel):
    id: str
    title: str
    duties: str
    social_value: str
    clients: str
    status: str
    sort_order: int
    profile_count: int = 0
    created_at: str
    updated_at: str
