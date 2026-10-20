"""Read-only activity inspection. The user-facing payload uses the mobile DTO."""

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from app.models.offline import OfflineActivityItem

ActivityFilter = Literal[
    "all", "pending", "planned", "in_progress", "completed", "closed"
]
RelatedSection = Literal["tasks", "media", "fragments", "messages"]
T = TypeVar("T")


class ActivityPage(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
    sampled_at: str


class AdminActivityItem(BaseModel):
    id: str
    user_id: str
    agent_id: str | None
    agent_name: str | None
    workspace_id: str | None
    title: str
    summary: str
    status: str
    stage: str
    reached: bool
    location_name: str | None
    created_at: str
    updated_at: str
    task_count: int
    media_count: int
    fragment_count: int


class AdminActivityDetail(BaseModel):
    activity: OfflineActivityItem
    user_id: str
    agent_name: str | None
    workspace_id: str | None
    conversation_id: str | None
    conversation_available: bool
    stage: str
    source: str
    conditions_ready_at: str | None
    task_generation: str
    archived_at: str | None
    auto_archived: bool
    travel_note: str | None
    counts: dict[str, int]
    copy_quality: dict[str, str | None]
    sampled_at: str


class AdminActivityRelatedPage(ActivityPage[dict[str, Any]]):
    section: RelatedSection
    mode: Literal["linked", "journey"] = "linked"
