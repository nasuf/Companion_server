from pydantic import BaseModel


class UserUpdate(BaseModel):
    email: str | None = None


class UserResponse(BaseModel):
    id: str
    username: str
    email: str | None = None
    created_at: str | None = None


class ProfileStatsResponse(BaseModel):
    workspace_id: str
    intimacy_stage: str
    intimacy_stage_label: str
    topic_intimacy: float
    companion_days: int
    chat_hours: int
    message_count: int
    companion_summary: str
