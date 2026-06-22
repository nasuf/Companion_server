from pydantic import BaseModel, Field


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


class UserLocationRequest(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    city: str | None = Field(default=None, max_length=80)
    region: str | None = Field(default=None, max_length=80)
    country: str | None = Field(default=None, max_length=80)
    source: str = Field(default="device", max_length=40)
    permission_status: str = Field(default="unknown", max_length=40)


class UserLocationResponse(BaseModel):
    has_location: bool
    latitude: float | None = None
    longitude: float | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    permission_status: str | None = None
    updated_at: str | None = None
