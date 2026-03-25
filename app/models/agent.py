from pydantic import BaseModel


class AgentCreate(BaseModel):
    name: str
    user_id: str
    personality: dict | None = None
    background: str | None = None
    values: dict | None = None
    gender: str | None = None


class AgentUpdate(BaseModel):
    name: str | None = None
    personality: dict | None = None
    background: str | None = None
    values: dict | None = None


class AgentResponse(BaseModel):
    id: str
    name: str
    user_id: str
    workspace_id: str | None = None
    personality: dict | None = None
    background: str | None = None
    values: dict | None = None
    gender: str | None = None
    life_overview: str | None = None
    created_at: str | None = None
