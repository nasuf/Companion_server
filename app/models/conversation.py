from pydantic import BaseModel


class ConversationCreate(BaseModel):
    user_id: str
    agent_id: str
    workspace_id: str | None = None
    title: str | None = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    agent_id: str
    workspace_id: str | None = None
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
