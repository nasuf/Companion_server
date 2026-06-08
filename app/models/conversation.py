from pydantic import BaseModel

from app.models.music import MusicCoListeningResponse


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
    interaction_days: int | None = None
    ai_status: str | None = None
    ai_status_label: str | None = None
    ai_activity: str | None = None
    music_co_listening: MusicCoListeningResponse | None = None
