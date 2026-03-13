from pydantic import BaseModel


class MessageCreate(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    metadata: dict | None = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict | None = None
    created_at: str | None = None


class ChatRequest(BaseModel):
    message: str
