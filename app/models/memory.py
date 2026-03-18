from pydantic import BaseModel


class MemoryResponse(BaseModel):
    id: str
    user_id: str
    type: str | None = None
    source: str = "user"
    level: int
    content: str
    summary: str | None = None
    importance: float
    created_at: str | None = None


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 10
