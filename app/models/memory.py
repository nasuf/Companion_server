from pydantic import BaseModel


class MemoryResponse(BaseModel):
    id: str
    user_id: str
    type: str | None = None
    main_category: str | None = None
    sub_category: str | None = None
    source: str = "user"
    level: int
    content: str
    summary: str | None = None
    importance: float
    created_at: str | None = None


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 10
    workspace_id: str | None = None
    main_category: str | None = None
    sub_category: str | None = None


class MemoryStatsBucket(BaseModel):
    key: str
    count: int


class MemoryStatsResponse(BaseModel):
    total: int
    by_level: list[MemoryStatsBucket]
    by_main_category: list[MemoryStatsBucket]
    by_sub_category: list[MemoryStatsBucket]
    by_main_sub: dict[str, int] = {}  # "身份-姓名" → count
