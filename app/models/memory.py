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


class MemoryStatsGroup(BaseModel):
    level: int
    main_category: str
    sub_category: str
    count: int


class MemoryStatsResponse(BaseModel):
    total: int
    groups: list[MemoryStatsGroup]


class MemoryHygieneMemory(BaseModel):
    id: str
    source: str
    level: int
    main_category: str | None = None
    sub_category: str | None = None
    content: str
    summary: str | None = None
    importance: float


class MemoryHygieneChange(BaseModel):
    action: str
    source: str
    main_category: str | None = None
    sub_category: str | None = None
    kept: MemoryHygieneMemory | None = None
    removed: MemoryHygieneMemory | None = None
    before: str | None = None
    after: str | None = None
    reason: str


class MemoryHygieneRequest(BaseModel):
    workspace_id: str | None = None
    allow_llm: bool = True
    max_memories_per_scope: int = 200


class MemoryHygieneResponse(BaseModel):
    scopes: int
    checked: int
    archived: int
    merged: int
    updated: int
    errors: int
    changes: list[MemoryHygieneChange]
