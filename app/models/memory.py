from pydantic import BaseModel, ConfigDict, Field


class MemoryResponse(BaseModel):
    id: str
    user_id: str
    type: str | None = None
    main_category: str | None = None
    sub_category: str | None = None
    source: str = "user"
    level: int
    content: str
    importance: float
    created_at: str | None = None
    quality: "MemoryQualityResponse | None" = None


class MemoryQualityResponse(BaseModel):
    confidence: float
    evidence_message_ids: list[str] = Field(default_factory=list)
    last_verified_at: str | None = None
    contradiction_state: str = "none"
    user_corrected_count: int = 0
    access_count: int = 0
    signals: list[str] = Field(default_factory=list)


class MemoryExportResponse(BaseModel):
    user_id: str
    workspace_id: str | None = None
    total: int
    memories: list[MemoryResponse]


class MemoryUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(default=None, min_length=1, max_length=4000)


class MemoryBulkDeleteRequest(BaseModel):
    memory_ids: list[str] = Field(min_length=1, max_length=200)


class MemoryBulkDeleteResponse(BaseModel):
    requested: int
    archived: int
    missing_or_forbidden: list[str] = Field(default_factory=list)


class WorkspaceMemoryWipeRequest(BaseModel):
    workspace_id: str
    include_ai: bool = True
    include_user: bool = True


class WorkspaceMemoryWipeResponse(BaseModel):
    workspace_id: str
    archived_user: int
    archived_ai: int


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
