from typing import Any, Literal

from pydantic import BaseModel, Field


class PromptTemplateResponse(BaseModel):
    key: str
    title: str
    stage: str
    category: str
    description: str
    default_text: str
    content: str
    is_enabled: bool = True
    updated_at: str | None = None
    source: str
    revision: int = 0
    web_managed: bool = False
    cache_synced: bool = True
    version_id: str | None = None


class PromptTemplateWriteGuard(BaseModel):
    expected_updated_at: str | None = None
    expected_revision: int | None = Field(default=None, ge=0)


class PromptTemplateUpdateRequest(PromptTemplateWriteGuard):
    content: str
    # 乐观锁: 前端携带其所见的 updated_at 快照; 与 DB 当前值不一致 → 409,
    # 防止两个管理员并发编辑时后保存者静默覆盖前者.
    expected_updated_at: str | None = None


class PromptTemplateEnabledRequest(PromptTemplateWriteGuard):
    is_enabled: bool


class PromptTemplateVersionResponse(BaseModel):
    id: str
    prompt_key: str
    content: str
    source: str
    change_type: str
    eval_result: dict[str, Any] | None = None
    revision: int | None = None
    persistence: str
    created_at: str


class PromptTemplateRestoreVersionRequest(PromptTemplateWriteGuard):
    version_id: str


class PromptTemplateReplayRequest(BaseModel):
    prompt_key: str
    rendered_prompt: str
    model_kind: Literal["chat", "utility"] = "utility"
    messages: list[dict[str, str]] | None = None


class PromptTemplateReplayResponse(BaseModel):
    prompt_key: str
    output: str
    rendered_prompt: str
