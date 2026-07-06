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
    canary_config: dict[str, Any] | None = None
    updated_at: str | None = None
    source: str


class PromptTemplateUpdateRequest(BaseModel):
    content: str
    # 乐观锁: 前端携带其所见的 updated_at 快照; 与 DB 当前值不一致 → 409,
    # 防止两个管理员并发编辑时后保存者静默覆盖前者.
    expected_updated_at: str | None = None


class PromptTemplateEnabledRequest(BaseModel):
    is_enabled: bool


class PromptTemplateVersionResponse(BaseModel):
    id: str
    prompt_key: str
    content: str
    source: str
    change_type: str
    eval_result: dict[str, Any] | None = None
    persistence: str
    created_at: str


class PromptTemplateRestoreVersionRequest(BaseModel):
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


class PromptCanaryConfigRequest(BaseModel):
    is_enabled: bool = False
    mode: Literal["off", "agents", "percent"] = "off"
    content: str | None = None
    agent_ids: list[str] = Field(default_factory=list)
    rollout_percent: int = Field(default=0, ge=0, le=100)


class PromptCanaryConfigResponse(BaseModel):
    prompt_key: str
    is_enabled: bool
    mode: Literal["off", "agents", "percent"]
    content: str | None = None
    agent_ids: list[str] = Field(default_factory=list)
    rollout_percent: int
    eval_result: dict[str, Any] | None = None
    updated_at: str | None = None
