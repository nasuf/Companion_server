from pydantic import BaseModel


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


class PromptTemplateUpdateRequest(BaseModel):
    content: str


class PromptTemplateVersionResponse(BaseModel):
    id: str
    prompt_key: str
    content: str
    source: str
    change_type: str
    persistence: str
    created_at: str


class PromptTemplateRestoreVersionRequest(BaseModel):
    version_id: str
