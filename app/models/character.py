"""Pydantic models for character template & profile management."""

from __future__ import annotations

from pydantic import BaseModel


# ── Template ──

class TemplateFieldSchema(BaseModel):
    key: str
    name: str
    type: str  # text / textarea / number / date / select / tags
    required: bool = False
    hint: str | None = None
    options: list[str] | None = None
    min: float | None = None
    max: float | None = None


class TemplateCategorySchema(BaseModel):
    key: str
    name: str
    sort: int
    fields: list[TemplateFieldSchema]


class TemplateSchemaBody(BaseModel):
    categories: list[TemplateCategorySchema]


class TemplateCreateRequest(BaseModel):
    name: str
    description: str | None = None
    schema_body: TemplateSchemaBody
    prompt_header: str | None = None
    defaults: str | None = None
    prompt_requirements: str | None = None


class TemplateUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    schema_body: TemplateSchemaBody | None = None
    prompt_header: str | None = None
    defaults: str | None = None
    prompt_requirements: str | None = None


class TemplateCopyRequest(BaseModel):
    name: str | None = None


class PromptPreviewRequest(BaseModel):
    schema_body: TemplateSchemaBody
    prompt_header: str | None = None
    defaults: str | None = None
    prompt_requirements: str | None = None
    career_id: str | None = None
    index: int = 0


class PromptPreviewResponse(BaseModel):
    prompt: str


class PromptDefaultsResponse(BaseModel):
    header: str
    requirements: str


class TemplateResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    schema_body: TemplateSchemaBody
    prompt_header: str | None = None
    defaults: str | None = None
    prompt_requirements: str | None = None
    status: str
    profile_count: int = 0
    created_at: str
    updated_at: str


# ── Profile ──

class ProfileGenerateRequest(BaseModel):
    template_id: str
    count: int = 1
    career_id: str | None = None  # None = 随机选择职业


class ProfileUpdateRequest(BaseModel):
    name: str | None = None
    data: dict | None = None
    status: str | None = None


class ProfileBatchStatusRequest(BaseModel):
    ids: list[str]
    status: str


class ProfileResponse(BaseModel):
    id: str
    template_id: str
    template_name: str | None = None
    career_id: str | None = None
    career_title: str | None = None
    name: str
    data: dict
    status: str
    agent_id: str | None = None
    created_at: str
    updated_at: str
