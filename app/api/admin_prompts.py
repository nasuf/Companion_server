from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_admin_jwt
from app.models.prompt_template import (
    PromptTemplateRestoreVersionRequest,
    PromptTemplateResponse,
    PromptTemplateUpdateRequest,
    PromptTemplateVersionResponse,
)
from app.services.prompt_store import (
    list_prompt_versions,
    list_prompts,
    reset_prompt_text,
    restore_prompt_version,
    update_prompt_text,
)

router = APIRouter(prefix="/admin-api/prompts", tags=["admin-prompts"])


@router.get("", response_model=list[PromptTemplateResponse])
async def get_prompts(_: str = Depends(require_admin_jwt)):
    prompts = await list_prompts()
    return [PromptTemplateResponse(**prompt) for prompt in prompts]


@router.put("/{key}", response_model=PromptTemplateResponse)
async def update_prompt(
    key: str,
    payload: PromptTemplateUpdateRequest,
    _: str = Depends(require_admin_jwt),
):
    try:
        prompt = await update_prompt_text(key, payload.content)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PromptTemplateResponse(**prompt)


@router.get("/{key}/versions", response_model=list[PromptTemplateVersionResponse])
async def get_prompt_versions(key: str, _: str = Depends(require_admin_jwt)):
    try:
        versions = await list_prompt_versions(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    return [PromptTemplateVersionResponse(**version) for version in versions]


@router.post("/{key}/reset", response_model=PromptTemplateResponse)
async def reset_prompt(key: str, _: str = Depends(require_admin_jwt)):
    try:
        prompt = await reset_prompt_text(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    return PromptTemplateResponse(**prompt)


@router.post("/{key}/restore-version", response_model=PromptTemplateResponse)
async def restore_prompt_from_version(
    key: str,
    payload: PromptTemplateRestoreVersionRequest,
    _: str = Depends(require_admin_jwt),
):
    try:
        prompt = await restore_prompt_version(key, payload.version_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt version not found") from None
    return PromptTemplateResponse(**prompt)
