from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.admin_auth import require_admin
from app.models.prompt_template import (
    PromptTemplateResponse,
    PromptTemplateUpdateRequest,
)
from app.services.prompt_store import list_prompts, reset_prompt_text, update_prompt_text

router = APIRouter(prefix="/admin-api/prompts", tags=["admin-prompts"])


@router.get("", response_model=list[PromptTemplateResponse])
async def get_prompts(_: str = Depends(require_admin)):
    prompts = await list_prompts()
    return [PromptTemplateResponse(**prompt) for prompt in prompts]


@router.put("/{key}", response_model=PromptTemplateResponse)
async def update_prompt(
    key: str,
    payload: PromptTemplateUpdateRequest,
    _: str = Depends(require_admin),
):
    try:
        prompt = await update_prompt_text(key, payload.content)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PromptTemplateResponse(**prompt)


@router.post("/{key}/reset", response_model=PromptTemplateResponse)
async def reset_prompt(key: str, _: str = Depends(require_admin)):
    try:
        prompt = await reset_prompt_text(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    return PromptTemplateResponse(**prompt)
