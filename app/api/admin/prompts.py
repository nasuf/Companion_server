from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_admin_jwt
from app.models.prompt_template import (
    PromptCanaryConfigRequest,
    PromptCanaryConfigResponse,
    PromptTemplateReplayRequest,
    PromptTemplateReplayResponse,
    PromptTemplateRestoreVersionRequest,
    PromptTemplateResponse,
    PromptTemplateUpdateRequest,
    PromptTemplateVersionResponse,
)
from app.services.llm.models import convert_messages, get_chat_model, get_utility_model, invoke_text
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.store import (
    disable_prompt_canary,
    get_prompt_canary_config,
    list_prompt_versions,
    list_prompts,
    reset_prompt_text,
    restore_prompt_version,
    update_prompt_canary_config,
    update_prompt_text,
)

router = APIRouter(prefix="/admin-api/prompts", tags=["admin-prompts"])

_ALLOWED_REPLAY_ROLES = {"system", "user", "assistant"}


def _normalize_replay_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")
        if role not in _ALLOWED_REPLAY_ROLES:
            raise HTTPException(status_code=400, detail=f"messages[{index}].role is invalid")
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=400, detail=f"messages[{index}].content is required")
        normalized.append({"role": role, "content": content})
    return normalized


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


@router.post("/replay", response_model=PromptTemplateReplayResponse)
async def replay_prompt_step(
    payload: PromptTemplateReplayRequest,
    _: str = Depends(require_admin_jwt),
):
    if payload.prompt_key not in PROMPT_DEFINITION_MAP:
        raise HTTPException(status_code=404, detail="Prompt not found")
    rendered_prompt = payload.rendered_prompt
    if not rendered_prompt.strip():
        raise HTTPException(status_code=400, detail="rendered_prompt is required")
    model = get_chat_model() if payload.model_kind == "chat" else get_utility_model()
    try:
        if payload.messages:
            replay_messages = _normalize_replay_messages(payload.messages)
            result = await model.ainvoke(convert_messages(replay_messages))
            output = str(getattr(result, "content", "") or "")
        else:
            output = await invoke_text(model, rendered_prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Prompt replay failed: {exc}") from exc
    return PromptTemplateReplayResponse(
        prompt_key=payload.prompt_key,
        rendered_prompt=rendered_prompt,
        output=output,
    )


@router.get("/{key}/versions", response_model=list[PromptTemplateVersionResponse])
async def get_prompt_versions(key: str, _: str = Depends(require_admin_jwt)):
    try:
        versions = await list_prompt_versions(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    return [PromptTemplateVersionResponse(**version) for version in versions]


@router.get("/{key}/canary", response_model=PromptCanaryConfigResponse)
async def get_prompt_canary(key: str, _: str = Depends(require_admin_jwt)):
    try:
        config = await get_prompt_canary_config(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    return PromptCanaryConfigResponse(**config)


@router.put("/{key}/canary", response_model=PromptCanaryConfigResponse)
async def update_prompt_canary(
    key: str,
    payload: PromptCanaryConfigRequest,
    _: str = Depends(require_admin_jwt),
):
    try:
        config = await update_prompt_canary_config(
            key,
            is_enabled=payload.is_enabled,
            mode=payload.mode,
            content=payload.content,
            agent_ids=payload.agent_ids,
            rollout_percent=payload.rollout_percent,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PromptCanaryConfigResponse(**config)


@router.delete("/{key}/canary", response_model=PromptCanaryConfigResponse)
async def delete_prompt_canary(key: str, _: str = Depends(require_admin_jwt)):
    try:
        config = await disable_prompt_canary(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    return PromptCanaryConfigResponse(**config)


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
