from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.prompt_template import (
    PromptTemplateEnabledRequest,
    PromptTemplateReplayRequest,
    PromptTemplateReplayResponse,
    PromptTemplateRestoreVersionRequest,
    PromptTemplateResponse,
    PromptTemplateUpdateRequest,
    PromptTemplateVersionResponse,
)
from app.services.llm.models import convert_messages, get_chat_model, get_utility_model, invoke_text
from app.services.llm.resilience import CallProfile
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting.store import (
    PromptUpdateConflictError,
    list_prompt_versions,
    list_prompts,
    reset_prompt_text,
    restore_prompt_version,
    set_prompt_enabled,
    update_prompt_text,
)

router = APIRouter(prefix="/admin-api/prompts", tags=["admin-prompts"])

_ALLOWED_REPLAY_ROLES = {"system", "user", "assistant"}

# Replay is a prompt-debugging tool: it must exercise the *exact* selected
# model. Ollama fallback is disabled on purpose — silently answering with a
# different local model would make the admin evaluate the wrong output. No
# retry either: the admin can just click again; a hung upstream should surface
# as an error instead of blocking the HTTP request for minutes.
_REPLAY_PROFILE = CallProfile(
    timeout_s=60.0,
    max_retries=0,
    retry_backoff_s=(),
    allow_ollama_fallback=False,
)


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
        prompt = await update_prompt_text(
            key,
            payload.content,
            expected_updated_at=payload.expected_updated_at,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    except PromptUpdateConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PromptTemplateResponse(**prompt)


@router.put("/{key}/enabled", response_model=PromptTemplateResponse)
async def update_prompt_enabled(
    key: str,
    payload: PromptTemplateEnabledRequest,
    _: str = Depends(require_admin_jwt),
):
    """启用/停用提示词. 停用后该模板从运行时最终输入中彻底移除:
    组合 section → 该段不注入; 独立步骤 prompt → 该 LLM 调用跳过走 fallback."""
    try:
        prompt = await set_prompt_enabled(key, payload.is_enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail="Prompt not found") from None
    return PromptTemplateResponse(**prompt)


@router.post("/replay", response_model=PromptTemplateReplayResponse)
async def replay_prompt_step(
    payload: PromptTemplateReplayRequest,
    _: str = Depends(require_admin_jwt),
):
    """Side-effect-free single-step replay for the trace debug panel.

    `output` is intentionally the model's *raw* text: no timestamp stripping,
    no `||` bubble splitting, no [EMO:] marker removal. The admin needs to see
    exactly what the prompt produced; production-only post-processing is
    explained in the web UI next to the output.
    """
    if payload.prompt_key not in PROMPT_DEFINITION_MAP:
        raise HTTPException(status_code=404, detail="Prompt not found")
    rendered_prompt = payload.rendered_prompt
    if not rendered_prompt.strip():
        raise HTTPException(status_code=400, detail="rendered_prompt is required")
    model = get_chat_model() if payload.model_kind == "chat" else get_utility_model()
    # Validate outside the try: a malformed payload must surface as 400, not be
    # re-wrapped by the generic 502 "replay failed" handler below.
    replay_messages = _normalize_replay_messages(payload.messages) if payload.messages else None
    try:
        if replay_messages:
            output = await invoke_text(
                model, convert_messages(replay_messages), profile=_REPLAY_PROFILE,
            )
        else:
            output = await invoke_text(model, rendered_prompt, profile=_REPLAY_PROFILE)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Prompt replay failed: {exc}") from exc
    return PromptTemplateReplayResponse(
        prompt_key=payload.prompt_key,
        rendered_prompt=rendered_prompt,
        output=str(output or ""),
    )


@router.get("/{key}/versions", response_model=list[PromptTemplateVersionResponse])
async def get_prompt_versions(
    key: str,
    limit: int = Query(default=20, ge=1, le=200),
    _: str = Depends(require_admin_jwt),
):
    try:
        versions = await list_prompt_versions(key, limit=limit)
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
