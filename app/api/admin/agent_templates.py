"""Admin API: agent template management.

A "template" is a fully-provisioned agent (same pipeline as a Flutter user's
manual agent creation: MBTI + background + L1 self-memory + embeddings +
schedule) owned by a reserved system user. New users (e.g. WeChat Mini Program
first login) are cheaply cloned from the *default* template so they can chat
immediately with zero LLM warm-up.

Endpoints (all admin-only):
  GET    /admin-api/agent-templates              — list templates + status + default
  POST   /admin-api/agent-templates              — create a template (runs provisioning)
  GET    /admin-api/agent-templates/{id}/status  — provisioning progress
  PUT    /admin-api/agent-templates/default      — set / clear the default template
  DELETE /admin-api/agent-templates/{id}         — delete a template (+ clear default)

Knowledge supplement (append facts to an EXISTING template, then publish):
  POST   /admin-api/agent-templates/{id}/knowledge/from-document — append memories
  GET    /admin-api/agent-templates/{id}/knowledge               — overview + agents
  POST   /admin-api/agent-templates/{id}/knowledge/sync          — publish (canary/all)
  GET    /admin-api/agent-templates/{id}/knowledge/sync-status   — job progress
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.models.agent import PersonalityInput
from app.services.agent_template import (
    count_active_clones,
    get_default_template_agent_id,
    get_or_create_template_user,
    list_template_agents,
    set_default_template_agent_id,
)
from app.services.agent_template.document_import import parse_agent_profile_document
from app.services.agent_template.knowledge import (
    KnowledgeSyncBusy,
    append_knowledge_to_template,
    get_knowledge_status,
    get_related_agents_with_pending,
    get_sync_progress,
    list_knowledge_items,
    start_knowledge_sync,
)
from app.services.agent_template.knowledge_import import parse_knowledge_document
from app.services.life_story import get_progress
from app.services.runtime.data_reset import hard_delete_agent_data

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/agent-templates",
    tags=["admin", "agent-templates"],
    dependencies=[Depends(require_admin_jwt)],
)


class TemplateCreateRequest(BaseModel):
    name: str
    personality: PersonalityInput
    gender: str | None = None
    background: str | None = None


class DefaultTemplateRequest(BaseModel):
    # None / empty clears the default (new users then stay agent-less).
    agent_id: str | None = None


class KnowledgeSyncRequest(BaseModel):
    # None / empty = full sync to ALL related agents; a non-empty list is a
    # canary sync to hand-picked agents (never advances the pending badge).
    # Capped: selected mode validates ids one-by-one; huge lists should use
    # full mode instead.
    agent_ids: list[str] | None = Field(default=None, max_length=200)


_DEFAULT_DOCUMENT_PERSONALITY = {
    "lively": 36,
    "rational": 62,
    "emotional": 78,
    "planned": 70,
    "spontaneous": 34,
    "creative": 58,
    "humor": 42,
}
_MAX_TEMPLATE_DOCUMENT_BYTES = 2 * 1024 * 1024


def _parse_personality_form(raw: str | None) -> PersonalityInput:
    if not raw or not raw.strip():
        return PersonalityInput(**_DEFAULT_DOCUMENT_PERSONALITY)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="personality 必须是 JSON") from exc
    try:
        return PersonalityInput.model_validate(value)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="personality 字段不合法") from exc


def _normalize_gender(value: str | None) -> str | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text in {"female", "male"}:
        return text
    if "女" in text:
        return "female"
    if "男" in text:
        return "male"
    raise HTTPException(status_code=400, detail="gender 只能是 male/female/男/女")


async def _require_template(agent_id: str):
    """Load a template agent or 404 (must belong to the template system user)."""
    owner = await get_or_create_template_user()
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or agent.userId != owner.id:
        raise HTTPException(status_code=404, detail="Template not found")
    return agent


async def _l1_memory_count(agent_id: str) -> int:
    rows = await db.query_raw(
        """
        SELECT COUNT(*)::int AS n
        FROM memories_ai m
        JOIN chat_workspaces w ON w.id = m.workspace_id
        WHERE w.agent_id = $1 AND w.status = 'active'
          AND m.level = 1 AND m.is_archived = FALSE
        """,
        agent_id,
    )
    return int(rows[0]["n"]) if rows else 0


async def _template_summary(agent, *, default_id: str | None) -> dict:
    progress = await get_progress(agent.id)
    knowledge = await get_knowledge_status(agent.id)
    return {
        "id": agent.id,
        "name": agent.name,
        "status": agent.status,
        "gender": agent.gender,
        "age": agent.age,
        "occupation": agent.occupation,
        "city": agent.city,
        "avatar_url": agent.avatarUrl,
        "created_at": str(agent.createdAt),
        "is_default": agent.id == default_id,
        "l1_memory_count": await _l1_memory_count(agent.id),
        "clone_count": await count_active_clones(agent.id),
        "progress": progress,
        # 记忆补充徽标: 有知识记忆晚于上次全量同步水位 → 提示"未同步"
        "knowledge_count": knowledge["knowledge_count"],
        "knowledge_pending": knowledge["pending_update"],
    }


@router.get("")
async def list_templates() -> dict:
    """List all template agents with provisioning status + the current default."""
    default_id = await get_default_template_agent_id()
    agents = await list_template_agents()
    return {
        "default_template_agent_id": default_id,
        "templates": [
            await _template_summary(a, default_id=default_id) for a in agents
        ],
    }


@router.post("")
async def create_template(data: TemplateCreateRequest) -> dict:
    """Create a template agent via the full provisioning pipeline.

    Runs the identical flow used when a Flutter user manually creates an agent,
    so the template ends up with all the same data. Provisioning is async
    (~90s); poll ``GET /{id}/status`` until stage == "complete".
    """
    # Import here to avoid a public<-admin import cycle at module load.
    from app.api.public.agents import create_agent_with_provisioning

    owner = await get_or_create_template_user()
    agent, workspace = await create_agent_with_provisioning(
        user_id=owner.id,
        name=data.name.strip() or "模板伙伴",
        personality=data.personality.model_dump(),
        background=data.background,
        gender=data.gender,
        # Templates coexist (one is the default); creating a new one must NOT
        # archive the template system user's other templates.
        stage_existing_workspaces=False,
    )
    logger.info("[TEMPLATE] created template agent %s", agent.id[:8])
    return {
        "id": agent.id,
        "name": agent.name,
        "status": agent.status,
        "workspace_id": workspace.id if workspace else None,
    }


@router.post("/from-document")
async def create_template_from_document(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    gender: str | None = Form(None),
    personality: str | None = Form(None),
) -> dict:
    """Create a template agent from an uploaded five-dimension profile document.

    The document is converted into the same CharacterProfile JSON shape used by
    the normal provisioning path, then existing L1 memory conversion / embedding
    / schedule generation still run unchanged.
    """
    raw = await file.read()
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(raw) > _MAX_TEMPLATE_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="文档过大，请控制在 2MB 以内")

    try:
        imported = parse_agent_profile_document(raw, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    personality_input = _parse_personality_form(personality)
    template_name = (name or imported.name or "").strip() or "文档模板伙伴"
    template_gender = _normalize_gender(gender) or _normalize_gender(imported.gender)

    from app.api.public.agents import create_agent_with_provisioning

    owner = await get_or_create_template_user()
    agent, workspace = await create_agent_with_provisioning(
        user_id=owner.id,
        name=template_name,
        personality=personality_input.model_dump(),
        background=imported.background,
        gender=template_gender,
        profile_override=imported.profile,
        career_template_override=imported.career_template,
        # Templates coexist (one is the default); creating a new one must NOT
        # archive the template system user's other templates.
        stage_existing_workspaces=False,
    )
    logger.info(
        "[TEMPLATE] created document template agent %s from %s",
        agent.id[:8],
        file.filename or "<upload>",
    )
    if imported.oversized_fields:
        logger.warning(
            "[TEMPLATE] document template %s has %d oversized fields: %s",
            agent.id[:8],
            len(imported.oversized_fields),
            ", ".join(p for p, _ in imported.oversized_fields[:5]),
        )
    return {
        "id": agent.id,
        "name": agent.name,
        "status": agent.status,
        "workspace_id": workspace.id if workspace else None,
        "source": "document",
        # 这条路径绕过 LLM, 收紧字数的 prompt 约束对它无效。超长字段转成记忆后检索
        # 时会被整条跳过 —— 不拦创建 (整份档案不该因为一个字段太长就作废), 但必须
        # 让管理员看见, 否则他得到一个"部分记忆是死的"模板还不知道。
        "oversized_fields": [
            {"field": path, "preview": piece[:120]}
            for path, piece in imported.oversized_fields
        ],
    }


@router.get("/{agent_id}/status")
async def template_status(agent_id: str) -> dict:
    """Provisioning progress for a single template agent."""
    owner = await get_or_create_template_user()
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or agent.userId != owner.id:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "id": agent.id,
        "status": agent.status,
        "progress": await get_progress(agent_id),
        "l1_memory_count": await _l1_memory_count(agent_id),
    }


@router.put("/default")
async def set_default_template(data: DefaultTemplateRequest) -> dict:
    """Set (or clear) the default template used to clone new users."""
    agent_id = (data.agent_id or "").strip() or None
    if agent_id is not None:
        owner = await get_or_create_template_user()
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if not agent or agent.userId != owner.id:
            raise HTTPException(status_code=404, detail="Template not found")
        if agent.status != "active":
            raise HTTPException(
                status_code=400,
                detail="模板尚未生成完成，无法设为默认",
            )
    await set_default_template_agent_id(agent_id)
    return {"default_template_agent_id": agent_id}


# ── Knowledge supplement (记忆补充 + 同步) ────────────────────────────


@router.post("/{agent_id}/knowledge/from-document")
async def append_template_knowledge(
    agent_id: str,
    file: UploadFile = File(...),
) -> dict:
    """Append a knowledge document (公司/产品/活动 facts) to an EXISTING template.

    New rows land in the template's memory bank immediately (future clones get
    them automatically); publishing to EXISTING clones goes through
    ``POST /{id}/knowledge/sync``.
    """
    agent = await _require_template(agent_id)
    if agent.status != "active":
        raise HTTPException(status_code=400, detail="模板尚未生成完成，无法补充记忆")

    raw = await file.read()
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(raw) > _MAX_TEMPLATE_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="文档过大，请控制在 2MB 以内")

    try:
        items = parse_knowledge_document(raw, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await append_knowledge_to_template(
            template_agent_id=agent_id,
            template_user_id=agent.userId,
            items=items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "[TEMPLATE] knowledge document %s appended to template %s (+%d/-%d)",
        file.filename or "<upload>",
        agent_id[:8],
        result["stored"],
        result["skipped_duplicates"],
    )
    status = await get_knowledge_status(agent_id)
    return {
        **result,
        **status,
        "items": [
            {"section": item.section, "label": item.label, "summary": item.summary}
            for item in items[:50]
        ],
    }


@router.get("/{agent_id}/knowledge")
async def template_knowledge_overview(agent_id: str) -> dict:
    """Knowledge rows + related agents (with per-agent pending diff) + job state."""
    await _require_template(agent_id)
    status = await get_knowledge_status(agent_id)
    return {
        "template_id": agent_id,
        **status,
        "items": await list_knowledge_items(agent_id),
        "agents": await get_related_agents_with_pending(agent_id),
        "sync": await get_sync_progress(agent_id),
    }


@router.post("/{agent_id}/knowledge/sync")
async def sync_template_knowledge(agent_id: str, data: KnowledgeSyncRequest) -> dict:
    """Publish template knowledge to cloned agents (background job).

    ``agent_ids`` empty/None → all related agents (full sync; clears the
    pending badge on success). A non-empty list → canary sync for testing a
    single agent's memory quality before the full rollout.
    """
    agent = await _require_template(agent_id)
    if agent.status != "active":
        raise HTTPException(status_code=400, detail="模板尚未生成完成，无法同步记忆")
    try:
        return await start_knowledge_sync(
            template_agent_id=agent_id, agent_ids=data.agent_ids
        )
    except KnowledgeSyncBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{agent_id}/knowledge/sync-status")
async def template_knowledge_sync_status(agent_id: str) -> dict:
    """Poll the background sync job progress."""
    await _require_template(agent_id)
    progress = await get_sync_progress(agent_id)
    return {"sync": progress or {"status": "idle"}}


@router.delete("/{agent_id}")
async def delete_template(agent_id: str) -> dict:
    """Hard-delete a template agent and all its data; clear default if needed."""
    owner = await get_or_create_template_user()
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or agent.userId != owner.id:
        raise HTTPException(status_code=404, detail="Template not found")

    default_id = await get_default_template_agent_id()
    stats = await hard_delete_agent_data(agent_id, owner.id)
    if default_id == agent_id:
        await set_default_template_agent_id(None)
    return {"ok": True, "stats": stats, "cleared_default": default_id == agent_id}
