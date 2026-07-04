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
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.models.agent import PersonalityInput
from app.services.agent_template import (
    get_default_template_agent_id,
    get_or_create_template_user,
    list_template_agents,
    set_default_template_agent_id,
)
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
        "progress": progress,
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
    )
    logger.info("[TEMPLATE] created template agent %s", agent.id[:8])
    return {
        "id": agent.id,
        "name": agent.name,
        "status": agent.status,
        "workspace_id": workspace.id if workspace else None,
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
