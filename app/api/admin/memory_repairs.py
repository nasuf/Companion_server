"""Admin memory repair queue endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.services.memory.repair_actions import (
    MemoryRepairActionError,
    MemoryRepairActionPayload,
    apply_memory_repair_action,
)
from app.services.memory.repair_queue import (
    list_memory_repair_items,
    update_memory_repair_item_status,
)
from app.services.memory.lifecycle.hygiene import run_memory_hygiene
from app.services.memory.lifecycle.quality_state import (
    backfill_memory_quality_states,
    list_low_quality_memory_states,
)

router = APIRouter(prefix="/admin-api/memory-repairs", tags=["admin-memory-repairs"])


class UpdateMemoryRepairRequest(BaseModel):
    status: str = Field(pattern="^(open|resolved|dismissed)$")
    resolution_note: str | None = None


class MemoryRepairActionRequest(BaseModel):
    action: str = Field(
        pattern="^(archive_memory|downgrade_memory|edit_memory|insert_replacement_memory|mark_verified|merge_memories)$"
    )
    memory_id: str | None = None
    memory_ids: list[str] = Field(default_factory=list, max_length=20)
    source: str | None = Field(default=None, pattern="^(user|ai)$")
    user_id: str | None = None
    workspace_id: str | None = None
    content: str | None = Field(default=None, min_length=1, max_length=4000)
    level: int | None = Field(default=None, ge=1, le=3)
    importance: float | None = Field(default=None, ge=0.1, le=0.99)
    memory_type: str | None = None
    main_category: str | None = None
    sub_category: str | None = None
    reason: str | None = Field(default=None, max_length=1000)


class QualityBackfillRequest(BaseModel):
    user_id: str | None = None
    workspace_id: str | None = None
    limit: int = Field(default=500, ge=1, le=2000)


class ConsolidationRunRequest(BaseModel):
    user_id: str | None = None
    workspace_id: str | None = None
    allow_llm: bool = True
    max_scopes: int = Field(default=50, ge=1, le=200)
    max_memories_per_scope: int = Field(default=200, ge=10, le=1000)


@router.get("")
async def list_memory_repairs(
    status: str | None = Query("open", pattern="^(open|resolved|dismissed|all)$"),
    source_type: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    workspace_id: str | None = None,
    memory_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    items = await list_memory_repair_items(
        status=status,
        source_type=source_type,
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        memory_id=memory_id,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "limit": limit,
        "offset": offset,
        "count": len(items),
    }


@router.get("/quality-states")
async def list_memory_quality_states(
    limit: int = Query(100, ge=1, le=500),
    max_confidence: float = Query(0.55, ge=0.0, le=0.99),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    items = await list_low_quality_memory_states(
        limit=limit,
        max_confidence=max_confidence,
    )
    return {"items": items, "count": len(items), "limit": limit}


@router.post("/quality-states/backfill")
async def backfill_memory_quality(
    payload: QualityBackfillRequest,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    return await backfill_memory_quality_states(
        user_id=payload.user_id,
        workspace_id=payload.workspace_id,
        limit=payload.limit,
    )


@router.post("/consolidation/run")
async def run_consolidation_now(
    payload: ConsolidationRunRequest,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    return await run_memory_hygiene(
        user_id=payload.user_id,
        workspace_id=payload.workspace_id,
        allow_llm=payload.allow_llm,
        max_scopes=payload.max_scopes,
        max_memories_per_scope=payload.max_memories_per_scope,
    )


@router.patch("/{item_id}")
async def update_memory_repair(
    item_id: str,
    payload: UpdateMemoryRepairRequest,
    user: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        item = await update_memory_repair_item_status(
            item_id,
            status=payload.status,
            resolution_note=(payload.resolution_note or "").strip() or None,
            resolved_by_id=user.get("sub"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if item is None:
        raise HTTPException(status_code=404, detail="memory_repair_item_not_found")
    return item


@router.post("/{item_id}/actions")
async def apply_memory_repair(
    item_id: str,
    payload: MemoryRepairActionRequest,
    user: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        return await apply_memory_repair_action(
            item_id,
            payload=MemoryRepairActionPayload(**payload.model_dump()),
            admin_id=user.get("sub"),
        )
    except MemoryRepairActionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
