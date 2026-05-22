"""Admin memory repair queue endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.services.memory.repair_queue import (
    list_memory_repair_items,
    update_memory_repair_item_status,
)

router = APIRouter(prefix="/admin-api/memory-repairs", tags=["admin-memory-repairs"])


class UpdateMemoryRepairRequest(BaseModel):
    status: str = Field(pattern="^(open|resolved|dismissed)$")
    resolution_note: str | None = None


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
