"""Admin memory repair queue endpoints."""

from __future__ import annotations

from typing import Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query
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


@router.get("/evidence/{side}/{memory_id}")
async def get_memory_evidence(
    side: str, memory_id: str = Path(min_length=1, max_length=200), user_id: str = Query(min_length=1, max_length=200),
    workspace_id: str = Query(min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=50), cursor: str | None = Query(None, max_length=300),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.memory.evidence_read import memory_evidence_detail
    try:
        return await memory_evidence_detail(user_id=user_id, workspace_id=workspace_id,
            side=side, memory_id=memory_id, limit=limit, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/evidence-audit")
async def preview_memory_evidence(
    side: str = Query(pattern="^(user|ai)$"), user_id: str = Query(min_length=1, max_length=200),
    workspace_id: str = Query(min_length=1, max_length=200),
    limit: int = Query(100, ge=1, le=500), after_id: str = Query("", max_length=200),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.memory.evidence_read import audit_evidence_page
    try:
        return await audit_evidence_page(user_id=user_id, workspace_id=workspace_id,
            side=side, limit=limit, after_id=after_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class UpdateMemoryRepairRequest(BaseModel):
    status: str = Field(pattern="^(open|resolved|dismissed)$")
    resolution_note: str | None = None


class MemoryRepairActionRequest(BaseModel):
    expected_updated_at: str | None = Field(default=None, max_length=64)
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


@router.get("/time-history/{side}/{memory_id}")
async def get_memory_time_history(
    side: str, memory_id: str = Path(min_length=1, max_length=200),
    user_id: str = Query(min_length=1, max_length=200), workspace_id: str = Query(min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=50), before: str | None = Query(None, max_length=20),
    at: datetime | None = Query(None), known_at: datetime | None = Query(None),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.memory.temporal_history import read_history
    try:
        return await read_history(user_id=user_id,workspace_id=workspace_id,side=side,
                                  memory_id=memory_id,before=before,limit=limit,at=at,known_at=known_at)
    except ValueError as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404,detail=str(exc)) from exc


class TopicCorrectionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=200)
    workspace_id: str = Field(min_length=1, max_length=200)
    expected_revision: int = Field(ge=1)
    action: str = Field(pattern="^(resume|pause|stop)$")
    reason: str = Field(min_length=1, max_length=500)


@router.get("/topics/page")
async def get_topic_preferences(
    user_id: str = Query(min_length=1, max_length=200),
    workspace_id: str = Query(min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=50), after_id: str = Query("", max_length=64),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.time_domain.topic_store import admin_page
    try:
        return await admin_page(user_id=user_id,workspace_id=workspace_id,limit=limit,after_id=after_id)
    except LookupError as exc:
        raise HTTPException(status_code=404,detail=str(exc)) from exc


@router.post("/topics/{topic_id}/correction")
async def correct_topic_preference(topic_id: str, payload: TopicCorrectionRequest,
                                  user: dict = Depends(require_admin_jwt)) -> dict:
    from app.services.time_domain.topic_store import admin_correct
    try:
        return await admin_correct(actor_id=user['sub'],user_id=payload.user_id,
            workspace_id=payload.workspace_id,ident=topic_id,expected_revision=payload.expected_revision,
            action=payload.action,reason=payload.reason)
    except LookupError as exc:
        raise HTTPException(status_code=404,detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409 if str(exc)=='topic_revision_conflict' else 400,detail=str(exc)) from exc


class TemporalRepairRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=200)
    workspace_id: str = Field(min_length=1, max_length=200)
    evidence_id: str = Field(min_length=1, max_length=200)
    expected_version: str = Field(pattern="^[a-f0-9]{64}$")


@router.get("/time-repair/{side}/{memory_id}")
async def preview_temporal_repair(
    side: str = Path(pattern="^(user|ai)$"),
    memory_id: str = Path(min_length=1, max_length=200),
    user_id: str = Query(min_length=1, max_length=200),
    workspace_id: str = Query(min_length=1, max_length=200),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.memory import temporal_repair
    from app.services.memory.manual_write import MemoryWriteError
    try:
        return await temporal_repair.preview(user_id=user_id, workspace_id=workspace_id,
                                             side=side, memory_id=memory_id)
    except MemoryWriteError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/time-repair/{side}/{memory_id}")
async def apply_temporal_repair(
    payload: TemporalRepairRequest,
    side: str = Path(pattern="^(user|ai)$"),
    memory_id: str = Path(min_length=1, max_length=200),
    user: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.memory import temporal_repair
    from app.services.memory.manual_write import MemoryWriteError
    try:
        return await temporal_repair.apply(side=side, memory_id=memory_id,
                                            actor=user, **payload.model_dump())
    except MemoryWriteError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="temporal_repair_source_changed") from exc


@router.get("/lifecycle/inventory")
async def memory_lifecycle_inventory(
    user_id: str | None = Query(None, max_length=200),
    workspace_id: str | None = Query(None, max_length=200),
    side: str | None = Query(None, pattern="^(user|ai)$"),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    """当前记忆库存: 按层级/来源/归档分布, 每侧计数带分母和采样时刻 (O01.05.V03)。"""
    from app.services.memory.lifecycle_diagnostics import inventory
    return await inventory(user_id=user_id, workspace_id=workspace_id, side=side)


@router.get("/lifecycle/maintenance")
async def memory_lifecycle_maintenance(
    user_id: str | None = Query(None, max_length=200),
    workspace_id: str | None = Query(None, max_length=200),
    days: int = Query(7, ge=1, le=90),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    """维护作业 (hygiene / l3_compression) 近窗口进度 (O01.05.V03)。"""
    from app.services.memory.lifecycle_diagnostics import maintenance
    return await maintenance(user_id=user_id, workspace_id=workspace_id, days=days)
