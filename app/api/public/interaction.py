"""Workspace consecutive-interaction calendar and makeup-card consume."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.api.ownership import require_workspace_owner
from app.models.interaction import (
    InteractionMakeupRequest,
    InteractionMakeupResponse,
    InteractionOverviewResponse,
)
from app.services.interaction_streak import MakeupError, apply_makeup, get_interaction_overview

router = APIRouter(prefix="/workspaces", tags=["interaction"])

_MAKEUP_HTTP = {
    "workspace_not_found": (404, "工作区不存在"),
    "cannot_makeup_today": (400, "今天还没过完，发一条消息就算互动"),
    "cannot_makeup_future": (400, "不能补签未来的日期"),
    "before_workspace": (400, "不能补签认识之前的日期"),
    "outside_lookback": (400, "只能补签最近 30 天内漏掉的日子"),
    "already_marked": (409, "这一天已经有互动了"),
    "insufficient_inventory": (400, "补签卡不足"),
}


def _raise_makeup(exc: MakeupError) -> None:
    status, detail = _MAKEUP_HTTP.get(str(exc), (400, "补签失败"))
    raise HTTPException(status_code=status, detail=detail) from exc


@router.get("/{workspace_id}/interaction", response_model=InteractionOverviewResponse)
async def get_workspace_interaction(
    workspace_id: str,
    year: int | None = Query(default=None, ge=2000, le=2100),
    month: int | None = Query(default=None, ge=1, le=12),
    workspace=Depends(require_workspace_owner),
    user: dict = Depends(require_user),
):
    try:
        return await get_interaction_overview(
            workspace_id,
            user.get("sub") or workspace.userId,
            year=year,
            month=month,
        )
    except MakeupError as exc:
        _raise_makeup(exc)


@router.post("/{workspace_id}/interaction/makeup", response_model=InteractionMakeupResponse)
async def makeup_workspace_interaction(
    workspace_id: str,
    data: InteractionMakeupRequest,
    workspace=Depends(require_workspace_owner),
    user: dict = Depends(require_user),
):
    try:
        return await apply_makeup(
            workspace_id,
            user.get("sub") or workspace.userId,
            data.date,
        )
    except MakeupError as exc:
        _raise_makeup(exc)
