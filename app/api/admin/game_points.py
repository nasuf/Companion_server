from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.game_points import (
    AdminGamePointLedgerItem,
    GameLevelInfo,
    GameLevelTiersPayload,
    GamePointGrantRequest,
    GamePointGrantResponse,
    GamePointRulePayload,
    GamePointRuleResponse,
)
from app.services import game_points

router = APIRouter(
    prefix="/admin-api/game-points",
    tags=["admin", "game-points"],
    dependencies=[Depends(require_admin_jwt)],
)


def _http_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    if code == "unsupported_game":
        return HTTPException(status_code=404, detail=code)
    return HTTPException(status_code=422, detail=code)


@router.get("/levels", response_model=list[GameLevelInfo])
async def list_game_levels():
    return await game_points.list_level_tiers()


@router.put("/levels", response_model=list[GameLevelInfo])
async def update_game_levels(payload: GameLevelTiersPayload):
    try:
        return await game_points.replace_level_tiers(
            [tier.model_dump() for tier in payload.tiers]
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/rules", response_model=list[GamePointRuleResponse])
async def list_game_point_rules():
    return await game_points.list_point_rules()


@router.put("/rules/{game_key}", response_model=GamePointRuleResponse)
async def update_game_point_rule(game_key: str, payload: GamePointRulePayload):
    try:
        return await game_points.update_point_rule(game_key, payload.rules)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/ledger", response_model=list[AdminGamePointLedgerItem])
async def list_game_point_ledger(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await game_points.list_admin_ledger(
        user_id=user_id,
        limit=limit,
        offset=offset,
    )


@router.post("/grant", response_model=GamePointGrantResponse)
async def grant_game_points(payload: GamePointGrantRequest):
    try:
        return await game_points.admin_grant(
            payload.user_id,
            payload.amount,
            note=payload.note,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
