from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.models.game_points import (
    GameLevelInfo,
    GamePointConvertRequest,
    GamePointConvertResponse,
    GameWalletResponse,
)
from app.services import game_points

router = APIRouter(prefix="/game-wallet", tags=["game-points"])


@router.get("", response_model=GameWalletResponse)
async def get_game_wallet(
    game_key: str | None = Query(default=None, max_length=40),
    payload: dict = Depends(require_user),
):
    return await game_points.get_state(str(payload["sub"]), game_key=game_key)


@router.get("/levels", response_model=list[GameLevelInfo])
async def list_game_levels(_: dict = Depends(require_user)):
    """The full ladder, so clients can show the level-explanation sheet.

    Same data the admin endpoint serves; read-only for regular users.
    """
    return await game_points.list_level_tiers()


@router.post("/convert", response_model=GamePointConvertResponse)
async def convert_game_points(
    data: GamePointConvertRequest,
    payload: dict = Depends(require_user),
):
    try:
        return await game_points.convert_to_shop(
            str(payload["sub"]),
            amount=data.amount,
        )
    except ValueError as exc:
        code = str(exc)
        if code == "insufficient_convertible":
            raise HTTPException(status_code=409, detail=code) from exc
        raise HTTPException(status_code=400, detail=code) from exc
