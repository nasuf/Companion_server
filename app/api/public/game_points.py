from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.models.game_points import (
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
