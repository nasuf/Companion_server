from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.jwt_auth import require_admin_jwt
from app.services.games import balance


router = APIRouter(
    prefix="/admin-api/game-configs",
    tags=["admin", "game-configs"],
    dependencies=[Depends(require_admin_jwt)],
)


class GameConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed", "adaptive"] = "adaptive"
    base_strength: int = Field(ge=0, le=100)
    min_strength: int = Field(ge=0, le=100)
    max_strength: int = Field(ge=0, le=100)
    target_user_rate: float = Field(ge=0.05, le=0.95)
    adjustment_window: int = Field(ge=2, le=50)
    minimum_games: int = Field(ge=1, le=20)
    maximum_step: int = Field(ge=1, le=15)
    algorithm_overrides: dict[str, Any] = Field(default_factory=dict)
    # AI reaction-time range (ms): each move takes a random wall-clock delay in
    # [min, max] so the opponent feels human rather than instant/robotic.
    # Defaults keep older restored versions valid.
    min_response_ms: int = Field(default=900, ge=0, le=8000)
    max_response_ms: int = Field(default=1600, ge=0, le=8000)

    @model_validator(mode="after")
    def validate_strength_range(self) -> GameConfigPayload:
        if not self.min_strength <= self.base_strength <= self.max_strength:
            raise ValueError("base_strength must be within min_strength and max_strength")
        if self.max_response_ms < self.min_response_ms:
            raise ValueError("max_response_ms must be >= min_response_ms")
        return self


def _http_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    if code == "config_version_not_found":
        return HTTPException(status_code=404, detail=code)
    if code == "unsupported_game":
        return HTTPException(status_code=404, detail=code)
    return HTTPException(status_code=422, detail=code)


@router.get("")
async def list_game_configs():
    return await balance.list_admin_configs()


class GameVisibilityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


@router.put("/{game_key}")
async def update_game_config(game_key: str, payload: GameConfigPayload):
    try:
        return await balance.publish_config(game_key, payload.model_dump())
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.put("/{game_key}/visibility")
async def set_game_visibility(game_key: str, payload: GameVisibilityPayload):
    """Show/hide a game in the client hub; does not create a config version."""
    try:
        return await balance.set_enabled(game_key, payload.enabled)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.get("/{game_key}/versions")
async def list_game_config_versions(
    game_key: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    try:
        return await balance.list_versions(game_key, limit)
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post("/{game_key}/versions/{version}/restore")
async def restore_game_config_version(game_key: str, version: int):
    try:
        stored = await balance.get_version(game_key, version)
        payload = GameConfigPayload.model_validate(
            {key: stored[key] for key in GameConfigPayload.model_fields if key in stored}
        )
        return await balance.publish_config(game_key, payload.model_dump())
    except ValueError as exc:
        raise _http_error(exc) from exc
