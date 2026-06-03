from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.models.music import (
    MusicConfigResponse,
    MusicFavoriteRequest,
    MusicFavoriteResponse,
    MusicLibrariesResponse,
    MusicPlaybackRequest,
    MusicPlaybackResponse,
    MusicTracksResponse,
)
from app.services import music

router = APIRouter(prefix="/music", tags=["music"])


def _handle_value_error(exc: ValueError) -> None:
    message = str(exc)
    if message == "agent_not_found":
        raise HTTPException(status_code=404, detail="Agent not found") from exc
    if message == "workspace_not_found":
        raise HTTPException(status_code=404, detail="Workspace not found") from exc
    raise exc


@router.get("/config", response_model=MusicConfigResponse)
async def get_music_config(_: dict = Depends(require_user)):
    return MusicConfigResponse(
        api_enabled=music.api_enabled(),
        default_libraries=music.default_libraries(),
        missing_config=music.missing_config(),
    )


@router.get("/libraries", response_model=MusicLibrariesResponse)
async def list_music_libraries(_: dict = Depends(require_user)):
    default_library = music.default_libraries()[0]
    return MusicLibrariesResponse(
        libraries=music.libraries(),
        default_library=default_library,
    )


@router.get("/tracks", response_model=MusicTracksResponse)
async def list_music_tracks(
    agent_id: str,
    workspace_id: str | None = None,
    library: str | None = None,
    limit: int = Query(default=music.DEFAULT_TRACK_LIMIT, ge=1, le=12),
    refresh: bool = False,
    user: dict = Depends(require_user),
):
    try:
        tracks = await music.list_square_tracks(
            user_id=user["sub"],
            agent_id=agent_id,
            workspace_id=workspace_id,
            library=library,
            limit=limit,
            refresh=refresh,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    return MusicTracksResponse(
        tracks=tracks,
        api_enabled=music.api_enabled(),
        library=library,
        cache_ttl_seconds=music.AUDIO_CACHE_TTL_SECONDS,
    )


@router.get("/favorites", response_model=MusicTracksResponse)
async def list_music_favorites(agent_id: str, user: dict = Depends(require_user)):
    try:
        tracks = await music.list_favorites(user_id=user["sub"], agent_id=agent_id)
    except ValueError as exc:
        _handle_value_error(exc)
    return MusicTracksResponse(
        tracks=tracks,
        api_enabled=music.api_enabled(),
        cache_ttl_seconds=music.AUDIO_CACHE_TTL_SECONDS,
    )


@router.post("/favorites", response_model=MusicFavoriteResponse)
async def add_music_favorite(
    data: MusicFavoriteRequest,
    user: dict = Depends(require_user),
):
    try:
        track = await music.add_favorite(
            user_id=user["sub"],
            agent_id=data.agent_id,
            workspace_id=data.workspace_id,
            payload=data.track,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    return MusicFavoriteResponse(track=track)


@router.delete("/favorites/{track_id}")
async def remove_music_favorite(
    track_id: str,
    agent_id: str,
    user: dict = Depends(require_user),
):
    try:
        await music.remove_favorite(
            user_id=user["sub"],
            agent_id=agent_id,
            track_id=track_id,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    return {"ok": True}


@router.get("/now-playing", response_model=MusicPlaybackResponse)
async def get_music_now_playing(agent_id: str, user: dict = Depends(require_user)):
    try:
        track, position_seconds, is_playing, updated_at = await music.get_now_playing(
            user_id=user["sub"],
            agent_id=agent_id,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    return MusicPlaybackResponse(
        track=track,
        position_seconds=position_seconds,
        is_playing=is_playing,
        updated_at=updated_at,
    )


@router.post("/now-playing", response_model=MusicPlaybackResponse)
async def update_music_now_playing(
    data: MusicPlaybackRequest,
    user: dict = Depends(require_user),
):
    try:
        track, position_seconds, is_playing, updated_at = await music.upsert_now_playing(
            user_id=user["sub"],
            agent_id=data.agent_id,
            workspace_id=data.workspace_id,
            payload=data.track,
            position_seconds=data.position_seconds,
            is_playing=data.is_playing,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    return MusicPlaybackResponse(
        track=track,
        position_seconds=position_seconds,
        is_playing=is_playing,
        updated_at=updated_at,
    )
