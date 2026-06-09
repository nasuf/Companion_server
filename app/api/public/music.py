from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app.api.jwt_auth import require_user
from app.models.music import (
    MusicCoListeningEndRequest,
    MusicCoListeningResponse,
    MusicConfigResponse,
    MusicFavoriteRequest,
    MusicFavoriteResponse,
    MusicLibrariesResponse,
    MusicPlaybackRequest,
    MusicPlaybackResponse,
    MusicTrackPlayUrlResponse,
    MusicTracksResponse,
)
from app.services import music
from app.services import music_status
from app.services.runtime.tasks import fire_background

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/music", tags=["music"])

_STREAM_TIMEOUT = httpx.Timeout(12.0, connect=4.0, read=8.0, write=4.0, pool=4.0)
_STREAM_REQUEST_HEADERS = {
    "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.5",
    "User-Agent": "CompanionMusicProxy/1.0",
}


def _handle_value_error(exc: ValueError) -> None:
    message = str(exc)
    if message == "agent_not_found":
        raise HTTPException(status_code=404, detail="Agent not found") from exc
    if message == "workspace_not_found":
        raise HTTPException(status_code=404, detail="Workspace not found") from exc
    if message == "conversation_not_found":
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    if message == "track_not_found":
        raise HTTPException(status_code=404, detail="Track not found") from exc
    raise exc


@router.get("/config", response_model=MusicConfigResponse)
async def get_music_config(_: dict = Depends(require_user)):
    return MusicConfigResponse(
        provider="jamendo",
        api_enabled=music.api_enabled(),
        default_libraries=music.default_libraries(),
        missing_config=music.missing_config(),
    )


@router.get("/libraries", response_model=MusicLibrariesResponse)
async def list_music_libraries(_: dict = Depends(require_user)):
    default_library = music.default_libraries()[0]
    return MusicLibrariesResponse(
        libraries=music.libraries(),
        provider="jamendo",
        default_library=default_library,
    )


@router.get("/tracks", response_model=MusicTracksResponse)
async def list_music_tracks(
    agent_id: str,
    workspace_id: str | None = None,
    library: str | None = None,
    limit: int = Query(default=music.DEFAULT_TRACK_LIMIT, ge=1, le=12),
    refresh: bool = False,
    exclude_track_id: str | None = None,
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
            exclude_track_ids={exclude_track_id} if exclude_track_id else None,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    return MusicTracksResponse(
        tracks=tracks,
        provider="jamendo",
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
        provider="jamendo",
        api_enabled=music.api_enabled(),
        cache_ttl_seconds=music.AUDIO_CACHE_TTL_SECONDS,
    )


@router.get("/tracks/{track_id}/play-url", response_model=MusicTrackPlayUrlResponse)
async def get_music_track_play_url(
    track_id: str,
    agent_id: str,
    user: dict = Depends(require_user),
):
    try:
        url = await music.resolve_play_url(
            user_id=user["sub"],
            agent_id=agent_id,
            track_id=track_id,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    return MusicTrackPlayUrlResponse(track_id=track_id, url=url)


@router.get("/tracks/{track_id}/stream.mp3")
async def stream_music_track(track_id: str, token: str, request: Request):
    if not music.validate_stream_token(track_id, token):
        raise HTTPException(status_code=403, detail="Invalid stream token")
    audio_urls = await music.resolve_stream_audio_urls(track_id)
    if not audio_urls:
        raise HTTPException(status_code=404, detail="Track stream not found")
    request_start = time.monotonic()
    request_id = f"{track_id}:{int(request_start * 1000) % 1_000_000}"
    client = httpx.AsyncClient(
        timeout=_STREAM_TIMEOUT,
        follow_redirects=True,
        trust_env=False,
    )
    upstream = None
    stream_context = None
    last_error: Exception | None = None
    range_header = request.headers.get("range") or "bytes=0-"
    upstream_headers = {**_STREAM_REQUEST_HEADERS, "Range": range_header}
    logger.info(
        "[music.stream] start request_id=%s track_id=%s client_range=%s candidates=%s",
        request_id,
        track_id,
        range_header,
        [_stream_candidate_kind(url) for url in audio_urls],
    )
    for audio_url in audio_urls:
        candidate_started = time.monotonic()
        candidate_kind = _stream_candidate_kind(audio_url)
        stream_context = client.stream("GET", audio_url, headers=upstream_headers)
        try:
            upstream = await stream_context.__aenter__()
        except httpx.HTTPError as exc:
            last_error = exc
            stream_context = None
            logger.warning(
                "[music.stream] candidate_error request_id=%s track_id=%s kind=%s elapsed_ms=%d error=%s",
                request_id,
                track_id,
                candidate_kind,
                _elapsed_ms(candidate_started),
                exc.__class__.__name__,
            )
            continue
        logger.info(
            "[music.stream] candidate_response request_id=%s track_id=%s kind=%s "
            "status=%s elapsed_ms=%d content_type=%s content_range=%s content_length=%s",
            request_id,
            track_id,
            candidate_kind,
            upstream.status_code,
            _elapsed_ms(candidate_started),
            upstream.headers.get("content-type"),
            upstream.headers.get("content-range"),
            upstream.headers.get("content-length"),
        )
        if upstream.status_code < 400:
            break
        await stream_context.__aexit__(None, None, None)
        upstream = None
        stream_context = None
    if upstream is None or stream_context is None:
        await client.aclose()
        raise HTTPException(status_code=502, detail="Music stream unavailable") from last_error
    headers = {
        "Content-Type": "audio/mpeg",
        "Accept-Ranges": upstream.headers.get("accept-ranges") or "bytes",
        "Cache-Control": "private, max-age=900",
    }
    for name in ("content-length", "content-range", "etag", "last-modified"):
        value = upstream.headers.get(name)
        if value:
            headers[name.title()] = value

    async def _logged_audio_chunks():
        total_bytes = 0
        first_chunk_seen = False
        try:
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    total_bytes += len(chunk)
                    if not first_chunk_seen:
                        first_chunk_seen = True
                        logger.info(
                            "[music.stream] first_chunk request_id=%s track_id=%s "
                            "elapsed_ms=%d bytes=%d",
                            request_id,
                            track_id,
                            _elapsed_ms(request_start),
                            len(chunk),
                        )
                yield chunk
            logger.info(
                "[music.stream] complete request_id=%s track_id=%s elapsed_ms=%d bytes=%d",
                request_id,
                track_id,
                _elapsed_ms(request_start),
                total_bytes,
            )
        except Exception:
            logger.exception(
                "[music.stream] stream_error request_id=%s track_id=%s elapsed_ms=%d bytes=%d",
                request_id,
                track_id,
                _elapsed_ms(request_start),
                total_bytes,
            )
            raise

    async def _close_stream():
        await stream_context.__aexit__(None, None, None)
        await client.aclose()

    return StreamingResponse(
        _logged_audio_chunks(),
        status_code=upstream.status_code,
        headers=headers,
        background=BackgroundTask(_close_stream),
        media_type="audio/mpeg",
    )


def _stream_candidate_kind(url: str) -> str:
    if "/download/track/" in url:
        return "download"
    return "stream"


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


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
            conversation_id=data.conversation_id,
            payload=data.track,
            position_seconds=data.position_seconds,
            is_playing=data.is_playing,
        )
    except ValueError as exc:
        _handle_value_error(exc)
    if data.conversation_id and not data.is_playing:
        fire_background(
            music_status.end_if_paused_after_timeout(
                user_id=user["sub"],
                agent_id=data.agent_id,
                conversation_id=data.conversation_id,
            )
        )
    return MusicPlaybackResponse(
        track=track,
        position_seconds=position_seconds,
        is_playing=is_playing,
        updated_at=updated_at,
    )


@router.post("/co-listening/end", response_model=MusicCoListeningResponse)
async def end_music_co_listening(
    data: MusicCoListeningEndRequest,
    user: dict = Depends(require_user),
):
    try:
        ended = await music.end_co_listening(
            user_id=user["sub"],
            agent_id=data.agent_id,
            conversation_id=data.conversation_id,
            reason=data.reason,
        )
        if ended is not None and ended.initiated_by != "agent_auto":
            await music_status.persist_and_emit_music_status(
                conversation_id=data.conversation_id,
                status="ended",
                track=ended.track,
                reason=data.reason,
            )
    except ValueError as exc:
        _handle_value_error(exc)
    return ended or MusicCoListeningResponse(status="ended")
