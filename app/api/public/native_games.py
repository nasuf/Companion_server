from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.api.jwt_auth import require_user
from app.models.game import (
    NativeCreateSessionRequest,
    NativeGameEventRecord,
    NativeGameEventRequest,
    NativeGameEventResponse,
    NativeSessionResponse,
)
from app.services.games import native

router = APIRouter(prefix="/games/native", tags=["games"])


def _http_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    if code in {"agent_not_found", "context_not_found", "session_not_found"}:
        return HTTPException(status_code=404, detail=code)
    if code == "daily_points_exhausted":
        # No game points left today; the client shows a "come back tomorrow" hint.
        return HTTPException(status_code=403, detail=code)
    if code in {
        "unsupported_game",
        "unsupported_event",
        "invalid_state",
        "invalid_turn",
        "invalid_move",
        "occupied_position",
        "game_not_finished",
        "invalid_outcome",
        "invalid_move_history",
        "invalid_action_history",
        "invalid_actor",
        "invalid_state_hash",
        "missing_terminal_state",
        "game_already_finished",
        "session_finished",
    }:
        return HTTPException(status_code=409, detail=code)
    return HTTPException(status_code=400, detail=code)


@router.get("/sessions", response_model=list[NativeSessionResponse])
async def list_native_game_sessions(
    game_key: str | None = Query(default=None, max_length=40),
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(require_user),
):
    return await native.list_sessions(
        user["sub"],
        game_key=game_key,
        limit=limit,
    )


@router.get("/sessions/latest")
async def latest_native_game_session(
    agent_id: str | None = Query(default=None),
    user: dict = Depends(require_user),
):
    """Latest terminal session summary (status/game_key) for the game hub."""
    return await native.get_latest_session_summary(
        user["sub"],
        agent_id=agent_id,
    )


@router.post("/sessions", response_model=NativeSessionResponse)
async def create_native_game_session(
    data: NativeCreateSessionRequest,
    user: dict = Depends(require_user),
):
    try:
        return await native.create_session(
            user_id=user["sub"],
            agent_id=data.agent_id,
            workspace_id=data.workspace_id,
            conversation_id=data.conversation_id,
            game_key=data.game_key,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_native_game_session(
    session_id: str,
    user: dict = Depends(require_user),
):
    try:
        await native.delete_session(session_id, user_id=user["sub"])
    except ValueError as exc:
        raise _http_error(exc) from exc
    return Response(status_code=204)


@router.get(
    "/sessions/{session_id}/events",
    response_model=list[NativeGameEventRecord],
)
async def list_native_game_events(
    session_id: str,
    limit: int = Query(default=500, ge=1, le=1000),
    user: dict = Depends(require_user),
):
    try:
        return await native.list_events(
            session_id,
            user_id=user["sub"],
            limit=limit,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/sessions/{session_id}/events",
    response_model=NativeGameEventResponse,
)
async def append_native_game_event(
    session_id: str,
    data: NativeGameEventRequest,
    user: dict = Depends(require_user),
):
    try:
        session, reply, event_id, duplicate = await native.handle_event(
            session_id=session_id,
            user_id=user["sub"],
            event_type=data.event_type,
            state=data.state,
            payload=data.payload,
            source=data.source,
            client_event_id=data.client_event_id,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return NativeGameEventResponse(
        session=session,
        companion_reply=reply,
        persisted_event_id=event_id,
        duplicate=duplicate,
    )
