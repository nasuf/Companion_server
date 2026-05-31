from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.config import settings
from app.models.game import (
    SudCallbackGetSsTokenRequest,
    SudCallbackGetUserInfoRequest,
    SudCallbackReportGameInfoRequest,
    SudCallbackUpdateSsTokenRequest,
    SudConfigResponse,
    SudCreateSessionRequest,
    SudGameEventRequest,
    SudGameEventResponse,
    SudSessionResponse,
)
from app.services.games import sud

router = APIRouter(prefix="/games/sud", tags=["games"])


def _callback_base() -> str:
    base = settings.sud_callback_public_base_url.strip().rstrip("/")
    return f"{base}/games/sud/callback" if base else "/games/sud/callback"


@router.get("/config", response_model=SudConfigResponse)
async def get_sud_config(_: dict = Depends(require_user)):
    base = _callback_base()
    return SudConfigResponse(
        sdk_enabled=sud.sdk_enabled(),
        app_id=settings.sud_app_id.strip(),
        app_key=settings.sud_app_key.strip(),
        bundle_id=settings.sud_bundle_id.strip() or "com.companion.app",
        is_test_env=settings.sud_is_test_env,
        default_mg_id=settings.sud_default_mg_id.strip(),
        missing_config=sud.missing_config(),
        callbacks={
            "get_sstoken": f"{base}/get_sstoken",
            "get_user_info": f"{base}/get_user_info",
            "report_game_info": f"{base}/report_game_info",
        },
    )


@router.get("/sessions", response_model=list[SudSessionResponse])
async def list_sud_sessions(user: dict = Depends(require_user)):
    return await sud.list_sessions(user["sub"])


@router.post("/sessions", response_model=SudSessionResponse)
async def create_sud_session(
    data: SudCreateSessionRequest,
    user: dict = Depends(require_user),
):
    try:
        return await sud.create_session(
            user_id=user["sub"],
            agent_id=data.agent_id,
            workspace_id=data.workspace_id,
            conversation_id=data.conversation_id,
            mg_id=data.mg_id,
            room_id=data.room_id,
            play_mode=data.play_mode,
            difficulty=data.difficulty,
        )
    except ValueError as exc:
        if str(exc) == "agent_not_found":
            raise HTTPException(status_code=404, detail="Agent not found") from exc
        if str(exc) == "context_not_found":
            raise HTTPException(status_code=404, detail="Game context not found") from exc
        raise


@router.post("/sessions/{session_id}/code", response_model=SudSessionResponse)
async def refresh_sud_session_code(
    session_id: str,
    user: dict = Depends(require_user),
):
    try:
        return await sud.refresh_code(session_id, user_id=user["sub"])
    except ValueError as exc:
        if str(exc) == "session_not_found":
            raise HTTPException(status_code=404, detail="Game session not found") from exc
        raise


@router.post("/sessions/{session_id}/events", response_model=SudGameEventResponse)
async def append_sud_game_event(
    session_id: str,
    data: SudGameEventRequest,
    user: dict = Depends(require_user),
):
    try:
        session, reply, event_id = await sud.handle_client_event(
            session_id=session_id,
            user_id=user["sub"],
            event_type=data.event_type,
            state=data.state,
            payload=data.payload,
            source=data.source,
        )
    except ValueError as exc:
        if str(exc) == "session_not_found":
            raise HTTPException(status_code=404, detail="Game session not found") from exc
        raise
    return SudGameEventResponse(
        session=session,
        companion_reply=reply,
        persisted_event_id=event_id,
    )


@router.post("/callback/get_sstoken")
async def callback_get_sstoken(data: SudCallbackGetSsTokenRequest):
    try:
        ss_token, expires_at, user_info, _ = await sud.user_info_from_code(data.code)
    except Exception as exc:
        return {
            "ret_code": 1,
            "ret_msg": f"invalid code: {exc}",
            "sdk_error_code": 1005,
            "data": {},
        }
    return {
        "ret_code": 0,
        "ret_msg": "",
        "sdk_error_code": 0,
        "data": {
            "ss_token": ss_token,
            "expire_date": sud.expire_ms(expires_at),
            "expire_date_str": str(sud.expire_ms(expires_at)),
            "user_info": user_info.model_dump(),
        },
    }


@router.post("/callback/update_sstoken")
async def callback_update_sstoken(data: SudCallbackUpdateSsTokenRequest):
    try:
        ss_token, expires_at, user_info, _ = await sud.refresh_ss_token(data.ss_token)
    except Exception as exc:
        return {
            "ret_code": 1,
            "ret_msg": f"invalid ss_token: {exc}",
            "sdk_error_code": 1005,
            "data": {},
        }
    return {
        "ret_code": 0,
        "ret_msg": "",
        "sdk_error_code": 0,
        "data": {
            "ss_token": ss_token,
            "expire_date": sud.expire_ms(expires_at),
            "expire_date_str": str(sud.expire_ms(expires_at)),
            "user_info": user_info.model_dump(),
        },
    }


@router.post("/callback/get_user_info")
async def callback_get_user_info(data: SudCallbackGetUserInfoRequest):
    try:
        user_info = await sud.user_info_from_token(data.ss_token)
    except Exception as exc:
        return {
            "ret_code": 1,
            "ret_msg": f"invalid token: {exc}",
            "sdk_error_code": 1005,
            "data": {},
        }
    return {
        "ret_code": 0,
        "ret_msg": "",
        "sdk_error_code": 0,
        "data": user_info.model_dump(),
    }


@router.post("/callback/report_game_info")
async def callback_report_game_info(data: SudCallbackReportGameInfoRequest):
    if data.ss_token:
        try:
            sud.decode_token(data.ss_token)
        except Exception:
            return {
                "ret_code": 1,
                "ret_msg": "invalid ss_token",
                "sdk_error_code": 1005,
                "data": {},
            }
    session = await sud.handle_sud_report(data.report_type, data.report_msg)
    return {
        "ret_code": 0,
        "ret_msg": "",
        "sdk_error_code": 0,
        "data": {
            "session_id": session.id if session else None,
            "received_at": datetime.now(UTC).isoformat(),
        },
    }
