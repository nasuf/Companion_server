from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_admin_jwt
from app.models.admin_chat_quota import (
    AdminChatQuotaResetRequest,
    AdminChatQuotaStatusResponse,
)
from app.observability.events import EVT_ADMIN_CHAT_QUOTA_RESET
from app.services import wallet
from app.services.vip import chat_quota

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/chat-quota",
    tags=["admin", "chat-quota"],
    dependencies=[Depends(require_admin_jwt)],
)


@router.get("/status", response_model=AdminChatQuotaStatusResponse)
async def get_chat_quota_status(user_id: str):
    is_vip = await wallet.is_vip(user_id)
    status = await chat_quota.preview(user_id, is_vip=is_vip)
    return {"user_id": user_id, "is_vip": is_vip, **status}


@router.post("/reset", response_model=AdminChatQuotaStatusResponse)
async def reset_chat_quota(
    payload: AdminChatQuotaResetRequest,
    claims: dict = Depends(require_admin_jwt),
):
    admin_id = str(claims.get("sub") or "")
    is_vip = await wallet.is_vip(payload.user_id)
    try:
        status = await chat_quota.admin_reset(payload.user_id, is_vip=is_vip)
    except Exception as exc:  # defensive: unexpected DB error, not a known ValueError
        raise HTTPException(status_code=500, detail="reset_failed") from exc
    logger.info(
        "admin chat quota reset user=%s period=%s/%s",
        payload.user_id[:8],
        status["period_scope"],
        status["period_key"],
        extra={
            "event": EVT_ADMIN_CHAT_QUOTA_RESET,
            "admin_id": admin_id,
            "target_user_id": payload.user_id,
            "period_scope": status["period_scope"],
            "period_key": status["period_key"],
            "note": (payload.note or "").strip(),
        },
    )
    return {"user_id": payload.user_id, "is_vip": is_vip, **status}
