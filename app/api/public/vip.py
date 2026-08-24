from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.models.vip import (
    ChatQuotaResponse,
    MusicQuotaReportRequest,
    MusicQuotaReportResponse,
    VipStatusResponse,
)
from app.services import wallet
from app.services.vip import chat_quota, music_quota

router = APIRouter(tags=["vip"])


@router.get("/me/vip", response_model=VipStatusResponse)
async def get_vip_status(payload: dict = Depends(require_user)):
    snapshot = await wallet.full_wallet(str(payload["sub"]))
    return snapshot


@router.get("/chat/quota", response_model=ChatQuotaResponse)
async def get_chat_quota(payload: dict = Depends(require_user)):
    user_id = str(payload["sub"])
    is_vip = await wallet.is_vip(user_id)
    return await chat_quota.preview(user_id, is_vip=is_vip)


@router.post("/music/quota/report", response_model=MusicQuotaReportResponse)
async def report_music_quota(
    data: MusicQuotaReportRequest,
    payload: dict = Depends(require_user),
):
    user_id = str(payload["sub"])
    is_vip = await wallet.is_vip(user_id)
    try:
        return await music_quota.report(
            user_id,
            is_vip=is_vip,
            delta_seconds=data.delta_seconds,
            paid_confirmed=data.paid_confirmed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid quota report") from exc
