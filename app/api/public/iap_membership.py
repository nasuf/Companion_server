from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_user
from app.models.iap import IapMembershipResponse
from app.services.payments import membership

router = APIRouter(prefix="/me/iap", tags=["iap"])


@router.get("/membership", response_model=IapMembershipResponse)
async def get_iap_membership(
    payload: dict = Depends(require_user),
    history_limit: int = Query(default=50, ge=1, le=100),
):
    """VIP 状态 + 连续包月订阅态 + 会员购买历史（订阅页 / 会员记录页共用）。"""
    user_id = str(payload["sub"])
    snapshot = await membership.get_membership(user_id, history_limit=history_limit)
    return snapshot
