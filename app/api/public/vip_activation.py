from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.models.vip_activation import VipCodeRedeemRequest, VipCodeRedeemResponse
from app.services.vip.activation_codes.errors import VipActivationError
from app.services.vip.activation_codes.redeem import redeem_code

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vip"])

_ERROR_STATUS = {
    "invalid_code": 404,
    "code_disabled": 403,
    "code_expired": 410,
    "code_exhausted": 409,
    "already_redeemed": 409,
}


@router.post("/me/vip/redeem-code", response_model=VipCodeRedeemResponse)
async def redeem_vip_code(
    data: VipCodeRedeemRequest,
    payload: dict = Depends(require_user),
):
    user_id = str(payload["sub"])
    try:
        result = await redeem_code(user_id, data.code)
    except VipActivationError as exc:
        status = _ERROR_STATUS.get(exc.code, 400)
        raise HTTPException(status_code=status, detail=exc.code) from exc
    return result
