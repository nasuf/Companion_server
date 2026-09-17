from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.vip_activation import (
    AdminVipCodeCreateRequest,
    AdminVipCodeEnabledRequest,
    AdminVipCodeItem,
    AdminVipCodeListResponse,
    AdminVipRedemptionListResponse,
)
from app.services.vip.activation_codes import admin as activation_admin
from app.services.vip.activation_codes.errors import VipActivationError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/vip-activation",
    tags=["admin", "vip-activation"],
    dependencies=[Depends(require_admin_jwt)],
)


def _parse_optional_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid_datetime") from exc


@router.get("/codes", response_model=AdminVipCodeListResponse)
async def list_codes(
    q: str | None = Query(default=None),
    enabled: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await activation_admin.list_codes(
        q=q, enabled=enabled, limit=limit, offset=offset
    )


@router.post("/codes", response_model=list[AdminVipCodeItem])
async def create_codes(
    data: AdminVipCodeCreateRequest,
    payload: dict = Depends(require_admin_jwt),
):
    admin_id = str(payload["sub"])
    max_redemptions = None if data.unlimited_redemptions else data.max_redemptions
    try:
        return await activation_admin.create_codes(
            duration_days=data.duration_days,
            count=data.count,
            max_redemptions=max_redemptions,
            valid_from=_parse_optional_dt(data.valid_from),
            valid_until=_parse_optional_dt(data.valid_until),
            note=data.note,
            created_by=admin_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/codes/{code_id}", response_model=AdminVipCodeItem)
async def patch_code(code_id: str, data: AdminVipCodeEnabledRequest):
    try:
        return await activation_admin.set_code_enabled(code_id, enabled=data.enabled)
    except VipActivationError as exc:
        if exc.code == "not_found":
            raise HTTPException(status_code=404, detail=exc.code) from exc
        raise HTTPException(status_code=400, detail=exc.code) from exc


@router.get("/redemptions", response_model=AdminVipRedemptionListResponse)
async def list_redemptions(
    user_id: str | None = Query(default=None),
    code_id: str | None = Query(default=None),
    code_q: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await activation_admin.list_redemptions(
        user_id=user_id,
        code_id=code_id,
        code_q=code_q,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.post("/redemptions/{redemption_id}/revoke")
async def revoke_redemption(redemption_id: str, payload: dict = Depends(require_admin_jwt)):
    admin_id = str(payload["sub"])
    try:
        return await activation_admin.revoke_redemption(
            redemption_id, admin_user_id=admin_id
        )
    except VipActivationError as exc:
        status = 404 if exc.code == "not_found" else 409
        raise HTTPException(status_code=status, detail=exc.code) from exc
