from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.admin_wallet import (
    AdminPointGrantRequest,
    AdminPointGrantResponse,
    AdminTicketGrantRequest,
    AdminTicketGrantResponse,
    AdminVipSetRequest,
    AdminVipSetResponse,
    AdminWalletBalancesResponse,
    AdminWalletLedgerItem,
    AdminWalletUserSearchItem,
)
from app.observability.events import (
    EVT_ADMIN_POINT_GRANT,
    EVT_ADMIN_TICKET_GRANT,
    EVT_ADMIN_VIP_SET,
)
from app.services import game_points, wallet
from app.services.vip import grants

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin-api/wallet",
    tags=["admin", "wallet"],
    dependencies=[Depends(require_admin_jwt)],
)


def _http_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    if code in {"user_not_found", "wallet_not_found"}:
        return HTTPException(status_code=404, detail=code)
    if code in {"invalid_amount", "no_change"}:
        return HTTPException(status_code=422, detail=code)
    return HTTPException(status_code=422, detail=code)


@router.get("/balances", response_model=AdminWalletBalancesResponse)
async def list_wallet_balances(
    search: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await wallet.list_admin_balances(
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get("/ledger", response_model=list[AdminWalletLedgerItem])
async def list_wallet_ledger(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await wallet.list_admin_ticket_ledger(
        user_id=user_id,
        limit=limit,
        offset=offset,
    )


@router.get("/point-ledger", response_model=list[AdminWalletLedgerItem])
async def list_wallet_point_ledger(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await wallet.list_admin_point_ledger(
        user_id=user_id,
        limit=limit,
        offset=offset,
    )


@router.get("/gift-ticket-ledger", response_model=list[AdminWalletLedgerItem])
async def list_wallet_gift_ticket_ledger(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await wallet.list_admin_gift_ticket_ledger(
        user_id=user_id,
        limit=limit,
        offset=offset,
    )


@router.get("/users", response_model=list[AdminWalletUserSearchItem])
async def search_grant_users(
    q: str = Query(default=""),
    limit: int = Query(default=20, ge=1, le=50),
):
    # Reuse the game-points fuzzy search (username / id / 微信昵称 / 手机号).
    return await game_points.search_users(q, limit=limit)


@router.post("/grant", response_model=AdminTicketGrantResponse)
async def grant_tickets(
    payload: AdminTicketGrantRequest,
    claims: dict = Depends(require_admin_jwt),
):
    admin_id = str(claims.get("sub") or "")
    try:
        result = await wallet.admin_adjust_tickets(
            payload.user_id,
            payload.amount,
            admin_id=admin_id,
            note=payload.note,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    logger.info(
        "admin ticket grant user=%s delta=%s",
        payload.user_id[:8],
        result["delta"],
        extra={
            "event": EVT_ADMIN_TICKET_GRANT,
            "admin_id": admin_id,
            "target_user_id": payload.user_id,
            "delta": result["delta"],
            "ticket_balance": result["ticket_balance"],
        },
    )
    return result


@router.post("/vip-set", response_model=AdminVipSetResponse)
async def set_vip_until(
    payload: AdminVipSetRequest,
    claims: dict = Depends(require_admin_jwt),
):
    admin_id = str(claims.get("sub") or "")
    parsed_until: datetime | None = None
    if payload.vip_until:
        try:
            parsed_until = datetime.fromisoformat(payload.vip_until.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_vip_until") from exc
        if parsed_until.tzinfo is None:
            parsed_until = parsed_until.replace(tzinfo=timezone.utc)
    try:
        result = await wallet.admin_set_vip_until(payload.user_id, parsed_until)
    except ValueError as exc:
        raise _http_error(exc) from exc
    if not result["is_vip"]:
        # 管理员主动结束/设过期 VIP 应立即生效 —— 不必等夜间 vip_expire_clear
        # cron 才清零限时钞票/失效礼包批次。clear_on_lapse 对已经是 0/无
        # vip_grant 批次的情况是安全的空操作，无条件调用不会误清正常用户。
        await grants.clear_on_lapse(payload.user_id)
    logger.info(
        "admin vip set user=%s vip_until=%s",
        payload.user_id[:8],
        result["vip_until"],
        extra={
            "event": EVT_ADMIN_VIP_SET,
            "admin_id": admin_id,
            "target_user_id": payload.user_id,
            "vip_until": result["vip_until"],
            "note": (payload.note or "").strip(),
        },
    )
    return result


@router.post("/point-grant", response_model=AdminPointGrantResponse)
async def grant_points(
    payload: AdminPointGrantRequest,
    claims: dict = Depends(require_admin_jwt),
):
    admin_id = str(claims.get("sub") or "")
    try:
        result = await wallet.admin_adjust_points(
            payload.user_id,
            payload.amount,
            admin_id=admin_id,
            note=payload.note,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    logger.info(
        "admin point grant user=%s delta=%s",
        payload.user_id[:8],
        result["delta"],
        extra={
            "event": EVT_ADMIN_POINT_GRANT,
            "admin_id": admin_id,
            "target_user_id": payload.user_id,
            "delta": result["delta"],
            "point_balance": result["point_balance"],
        },
    )
    return result
