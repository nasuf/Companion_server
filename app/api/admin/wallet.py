from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.admin_wallet import (
    AdminPointGrantRequest,
    AdminPointGrantResponse,
    AdminTicketGrantRequest,
    AdminTicketGrantResponse,
    AdminWalletBalancesResponse,
    AdminWalletLedgerItem,
    AdminWalletUserSearchItem,
)
from app.observability.events import EVT_ADMIN_POINT_GRANT, EVT_ADMIN_TICKET_GRANT
from app.services import game_points, wallet

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
