from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.models.wallet import (
    WalletBalanceResponse,
    WalletExchangeRequest,
    WalletLedgerItem,
)
from app.services import wallet

router = APIRouter(prefix="/wallet", tags=["wallet"])


@router.get("", response_model=WalletBalanceResponse)
async def get_wallet(
    agent_id: str | None = Query(default=None),
    payload: dict = Depends(require_user),
):
    return await wallet.get_balance(str(payload["sub"]), agent_id=agent_id)


@router.post("/exchange", response_model=WalletBalanceResponse)
async def exchange_currency(
    data: WalletExchangeRequest,
    payload: dict = Depends(require_user),
):
    if data.from_currency != "ticket" or data.to_currency != "point":
        raise HTTPException(status_code=400, detail="Unsupported exchange pair")
    try:
        return await wallet.exchange_ticket_to_points(
            str(payload["sub"]),
            ticket_amount=data.ticket_amount,
        )
    except ValueError as exc:
        if str(exc) == "insufficient_ticket_balance":
            raise HTTPException(status_code=409, detail="Insufficient ticket balance") from exc
        raise HTTPException(status_code=400, detail="Invalid exchange amount") from exc


@router.get("/ledger", response_model=list[WalletLedgerItem])
async def get_wallet_ledger(
    currency: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    payload: dict = Depends(require_user),
):
    if currency not in {None, "ticket", "point"}:
        raise HTTPException(status_code=400, detail="Unsupported currency")
    return await wallet.list_ledger(
        str(payload["sub"]),
        currency=currency,
        limit=limit,
        offset=offset,
    )
