from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.models.store_inventory import (
    StoreExchangeRequest,
    StoreExchangeResponse,
    StoreInventoryResponse,
)
from app.services import store_inventory

router = APIRouter(prefix="/store", tags=["store"])


@router.get("/inventory", response_model=StoreInventoryResponse)
async def get_store_inventory(payload: dict = Depends(require_user)):
    return await store_inventory.list_inventory(str(payload["sub"]))


@router.post("/exchange", response_model=StoreExchangeResponse)
async def exchange_store_product(
    data: StoreExchangeRequest,
    payload: dict = Depends(require_user),
):
    try:
        return await store_inventory.exchange_product(
            str(payload["sub"]),
            data.product_kind,
        )
    except ValueError as exc:
        if str(exc) == "unknown_product":
            raise HTTPException(status_code=404, detail="Unknown product") from exc
        if str(exc) == "insufficient_point_balance":
            raise HTTPException(status_code=409, detail="Insufficient point balance") from exc
        raise HTTPException(status_code=400, detail="Invalid exchange request") from exc
