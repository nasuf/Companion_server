from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.models.store_inventory import (
    StoreBundlePurchaseRequest,
    StoreBundlePurchaseResponse,
    StoreCatalogResponse,
    StoreExchangeRequest,
    StoreExchangeResponse,
    StoreInventoryResponse,
)
from app.services import store_bundles, store_inventory

router = APIRouter(prefix="/store", tags=["store"])


@router.get("/inventory", response_model=StoreInventoryResponse)
async def get_store_inventory(payload: dict = Depends(require_user)):
    return await store_inventory.list_inventory(str(payload["sub"]))


@router.get("/catalog", response_model=StoreCatalogResponse)
async def get_store_catalog(payload: dict = Depends(require_user)):
    return await store_inventory.get_catalog(str(payload["sub"]))


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


@router.post("/bundles", response_model=StoreBundlePurchaseResponse)
async def purchase_store_bundle(
    data: StoreBundlePurchaseRequest,
    payload: dict = Depends(require_user),
):
    try:
        return await store_bundles.purchase_bundle(
            str(payload["sub"]),
            data.bundle_kind,
            tier_id=data.tier_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "unknown_bundle" or detail == "unknown_tier":
            raise HTTPException(status_code=404, detail="Unknown bundle") from exc
        if detail == "insufficient_ticket_balance":
            raise HTTPException(status_code=409, detail="Insufficient ticket balance") from exc
        if detail == "payment_required":
            raise HTTPException(
                status_code=402,
                detail="WeChat payment is not connected yet",
            ) from exc
        if detail == "vip_trial_used":
            raise HTTPException(status_code=409, detail="VIP trial already used") from exc
        raise HTTPException(status_code=400, detail="Invalid bundle purchase") from exc
