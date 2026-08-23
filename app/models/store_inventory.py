from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.wallet import WalletBalanceResponse


class StoreInventoryItem(BaseModel):
    product_kind: str
    quantity: int
    acquired_at: str | None = None
    updated_at: str | None = None


class StoreInventoryResponse(BaseModel):
    items: list[StoreInventoryItem] = Field(default_factory=list)


class StoreExchangeRequest(BaseModel):
    product_kind: str


class StoreExchangeResponse(BaseModel):
    wallet: WalletBalanceResponse
    inventory_item: StoreInventoryItem


class StoreCatalogProduct(BaseModel):
    product_kind: str
    title: str
    member_price: int
    list_price: int
    price: int
    category: str
    subcategory: str | None = None
    contents: str | None = None


class StoreCatalogResponse(BaseModel):
    is_vip: bool
    vip_trial_available: bool
    products: list[StoreCatalogProduct] = Field(default_factory=list)
    bundles: dict[str, Any] = Field(default_factory=dict)


class StoreBundlePurchaseRequest(BaseModel):
    bundle_kind: Literal["music_coupon", "game_points", "makeup_card", "vip_trial"]
    tier_id: str | None = None


class StoreBundlePurchaseResponse(BaseModel):
    wallet: WalletBalanceResponse
    inventory_item: StoreInventoryItem | None = None
    game_balance: int | None = None
    vip_until: str | None = None
