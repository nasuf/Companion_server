from __future__ import annotations

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
