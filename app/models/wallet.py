from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


WalletCurrency = Literal["ticket", "point"]


class WalletBalanceResponse(BaseModel):
    ticket_balance: int
    point_balance: int
    achievement_points_synced: int
    gift_ticket_balance: int = 0


class WalletExchangeRequest(BaseModel):
    from_currency: Literal["ticket"] = "ticket"
    to_currency: Literal["point"] = "point"
    ticket_amount: int = Field(gt=0)


class WalletLedgerItem(BaseModel):
    id: str
    currency: WalletCurrency
    delta: int
    balance_after: int
    source: str
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
