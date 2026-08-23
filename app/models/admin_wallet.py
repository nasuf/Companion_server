from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AdminWalletBalanceItem(BaseModel):
    user_id: str
    username: str
    display_name: str | None = None
    nickname: str | None = None
    ticket_balance: int
    point_balance: int
    updated_at: str | None = None


class AdminWalletBalancesResponse(BaseModel):
    items: list[AdminWalletBalanceItem]
    total: int


class AdminWalletLedgerItem(BaseModel):
    id: str
    user_id: str
    username: str | None = None
    display_name: str | None = None
    nickname: str | None = None
    currency: str
    delta: int
    balance_after: int
    source: str
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AdminWalletUserSearchItem(BaseModel):
    user_id: str
    username: str
    nickname: str | None = None


class AdminTicketGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    # Positive adds tickets; negative deducts (floored at 0). 0 is rejected in
    # the service as invalid_amount. Magnitude capped to match user-facing sends.
    amount: int = Field(ge=-1_000_000, le=1_000_000)
    note: str | None = Field(default=None, max_length=200)


class AdminTicketGrantResponse(BaseModel):
    user_id: str
    ticket_balance: int
    point_balance: int
    achievement_points_synced: int
    delta: int


class AdminPointGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    amount: int = Field(ge=-1_000_000, le=1_000_000)
    note: str | None = Field(default=None, max_length=200)


class AdminPointGrantResponse(BaseModel):
    user_id: str
    ticket_balance: int
    point_balance: int
    achievement_points_synced: int
    delta: int
