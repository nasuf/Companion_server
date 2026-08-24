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
    # CLAUDE.md 权益项 3: VIP 每月赠送的限时钞票, 与永久 ticket_balance 分列
    # 展示（会随 VIP 过期清零）。
    gift_ticket_balance: int = 0
    is_vip: bool = False
    vip_until: str | None = None


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


class AdminVipSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    # ISO datetime string, or null to immediately end VIP (clears vip_until).
    # 到期即视为非 VIP（is_vip_from_row: vip_until > now），管理员延长只需传
    # 一个更晚的时间；本接口本身不触发每月发放/到期清零，那两件事仍由
    # jobs/scheduler.py 的 cron 按 vip_last_grant_at 锚点各自处理。
    vip_until: str | None = None
    note: str | None = Field(default=None, max_length=200)


class AdminVipSetResponse(BaseModel):
    user_id: str
    is_vip: bool
    vip_until: str | None = None
