from __future__ import annotations

from pydantic import BaseModel, Field


class VipCodeRedeemRequest(BaseModel):
    code: str = Field(min_length=4, max_length=64)


class VipCodeRedemptionItem(BaseModel):
    id: str
    code_id: str | None = None
    duration_days: int
    redeemed_at: str
    effective_start: str | None = None
    effective_end: str | None = None


class VipCodeRedeemResponse(BaseModel):
    vip: dict
    redemption: VipCodeRedemptionItem


class AdminVipCodeCreateRequest(BaseModel):
    duration_days: int = Field(gt=0, le=3650)
    count: int = Field(default=1, ge=1, le=500)
    max_redemptions: int | None = Field(default=1, ge=1)
    unlimited_redemptions: bool = False
    valid_from: str | None = None
    valid_until: str | None = None
    note: str | None = Field(default=None, max_length=500)


class AdminVipCodeItem(BaseModel):
    id: str
    code: str
    duration_days: int
    max_redemptions: int | None
    redemption_count: int
    enabled: bool
    valid_from: str | None = None
    valid_until: str | None = None
    note: str | None = None
    created_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class AdminVipCodeListResponse(BaseModel):
    items: list[AdminVipCodeItem]
    total: int
    limit: int
    offset: int


class AdminVipCodeEnabledRequest(BaseModel):
    enabled: bool


class AdminVipRedemptionItem(BaseModel):
    id: str
    code_id: str
    code: str
    code_preview: str
    user_id: str
    user_display_name: str | None = None
    user_email: str | None = None
    duration_days: int
    status: str
    redeemed_at: str | None = None
    effective_start: str | None = None
    effective_end: str | None = None
    revoked_at: str | None = None
    revoked_by: str | None = None


class AdminVipRedemptionListResponse(BaseModel):
    items: list[AdminVipRedemptionItem]
    total: int
    limit: int
    offset: int
