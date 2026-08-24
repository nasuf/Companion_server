from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AdminChatQuotaStatusResponse(BaseModel):
    """对话额度当前状态（免费/VIP 用户重置对话额度前后展示）。"""

    user_id: str
    is_vip: bool
    period_scope: Literal["day", "month"]
    period_key: str
    used: int
    limit: int
    free_remaining: int
    mode: Literal["free", "paid", "blocked"]
    per_msg_cost: float
    spendable_tickets: int


class AdminChatQuotaResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    note: str | None = Field(default=None, max_length=200)
