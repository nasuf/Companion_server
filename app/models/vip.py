from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VipStatusResponse(BaseModel):
    """统一 VIP 状态（CLAUDE.md 权益项总览）：供 Store/Profile/Chat/Music 共用一个来源。"""

    is_vip: bool
    vip_until: str | None = None
    vip_trial_available: bool
    gift_ticket_balance: int
    ticket_balance: int
    point_balance: int
    spendable_tickets: int


class ChatQuotaResponse(BaseModel):
    mode: Literal["free", "paid", "blocked"]
    free_remaining: int
    per_msg_cost: float
    spendable_tickets: int


class MusicQuotaReportRequest(BaseModel):
    # 客户端心跳约 15s 上报一次；上限留足重连/前台切回的补报余量，同时防止
    # 恶意或异常客户端一次报出天文数字般的收听时长把每日统计弄脏。
    delta_seconds: int = Field(gt=0, le=300)
    paid_confirmed: bool = False


class MusicQuotaReportResponse(BaseModel):
    action: Literal["none", "confirm_ticket", "buy_coupon", "buy_vip"]
    accepted_seconds: int
    pending_seconds: int
    ticket_cost: int
