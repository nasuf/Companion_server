from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.vip import VipStatusResponse
from app.models.wallet import WalletBalanceResponse


class IapVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1)
    # 以下为客户端上报的辅助字段，服务端只信 transaction_id（一切以 Apple 校验为准）。
    product_id: str | None = None
    signed_transaction: str | None = None
    agent_id: str | None = None


class IapVerifyResponse(BaseModel):
    """到账后回给客户端的一致快照：新钱包 + 新 VIP 状态。

    正常到账 status='granted'；对已退款/撤销的交易重复校验会回放其终态
    （refunded/revoked/failed），客户端据此不再当作成功。
    """

    status: Literal["granted", "refunded", "revoked", "failed"]
    kind: Literal["subscription", "consumable"]
    replay: bool = False
    wallet: WalletBalanceResponse
    vip: VipStatusResponse


class IapHistoryItem(BaseModel):
    transaction_id: str
    product_id: str
    product_label: str
    kind: Literal["subscription", "consumable"]
    status: Literal["granted", "refunded", "revoked"]
    purchase_date: str | None = None
    expires_date: str | None = None


class IapSubscriptionStatus(BaseModel):
    product_id: str
    product_label: str
    status: str
    auto_renew_enabled: bool
    auto_renew_product_id: str | None = None
    expires_date: str | None = None
    grace_period_expires_date: str | None = None
    updated_at: str


class IapMembershipResponse(BaseModel):
    vip: VipStatusResponse
    subscription: IapSubscriptionStatus | None = None
    auto_renew_active: bool
    history: list[IapHistoryItem]


class AdminIapTransactionItem(BaseModel):
    id: str
    provider: str
    transaction_id: str
    original_transaction_id: str | None = None
    product_id: str
    kind: str
    environment: str
    user_id: str
    username: str | None = None
    nickname: str | None = None
    quantity: int
    status: str
    purchase_date: str | None = None
    expires_date: str | None = None
    created_at: str


class AdminIapSubscriptionItem(BaseModel):
    original_transaction_id: str
    user_id: str
    product_id: str
    environment: str
    status: str
    auto_renew_status: bool | None = None
    auto_renew_product_id: str | None = None
    expires_date: str | None = None
    grace_period_expires_date: str | None = None
    updated_at: str


class AdminIapNotificationItem(BaseModel):
    id: str
    notification_uuid: str
    notification_type: str
    subtype: str | None = None
    environment: str | None = None
    original_transaction_id: str | None = None
    transaction_id: str | None = None
    processed_at: str | None = None
    process_error: str | None = None
    received_at: str


class AdminVipMemberItem(BaseModel):
    """一个有 VIP 的用户：当前状态 + 套餐 + 过期/续期 + 最近一笔 VIP 交易时间。"""

    user_id: str
    username: str | None = None
    nickname: str | None = None
    is_active: bool
    plan_label: str
    product_id: str | None = None
    is_auto_renew: bool
    subscription_status: str | None = None
    vip_until: str | None = None
    next_renewal_date: str | None = None
    grace_period_expires_date: str | None = None
    vip_trial_used: bool = False
    environment: str | None = None
    started_at: str | None = None  # 发起时间（最近一笔 VIP 交易的 purchase_date）
    credited_at: str | None = None  # 到账时间（最近一笔 VIP 交易的 created_at）
    last_transaction_id: str | None = None


class AdminVipMemberList(BaseModel):
    items: list[AdminVipMemberItem]
    total: int
    active_count: int


class AdminRechargeItem(BaseModel):
    """一笔钞票充值（消耗型内购）：到账钞票 + 实付金额 + 发起/到账时间 + 状态。"""

    transaction_id: str
    user_id: str
    username: str | None = None
    nickname: str | None = None
    product_id: str
    product_label: str
    tickets: int
    quantity: int
    amount: float | None = None  # 实付金额（货币主单位，如 29.00）；旧交易无
    currency: str | None = None  # ISO4217，如 CNY/USD
    storefront: str | None = None  # ISO 国家码，如 CHN/USA
    environment: str
    status: str
    initiated_at: str | None = None  # 发起时间（Apple purchase_date）
    credited_at: str | None = None  # 到账时间（我方入账 created_at）
    updated_at: str | None = None


class AdminRechargeList(BaseModel):
    items: list[AdminRechargeItem]
    total: int
    total_tickets: int
    distinct_users: int
