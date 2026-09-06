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
