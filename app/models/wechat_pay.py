"""服务号 H5 微信支付的请求/应答模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WechatPayProductItem(BaseModel):
    """H5 商品卡。价格由服务端下发，前端只负责展示，不参与定价。"""

    product_id: str
    title: str
    price_fen: int
    list_price_fen: int = 0
    vip_days: int = 0
    ticket_amount: int = 0
    badge: str | None = None
    purchasable: bool = True
    unavailable_reason: str | None = None  # 如体验价已用过


class WechatPayCatalogResponse(BaseModel):
    """H5 商城页所需的全部数据：可售商品 + 当前账号的权益与身份确认信息。"""

    enabled: bool
    openid_bound: bool  # false 时前端需先走一次服务号网页授权
    vip_plans: list[WechatPayProductItem]
    trial: WechatPayProductItem | None = None
    ticket_packs: list[WechatPayProductItem]
    # 付款前让用户确认"买给哪个账号"，避免在错的账号上付钱。
    account_label: str | None = None
    vip_is_active: bool = False
    vip_until: str | None = None
    ticket_balance: int = 0


class WechatPayOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, max_length=64)


class WechatPayParams(BaseModel):
    """`WeixinJSBridge.invoke('getBrandWCPayRequest', …)` 的入参。"""

    appId: str  # noqa: N815 (微信侧字段名，必须保持大小写)
    timeStamp: str  # noqa: N815
    nonceStr: str  # noqa: N815
    package: str
    signType: str  # noqa: N815
    paySign: str  # noqa: N815


class WechatPayOrderResponse(BaseModel):
    out_trade_no: str
    amount_fen: int
    pay_params: WechatPayParams


class WechatPayOrderStatusResponse(BaseModel):
    """H5 支付完成后轮询这个接口——以服务端查单为准，不信前端回调结果。"""

    out_trade_no: str
    status: Literal["created", "paid", "granted", "closed", "refunded", "failed"]
    product_title: str
    amount_fen: int
    vip_is_active: bool = False
    vip_until: str | None = None
    ticket_balance: int = 0


class AdminWechatPayOrderItem(BaseModel):
    out_trade_no: str
    transaction_id: str | None = None
    user_id: str
    username: str | None = None
    nickname: str | None = None
    product_id: str
    product_title: str
    amount_fen: int
    paid_amount_fen: int | None = None
    vip_days: int = 0
    ticket_amount: int = 0
    status: str
    trade_state: str | None = None
    refunded_fen: int = 0
    fail_reason: str | None = None
    # 回调是否到达、是否处理成功——区分"没收到回调"和"收到但处理失败"。
    notification_count: int = 0
    last_notification_at: str | None = None
    last_notification_error: str | None = None
    success_time: str | None = None
    granted_at: str | None = None
    created_at: str


class AdminWechatPayOrderList(BaseModel):
    items: list[AdminWechatPayOrderItem]
    total: int
    granted_count: int
    paid_amount_fen_total: int


class AdminWechatPayRefundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_trade_no: str = Field(min_length=1, max_length=64)
    reason: str | None = Field(default=None, max_length=200)


class AdminWechatPayRefundResponse(BaseModel):
    out_refund_no: str
    status: str
    replay: bool = False


class WechatPayAuthorizeRequest(BaseModel):
    """服务号网页授权（snsapi_base）回跳的 code，用来登记付款 openid。"""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=512)


class WechatPayAuthorizeResponse(BaseModel):
    openid_bound: bool
