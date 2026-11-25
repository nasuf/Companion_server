"""服务号 H5 微信支付的商品目录（代码常量，价格以「分」为单位）。

与 `payments/catalog.py`（Apple IAP）**完全独立**：两个渠道各有自己的商品 id、
各自的交易表，互不读写。共享的只有最终权益语义（VIP 天数 / 钞票数），以及
`vip/entitlements.py` 里合并出 `vip_until` 的那一步。

价格与 Flutter `store_subscription_view.dart` / `store_data.dart` 的档位一致
（月卡 ¥39、季卡 ¥89、年卡 ¥249、体验 ¥1、钞票 1/8/18/30/98/198 元）。改价必须
同步改这里和 Flutter，并在 Web H5 上核对——H5 价格由服务端下发，不由前端决定。

内测不开连续包月：微信侧的「支付中签约」审核更严，且自动续费的退订/对账链路
需要单独设计。所有商品都是一次性购买（kind=consumable）。
"""

from __future__ import annotations

from dataclasses import dataclass

PROVIDER_WECHAT = "wechat"

# wallet_ledger 的 source（配 source_id=out_trade_no 做幂等回链）。
SOURCE_WECHAT_PAY = "wechat_pay"
SOURCE_WECHAT_PAY_REFUND = "wechat_pay_refund"

KIND_CONSUMABLE = "consumable"


@dataclass(frozen=True)
class WechatProduct:
    product_id: str
    title: str  # H5 商品卡与微信账单描述共用
    price_fen: int  # 实付金额（分）；服务端唯一价格来源
    list_price_fen: int = 0  # 划线原价（分），0 表示不展示
    vip_days: int = 0
    ticket_amount: int = 0
    badge: str | None = None  # H5 角标，如「最划算」

    @property
    def grants_vip(self) -> bool:
        return self.vip_days > 0

    @property
    def grants_tickets(self) -> bool:
        return self.ticket_amount > 0

    @property
    def kind(self) -> str:
        return KIND_CONSUMABLE


_VIP_PRODUCTS: tuple[WechatProduct, ...] = (
    WechatProduct("wx.vip.month", "月卡", 3900, 6900, vip_days=31),
    WechatProduct("wx.vip.quarter", "季卡", 8900, 16500, vip_days=93),
    WechatProduct("wx.vip.year", "年卡", 24900, 39900, vip_days=372, badge="最划算"),
)

# ¥1 体验：账号级限购一次，由 user_wallets.vip_trial_used 软控（见 grant.py），
# 与 Apple 的 `com.bansheng.vip.trial` 共用同一个标记位——用户在任一渠道体验过
# 就不能在另一渠道再买一次。
_TRIAL_PRODUCT = WechatProduct("wx.vip.trial", "月度 VIP 体验", 100, vip_days=30)

_TICKET_PRODUCTS: tuple[WechatProduct, ...] = (
    WechatProduct("wx.ticket.10", "10 钞票", 100, ticket_amount=10),
    WechatProduct("wx.ticket.80", "80 钞票", 800, ticket_amount=80),
    WechatProduct("wx.ticket.180", "180 钞票", 1800, ticket_amount=180),
    WechatProduct("wx.ticket.300", "300 钞票", 3000, ticket_amount=300),
    WechatProduct("wx.ticket.980", "980 钞票", 9800, ticket_amount=980),
    WechatProduct("wx.ticket.1980", "1980 钞票", 19800, ticket_amount=1980),
)

VIP_PRODUCTS = _VIP_PRODUCTS
TRIAL_PRODUCT = _TRIAL_PRODUCT
TICKET_PRODUCTS = _TICKET_PRODUCTS

WECHAT_PRODUCTS: dict[str, WechatProduct] = {
    product.product_id: product
    for product in (*_VIP_PRODUCTS, _TRIAL_PRODUCT, *_TICKET_PRODUCTS)
}

TRIAL_PRODUCT_ID = _TRIAL_PRODUCT.product_id


def product_for(product_id: str) -> WechatProduct | None:
    return WECHAT_PRODUCTS.get(product_id)
