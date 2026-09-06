"""Apple IAP 商品 → 权益映射（代码常量，与 Flutter 端 product id 表 lockstep）。

沿用 store_catalog.py 的"商品定义放代码"惯例：product id 集合小、随发版走，
放代码可 code review + 单测 + 无迁移。**改这里必须同步改 App Store Connect 后台
和 Flutter `iap_products.dart`，三处一致。**

权益语义：
- consumable + ticket_amount>0 → 钞票充值包（到账加永久钞票）。
- consumable + vip_days>0     → VIP 时长包 / ¥1 体验（在 max(now, 现到期) 上叠加天数）。
- subscription               → 自动续订 VIP（vip_until 以 Apple expires_date 为准）。
"""

from __future__ import annotations

from dataclasses import dataclass

KIND_SUBSCRIPTION = "subscription"
KIND_CONSUMABLE = "consumable"

# 钞票充值到账 / 订阅激活 写入 wallet_ledger 的 source（配 source_id=transaction_id
# 做幂等回链）。退款反向清算用 SOURCE_REFUND。
SOURCE_APPLE_IAP = "iap_apple"
SOURCE_APPLE_IAP_REFUND = "iap_apple_refund"


@dataclass(frozen=True)
class IapProduct:
    product_id: str
    kind: str  # KIND_SUBSCRIPTION | KIND_CONSUMABLE（= Apple 商品类型）
    ticket_amount: int = 0  # 消耗型钞票包：单份到账钞票数
    vip_days: int = 0  # VIP 时长包/体验的天数；订阅时作为 expires 缺失的兜底

    @property
    def grants_vip(self) -> bool:
        return self.kind == KIND_SUBSCRIPTION or self.vip_days > 0

    @property
    def grants_tickets(self) -> bool:
        return self.ticket_amount > 0


_PREFIX = "com.bansheng"

APPLE_PRODUCTS: dict[str, IapProduct] = {
    # ── VIP 订阅（自动续订）──
    f"{_PREFIX}.vip.monthly.auto": IapProduct(
        f"{_PREFIX}.vip.monthly.auto", KIND_SUBSCRIPTION, vip_days=31
    ),
    # ── VIP 时长包（消耗型，到期不自动扣）──
    f"{_PREFIX}.vip.month": IapProduct(f"{_PREFIX}.vip.month", KIND_CONSUMABLE, vip_days=31),
    f"{_PREFIX}.vip.quarter": IapProduct(f"{_PREFIX}.vip.quarter", KIND_CONSUMABLE, vip_days=93),
    f"{_PREFIX}.vip.year": IapProduct(f"{_PREFIX}.vip.year", KIND_CONSUMABLE, vip_days=372),
    # ── ¥1 体验（消耗型 30 天；账号级"仅一次"由 vip_trial_used 软控，见 grant.py）──
    f"{_PREFIX}.vip.trial": IapProduct(f"{_PREFIX}.vip.trial", KIND_CONSUMABLE, vip_days=30),
    # ── 钞票充值（消耗型）——与 store_data.dart 的 _ticketRechargePacks 档位一致 ──
    f"{_PREFIX}.ticket.10": IapProduct(f"{_PREFIX}.ticket.10", KIND_CONSUMABLE, ticket_amount=10),
    f"{_PREFIX}.ticket.80": IapProduct(f"{_PREFIX}.ticket.80", KIND_CONSUMABLE, ticket_amount=80),
    f"{_PREFIX}.ticket.180": IapProduct(f"{_PREFIX}.ticket.180", KIND_CONSUMABLE, ticket_amount=180),
    f"{_PREFIX}.ticket.300": IapProduct(f"{_PREFIX}.ticket.300", KIND_CONSUMABLE, ticket_amount=300),
    f"{_PREFIX}.ticket.980": IapProduct(f"{_PREFIX}.ticket.980", KIND_CONSUMABLE, ticket_amount=980),
    f"{_PREFIX}.ticket.1980": IapProduct(
        f"{_PREFIX}.ticket.1980", KIND_CONSUMABLE, ticket_amount=1980
    ),
}


def product_for(product_id: str) -> IapProduct | None:
    return APPLE_PRODUCTS.get(product_id)
