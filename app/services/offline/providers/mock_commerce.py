from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.services.offline.providers.gift_types import (
    GiftOrderResult,
    GiftProductCandidate,
    RecipientAddress,
)


class MockGiftCommerceProvider:
    name = "mock"

    async def search_products(
        self,
        *,
        query: str,
        min_amount_cents: int,
        max_amount_cents: int,
        limit: int = 5,
    ) -> list[GiftProductCandidate]:
        amount = max(min_amount_cents, min(max_amount_cents, (min_amount_cents + max_amount_cents) // 2))
        return [
            GiftProductCandidate(
                external_product_id=f"MOCK-PRODUCT-{uuid4().hex[:8].upper()}",
                title=query[:60] or "小礼物",
                price_cents=amount,
                image_url=product_image_for(query),
                product_url=None,
                shop_name="Mock Gift Store",
                source=self.name,
                raw={"mock": True},
            )
        ][:limit]

    async def place_order(
        self,
        *,
        candidate: GiftProductCandidate,
        address: RecipientAddress,
        idempotency_key: str,
    ) -> GiftOrderResult:
        token = uuid4().hex[:10].upper()
        now = datetime.now(UTC)
        return GiftOrderResult(
            provider=self.name,
            provider_order_id=f"MOCK-{token}",
            status="shipping",
            paid_amount_cents=candidate.price_cents,
            product_image_url=candidate.image_url,
            tracking_number=f"RW{token}",
            shipped_at=now + timedelta(hours=8),
            delivered_at=now + timedelta(days=3),
            raw={"mock": True, "idempotency_key": idempotency_key},
        )


def product_image_for(gift_name: str) -> str:
    lower = gift_name.lower()
    if "咖啡" in gift_name or "coffee" in lower:
        return "https://images.unsplash.com/photo-1509042239860-f550ce710b93"
    if "书" in gift_name or "绘本" in gift_name:
        return "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f"
    if "围巾" in gift_name or "毛绒" in gift_name:
        return "https://images.unsplash.com/photo-1516762689617-e1cffcef479d"
    return "https://images.unsplash.com/photo-1513201099705-a9746e1e201f"
