from __future__ import annotations

from typing import Protocol

from app.services.offline.providers.gift_types import (
    GiftOrderResult,
    GiftProductCandidate,
    RecipientAddress,
)


class GiftCommerceProvider(Protocol):
    name: str

    async def search_products(
        self,
        *,
        query: str,
        min_amount_cents: int,
        max_amount_cents: int,
        limit: int = 5,
    ) -> list[GiftProductCandidate]:
        ...

    async def place_order(
        self,
        *,
        candidate: GiftProductCandidate,
        address: RecipientAddress,
        idempotency_key: str,
    ) -> GiftOrderResult:
        ...
