from __future__ import annotations

from typing import Protocol

from app.services.offline.providers.gift_types import GiftTrackingSnapshot


class GiftLogisticsProvider(Protocol):
    name: str

    async def fetch_tracking(
        self,
        *,
        provider: str,
        provider_order_id: str | None,
        tracking_number: str | None,
    ) -> GiftTrackingSnapshot:
        ...
