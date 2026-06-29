from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.services.offline.providers.gift_types import (
    GiftTrackingSnapshot,
    GiftTrackingUpdate,
)


class MockGiftLogisticsProvider:
    name = "mock"

    async def fetch_tracking(
        self,
        *,
        provider: str,
        provider_order_id: str | None,
        tracking_number: str | None,
    ) -> GiftTrackingSnapshot:
        now = datetime.now(UTC)
        events = [
            GiftTrackingUpdate(
                status="ordered",
                title="订单已创建，等待商家发货",
                description="小惊喜已经进入准备流程。",
                location="上海",
                occurred_at=now,
                raw={"mock": True},
            ),
            GiftTrackingUpdate(
                status="packed",
                title="包裹已揽收，从上海发出",
                description="礼物已经打包好，正在向你飞奔。",
                location="上海",
                occurred_at=now + timedelta(hours=8),
                raw={"mock": True},
            ),
            GiftTrackingUpdate(
                status="shipping",
                title="包裹正在运输中",
                description="下一站会继续更新。",
                location="转运中心",
                occurred_at=now + timedelta(days=1),
                raw={"mock": True},
            ),
        ]
        return GiftTrackingSnapshot(
            current_status="shipping",
            tracking_number=tracking_number,
            events=events,
            raw={"mock": True, "provider_order_id": provider_order_id},
        )
