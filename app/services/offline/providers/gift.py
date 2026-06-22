from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4


@dataclass(frozen=True)
class MockOrder:
    provider_order_id: str
    tracking_number: str
    product_image_url: str


def product_image_for(gift_name: str) -> str:
    lower = gift_name.lower()
    if "咖啡" in gift_name or "coffee" in lower:
        return "https://images.unsplash.com/photo-1509042239860-f550ce710b93"
    if "书" in gift_name or "绘本" in gift_name:
        return "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f"
    if "围巾" in gift_name or "毛绒" in gift_name:
        return "https://images.unsplash.com/photo-1516762689617-e1cffcef479d"
    return "https://images.unsplash.com/photo-1513201099705-a9746e1e201f"


async def mock_order_gift(gift_name: str) -> MockOrder:
    token = uuid4().hex[:10].upper()
    return MockOrder(
        provider_order_id=f"MOCK-{token}",
        tracking_number=f"RW{token}",
        product_image_url=product_image_for(gift_name),
    )


async def mock_tracking_events() -> list[dict]:
    now = datetime.now(UTC)
    return [
        {
            "status": "ordered",
            "title": "订单已创建，等待商家发货",
            "description": "小惊喜已经进入准备流程。",
            "location": "上海",
            "occurred_at": now,
        },
        {
            "status": "packed",
            "title": "包裹已揽收，从上海发出",
            "description": "礼物已经打包好，正在向你飞奔。",
            "location": "上海",
            "occurred_at": now + timedelta(hours=8),
        },
        {
            "status": "shipping",
            "title": "包裹正在运输中",
            "description": "下一站会继续更新。",
            "location": "转运中心",
            "occurred_at": now + timedelta(days=1),
        },
    ]
