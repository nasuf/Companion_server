from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.services.offline.providers._provider_utils import parse_provider_dt
from app.services.offline.providers.ali1688_client import Ali1688Client
from app.services.offline.providers.ali1688_token import get_access_token
from app.services.offline.providers.gift_types import (
    GiftProviderError,
    GiftTrackingSnapshot,
    GiftTrackingUpdate,
)

logger = logging.getLogger(__name__)

# ⚠️ 接口名以官方文档为准。
_LOGISTICS_NAMESPACE = "com.alibaba.logistics"
_LOGISTICS_API = "alibaba.trade.getLogisticsTraceInfo"  # 按订单号拉物流轨迹


class Ali1688GiftLogisticsProvider:
    """1688 物流轨迹 provider：按订单号拉取物流跟踪事件。"""

    name = "ali1688"

    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        access_token: str,
        timeout_s: float = 10.0,
    ) -> None:
        self._client = Ali1688Client(
            app_key=app_key,
            app_secret=app_secret,
            access_token=access_token,
            access_token_getter=get_access_token,
            timeout_s=timeout_s,
        )

    async def fetch_tracking(
        self,
        *,
        provider: str,
        provider_order_id: str | None,
        tracking_number: str | None,
    ) -> GiftTrackingSnapshot:
        if not provider_order_id:
            raise GiftProviderError("1688 物流查询需要 provider_order_id（1688 订单号）")
        data = await self._client.call(
            namespace=_LOGISTICS_NAMESPACE,
            api_name=_LOGISTICS_API,
            biz_params={"orderId": provider_order_id, "webSite": "1688"},
        )
        logistics = (
            data.get("result")
            or data.get("logisticsTraceInfo")
            or data.get("logisticsInfo")
            or data
        )
        steps = []
        if isinstance(logistics, dict):
            steps = (
                logistics.get("logisticsSteps")
                or logistics.get("steps")
                or logistics.get("traces")
                or []
            )
        elif isinstance(logistics, list):
            steps = logistics
        if not isinstance(steps, list):
            steps = []

        events = [
            event
            for item in steps
            if isinstance(item, dict)
            for event in [_event_from_step(item)]
            if event
        ]
        mail_no = None
        if isinstance(logistics, dict):
            mail_no = logistics.get("mailNo") or logistics.get("logisticsBillNo")
        return GiftTrackingSnapshot(
            current_status=_normalize_status(events[-1].status if events else None),
            tracking_number=mail_no or tracking_number,
            events=events,
            raw=data,
        )


def _event_from_step(step: dict[str, Any]) -> GiftTrackingUpdate | None:
    title = str(
        step.get("remark") or step.get("acceptStation") or step.get("desc") or step.get("status") or ""
    )
    if not title:
        return None
    return GiftTrackingUpdate(
        status=_normalize_status(step.get("status") or step.get("action")) or "shipping",
        title=title[:120],
        description=step.get("remark") or step.get("desc"),
        location=step.get("location") or step.get("city") or step.get("acceptAddress"),
        occurred_at=parse_provider_dt(step.get("acceptTime") or step.get("time") or step.get("gmtCreate"))
        or datetime.now(UTC),
        raw=step,
    )


def _normalize_status(value: Any) -> str | None:
    status = str(value or "").strip().lower()
    if status in {"created", "ordered", "waitsellersend", "new"}:
        return "ordered"
    if status in {"packed", "accepted", "collected", "pickup", "gotgoods"}:
        return "packed"
    if status in {"shipping", "shipped", "transport", "on_route", "send"}:
        return "shipping"
    if status in {"delivered", "signed", "completed", "received", "sign"}:
        return "delivered"
    return status or None
