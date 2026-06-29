from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from app.services.offline.providers._provider_utils import parse_provider_dt
from app.services.offline.providers.gift_types import (
    GiftProviderError,
    GiftTrackingSnapshot,
    GiftTrackingUpdate,
)


class CustomHttpGiftLogisticsProvider:
    """Adapter for a real logistics tracking service."""

    name = "custom_http"

    def __init__(self, *, base_url: str, api_key: str = "", timeout_s: float = 10.0) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._api_key = api_key.strip()
        self._timeout_s = timeout_s
        if not self._base_url:
            raise GiftProviderError("GIFT_LOGISTICS_BASE_URL is required for custom_http")

    async def fetch_tracking(
        self,
        *,
        provider: str,
        provider_order_id: str | None,
        tracking_number: str | None,
    ) -> GiftTrackingSnapshot:
        payload = {
            "provider": provider,
            "provider_order_id": provider_order_id,
            "tracking_number": tracking_number,
        }
        data = await self._post("/tracking", payload)
        items = data.get("events") or data.get("traces") or []
        if not isinstance(items, list):
            raise GiftProviderError("logistics response events must be a list")
        return GiftTrackingSnapshot(
            current_status=_normalize_tracking_status(data.get("current_status") or data.get("status")),
            tracking_number=data.get("tracking_number") or tracking_number,
            events=[event for item in items if isinstance(item, dict) for event in [_event_from_payload(item)] if event],
            raw=data,
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(f"{self._base_url}{path}", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise GiftProviderError(f"logistics provider request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise GiftProviderError("logistics provider response must be a JSON object")
        if data.get("ok") is False:
            raise GiftProviderError(str(data.get("error") or "logistics provider rejected request"))
        return data


def _event_from_payload(data: dict[str, Any]) -> GiftTrackingUpdate | None:
    title = str(data.get("title") or data.get("desc") or data.get("description") or "")
    if not title:
        return None
    return GiftTrackingUpdate(
        status=_normalize_tracking_status(data.get("status")) or "shipping",
        title=title[:120],
        description=data.get("description") or data.get("remark"),
        location=data.get("location") or data.get("city"),
        occurred_at=parse_provider_dt(data.get("occurred_at") or data.get("time")) or datetime.now(UTC),
        raw=data,
    )


def _normalize_tracking_status(value: Any) -> str | None:
    status = str(value or "").strip().lower()
    if status in {"created", "ordered", "paid"}:
        return "ordered"
    if status in {"packed", "accepted", "collected", "pickup"}:
        return "packed"
    if status in {"shipping", "shipped", "in_transit", "transporting", "on_route"}:
        return "shipping"
    if status in {"delivered", "signed", "completed", "received"}:
        return "delivered"
    return status or None


