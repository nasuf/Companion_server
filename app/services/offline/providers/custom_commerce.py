from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from app.services.offline.providers.gift_types import (
    GiftOrderResult,
    GiftProductCandidate,
    GiftProviderError,
    RecipientAddress,
)


class CustomHttpGiftCommerceProvider:
    """Adapter for a real buyer-side purchasing service.

    The external service owns marketplace-specific details: Taobao auth,
    search ranking, buyer checkout, payment, and order idempotency. The app
    receives only normalized product/order data.
    """

    name = "custom_http"

    def __init__(self, *, base_url: str, api_key: str = "", timeout_s: float = 12.0) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._api_key = api_key.strip()
        self._timeout_s = timeout_s
        if not self._base_url:
            raise GiftProviderError("GIFT_COMMERCE_BASE_URL is required for custom_http")

    async def search_products(
        self,
        *,
        query: str,
        min_amount_cents: int,
        max_amount_cents: int,
        limit: int = 5,
    ) -> list[GiftProductCandidate]:
        payload = {
            "query": query,
            "min_amount_cents": min_amount_cents,
            "max_amount_cents": max_amount_cents,
            "currency": "CNY",
            "limit": limit,
            "constraints": {
                "stock_required": True,
                "exclude_food_medicine": True,
                "exclude_size_dependent_clothing": True,
            },
        }
        data = await self._post("/search", payload)
        items = data.get("items") or data.get("results") or []
        if not isinstance(items, list):
            raise GiftProviderError("commerce search response items must be a list")
        candidates = [_candidate_from_payload(item) for item in items if isinstance(item, dict)]
        return [item for item in candidates if item is not None][:limit]

    async def place_order(
        self,
        *,
        candidate: GiftProductCandidate,
        address: RecipientAddress,
        idempotency_key: str,
    ) -> GiftOrderResult:
        payload = {
            "idempotency_key": idempotency_key,
            "product": {
                "external_product_id": candidate.external_product_id,
                "title": candidate.title,
                "price_cents": candidate.price_cents,
                "product_url": candidate.product_url,
                "image_url": candidate.image_url,
                "shop_name": candidate.shop_name,
                "raw": candidate.raw,
            },
            "address": address.as_payload(),
            "payment": {"mode": "provider_account"},
        }
        data = await self._post("/orders", payload)
        provider_order_id = str(data.get("provider_order_id") or data.get("order_id") or "")
        if not provider_order_id:
            raise GiftProviderError("commerce order response missing provider_order_id")
        return GiftOrderResult(
            provider=str(data.get("provider") or self.name),
            provider_order_id=provider_order_id,
            status=_normalize_order_status(str(data.get("status") or "ordered")),
            paid_amount_cents=_int_cents(data.get("paid_amount_cents"), candidate.price_cents),
            product_image_url=data.get("product_image_url") or candidate.image_url,
            tracking_number=data.get("tracking_number"),
            shipped_at=_parse_dt(data.get("shipped_at")),
            delivered_at=_parse_dt(data.get("delivered_at")),
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
            raise GiftProviderError(f"commerce provider request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise GiftProviderError("commerce provider response must be a JSON object")
        if data.get("ok") is False:
            raise GiftProviderError(str(data.get("error") or "commerce provider rejected request"))
        return data


def _candidate_from_payload(data: dict[str, Any]) -> GiftProductCandidate | None:
    product_id = str(data.get("external_product_id") or data.get("item_id") or data.get("id") or "")
    title = str(data.get("title") or data.get("name") or "")
    price_cents = _int_cents(data.get("price_cents") or data.get("amount_cents"), 0)
    if not product_id or not title or price_cents <= 0:
        return None
    return GiftProductCandidate(
        external_product_id=product_id,
        title=title[:120],
        price_cents=price_cents,
        image_url=data.get("image_url") or data.get("pic_url"),
        product_url=data.get("product_url") or data.get("url"),
        shop_name=data.get("shop_name") or data.get("seller_name"),
        source=str(data.get("source") or "custom_http"),
        raw=data,
    )


def _normalize_order_status(status: str) -> str:
    value = status.strip().lower()
    if value in {"paid", "ordered", "created"}:
        return "ordered"
    if value in {"shipped", "shipping", "in_transit", "packed"}:
        return "shipping"
    if value in {"delivered", "signed", "completed"}:
        return "delivered"
    return "ordered"


def _int_cents(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except Exception:
            return None
    return None
