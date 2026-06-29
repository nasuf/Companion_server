from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class RecipientAddress:
    recipient_name: str
    phone: str
    province: str = ""
    city: str = ""
    district: str = ""
    detail: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipientAddress":
        return cls(
            recipient_name=str(data.get("recipient_name") or data.get("recipientName") or ""),
            phone=str(data.get("phone") or ""),
            province=str(data.get("province") or ""),
            city=str(data.get("city") or ""),
            district=str(data.get("district") or ""),
            detail=str(data.get("detail") or ""),
        )

    def as_payload(self) -> dict[str, str]:
        return {
            "recipient_name": self.recipient_name,
            "phone": self.phone,
            "province": self.province,
            "city": self.city,
            "district": self.district,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GiftProductCandidate:
    external_product_id: str
    title: str
    price_cents: int
    image_url: str | None = None
    product_url: str | None = None
    shop_name: str | None = None
    source: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GiftOrderResult:
    provider: str
    provider_order_id: str
    status: str
    paid_amount_cents: int
    product_image_url: str | None = None
    tracking_number: str | None = None
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GiftTrackingUpdate:
    status: str
    title: str
    description: str | None = None
    location: str | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GiftTrackingSnapshot:
    current_status: str | None = None
    tracking_number: str | None = None
    events: list[GiftTrackingUpdate] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class GiftProviderError(RuntimeError):
    """Raised when an external gift provider cannot complete its part."""
