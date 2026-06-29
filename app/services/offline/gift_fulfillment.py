from __future__ import annotations

from typing import Any

from app.services.offline import gift_repository as gift_repo
from app.services.offline.gift_selection import select_best_candidate
from app.services.offline.providers.gift_types import (
    GiftOrderResult,
    GiftProductCandidate,
    GiftProviderError,
    GiftTrackingSnapshot,
    GiftTrackingUpdate,
    RecipientAddress,
)
from app.services.offline.providers.registry import get_commerce_provider, get_logistics_provider


def commerce_provider_name() -> str:
    return get_commerce_provider().name


def logistics_provider_name() -> str:
    return get_logistics_provider().name


async def purchase_gift(
    *,
    gift_id: str,
    spec: dict[str, Any],
    address: dict[str, Any],
) -> tuple[GiftProductCandidate, GiftOrderResult]:
    provider = get_commerce_provider()
    amount = int(spec["amount_cents"])
    candidates = await provider.search_products(
        query=spec["gift_name"],
        min_amount_cents=round(amount * 0.8),
        max_amount_cents=round(amount * 1.2),
        limit=5,
    )
    if not candidates:
        raise GiftProviderError(f"no purchasable product found for {spec['gift_name']}")
    candidate = await select_best_candidate(candidates, spec)
    order = await provider.place_order(
        candidate=candidate,
        address=RecipientAddress.from_dict(address),
        idempotency_key=gift_id,
    )
    return candidate, order


def candidate_snapshot(candidate: GiftProductCandidate) -> dict[str, Any]:
    return {
        "external_product_id": candidate.external_product_id,
        "title": candidate.title,
        "price_cents": candidate.price_cents,
        "image_url": candidate.image_url,
        "product_url": candidate.product_url,
        "shop_name": candidate.shop_name,
        "source": candidate.source,
        "raw": candidate.raw,
    }


async def sync_tracking_events(user_id: str, gift: dict[str, Any]) -> GiftTrackingSnapshot | None:
    provider = get_logistics_provider()
    snapshot = await provider.fetch_tracking(
        provider=gift.get("provider") or "mock",
        provider_order_id=gift.get("provider_order_id"),
        tracking_number=gift.get("tracking_number"),
    )
    if snapshot.events:
        await gift_repo.add_tracking_events(
            gift["id"],
            [_tracking_event_payload(event) for event in snapshot.events],
        )
    await gift_repo.update_tracking_snapshot(
        gift["id"],
        user_id,
        status=_gift_status_from_tracking(snapshot.current_status),
        tracking_number=snapshot.tracking_number,
        logistics_provider=provider.name,
        logistics_payload=snapshot.raw,
    )
    return snapshot


def latest_tracking_event(
    events: list[GiftTrackingUpdate],
    status: str,
) -> GiftTrackingUpdate | None:
    matches = [event for event in events if event.status == status]
    return max(matches, key=lambda event: event.occurred_at) if matches else None


def _tracking_event_payload(event: GiftTrackingUpdate) -> dict[str, Any]:
    return {
        "status": event.status,
        "title": event.title,
        "description": event.description,
        "location": event.location,
        "occurred_at": event.occurred_at,
    }


def _gift_status_from_tracking(status: str | None) -> str | None:
    if status == "delivered":
        return "delivered"
    if status in {"packed", "shipping"}:
        return "shipping"
    if status == "ordered":
        return "ordered"
    return None
