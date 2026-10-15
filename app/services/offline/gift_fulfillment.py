from __future__ import annotations

from typing import Any

from app.services.offline import gift_repository as gift_repo
from app.services.offline.gift_selection import select_best_candidate
from app.services.offline.gift_status import (
    MANUAL_PROVIDER,
    advances_transit,
    gift_status_for_event,
    normalize_provider_event_status,
)
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
    """从物流 provider 拉轨迹并落库。

    手动礼物（管理员线下寄出）没有可查询的 provider，轨迹全靠后台录入，这里直接跳过——
    否则默认的 mock provider 会往真实寄出的礼物里塞假轨迹。
    """
    if gift.get("provider") == MANUAL_PROVIDER:
        return None
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
    target = provider_gift_status(snapshot.current_status)
    await gift_repo.update_tracking_snapshot(
        gift["id"],
        user_id,
        # 派送中由调用方走 CAS（需要幂等通知用户），这里只推进 ordered → shipping。
        status=target if target != "out_for_delivery" and advances_transit(gift.get("status"), target) else None,
        tracking_number=snapshot.tracking_number,
        logistics_provider=provider.name,
        logistics_payload=snapshot.raw,
    )
    return snapshot


def provider_gift_status(status: str | None) -> str | None:
    """provider 当前状态 → 礼物状态；快递侧已送达只算派送中，签收要用户确认。"""
    return gift_status_for_event(normalize_provider_event_status(status)) if status else None


def _tracking_event_payload(event: GiftTrackingUpdate) -> dict[str, Any]:
    return {
        # 轨迹里的 delivered 只由用户签收写入；快递侧送达记为派送中。
        "status": normalize_provider_event_status(event.status),
        "title": event.title,
        "description": event.description,
        "location": event.location,
        "occurred_at": event.occurred_at,
    }
