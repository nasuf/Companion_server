from __future__ import annotations

from app.config import settings
from app.services.offline.providers.commerce_base import GiftCommerceProvider
from app.services.offline.providers.custom_commerce import CustomHttpGiftCommerceProvider
from app.services.offline.providers.custom_logistics import CustomHttpGiftLogisticsProvider
from app.services.offline.providers.gift_types import GiftProviderError
from app.services.offline.providers.logistics_base import GiftLogisticsProvider
from app.services.offline.providers.mock_commerce import MockGiftCommerceProvider
from app.services.offline.providers.mock_logistics import MockGiftLogisticsProvider


def get_commerce_provider() -> GiftCommerceProvider:
    provider = (settings.gift_commerce_provider or "mock").strip().lower()
    if provider == "mock":
        return MockGiftCommerceProvider()
    if provider == "custom_http":
        return CustomHttpGiftCommerceProvider(
            base_url=settings.gift_commerce_base_url,
            api_key=settings.gift_commerce_api_key,
            timeout_s=settings.gift_commerce_timeout_s,
        )
    raise GiftProviderError(f"unsupported gift commerce provider: {provider}")


def get_logistics_provider() -> GiftLogisticsProvider:
    provider = (settings.gift_logistics_provider or "mock").strip().lower()
    if provider == "mock":
        return MockGiftLogisticsProvider()
    if provider == "custom_http":
        return CustomHttpGiftLogisticsProvider(
            base_url=settings.gift_logistics_base_url,
            api_key=settings.gift_logistics_api_key,
            timeout_s=settings.gift_logistics_timeout_s,
        )
    raise GiftProviderError(f"unsupported gift logistics provider: {provider}")
