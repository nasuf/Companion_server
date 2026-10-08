"""Public-response caching, native-card boundaries and recoverable failures."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.services.offline.providers import cleversee


@pytest.fixture
async def transport(monkeypatch):
    class Cache:
        def __init__(self):
            self.rows = {}

        async def get(self, key):
            return self.rows.get(key)

        async def set(self, key, value, **kwargs):
            if kwargs.get("nx") and key in self.rows:
                return False
            self.rows[key] = value
            return True

        async def eval(self, *args):
            if len(args) == 4:
                _, _, key, token = args
                if self.rows.get(key) == token:
                    self.rows.pop(key, None)
            return 1

    cache = Cache()
    monkeypatch.setattr(cleversee, "get_redis", AsyncMock(return_value=cache))
    monkeypatch.setattr(cleversee, "_slots", asyncio.Semaphore(2))
    monkeypatch.setattr(settings, "ali_cloud_access_key_id", "fixture-id")
    monkeypatch.setattr(settings, "ali_cloud_access_key_secret", "fixture-secret")
    monkeypatch.setattr(settings, "ali_cleversee_api_key", "fixture-api")
    return cache


async def test_poi_cache_is_query_bound_and_does_not_expose_coordinates_or_keys(
    transport, monkeypatch
):
    sdk = AsyncMock(return_value={"data": [{"id": "P1"}]})
    monkeypatch.setattr(cleversee, "_sdk_query", sdk)
    assert await cleversee.places("镇江 咖啡 坐标32.21,119.43") == [{"id": "P1"}]
    assert await cleversee.places("镇江 咖啡 坐标32.21,119.43") == [{"id": "P1"}]
    assert sdk.await_count == 1
    await cleversee.places("南京 咖啡")
    assert sdk.await_count == 2
    assert all(
        "fixture" not in key and "32.21" not in key and "镇江" not in key
        for key in transport.rows
    )


async def test_qa_uses_native_cards_and_caches_across_random_session_ids(
    transport, monkeypatch
):
    http = AsyncMock(
        return_value={
            "content": "纬度1，经度2，猜测图书馆坐标",
            "cards": [{"cardData": {"places": [{"id": "native"}]}}],
        }
    )
    monkeypatch.setattr(cleversee, "_http", http)
    for _ in range(2):
        assert await cleversee.qa_places("镇江图书馆", lat=32.2, lng=119.4) == [
            {"id": "native"}
        ]
    assert http.await_count == 1
    assert http.call_args.args[1]["locationInfo"]["latitude"] == 32.2


async def test_failure_is_not_cached_and_authentication_is_not_logged(
    transport, monkeypatch, caplog
):
    sdk = AsyncMock(
        side_effect=[
            RuntimeError("credential fixture-secret"),
            {"data": [{"id": "recovered"}]},
        ]
    )
    monkeypatch.setattr(cleversee, "_sdk_query", sdk)
    assert await cleversee.places("镇江咖啡") == []
    assert "fixture-secret" not in caplog.text and "RuntimeError" in caplog.text
    assert await cleversee.places("镇江咖啡") == [{"id": "recovered"}]
    assert sdk.await_count == 2


async def test_fresh_event_read_bypasses_cached_announcement(transport, monkeypatch):
    monkeypatch.setattr(
        "app.services.offline.activity_images._public_url", AsyncMock(return_value=True)
    )
    http = AsyncMock(
        side_effect=[
            {"data": {"statusCode": 200, "text": "原公告"}},
            {"data": {"statusCode": 200, "text": "本场取消公告"}},
        ]
    )
    monkeypatch.setattr(cleversee, "_http", http)
    assert await cleversee.read_page("https://organizer.example/event") == "原公告"
    assert await cleversee.read_page("https://organizer.example/event") == "原公告"
    assert (
        await cleversee.read_page("https://organizer.example/event", fresh=True)
        == "本场取消公告"
    )
    assert http.await_count == 2
