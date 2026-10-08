"""Paid public facts are shared; caller state, failures and fresh checks aren't."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.services.offline import public_cache as cache


class Cache:
    def __init__(self):
        self.rows = {}
        self.ttls = {}

    async def get(self, key):
        return self.rows.get(key)

    async def set(self, key, value, *, ex, nx=False):
        if nx and key in self.rows:
            return False
        self.rows[key] = value
        self.ttls[key] = ex
        return True

    async def eval(self, script, count, key, token):
        if self.rows.get(key) == token:
            del self.rows[key]
            return 1
        return 0


@pytest.fixture
def shared(monkeypatch):
    redis = Cache()
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=redis))
    return redis


async def reuse(load, **kwargs):
    return await cache.public_cached(
        "test",
        ["public-place"],
        load,
        valid=lambda value: isinstance(value, dict),
        **kwargs,
    )


async def test_concurrent_misses_one_load_and_no_shared_mutable_result(shared):
    started, finish = asyncio.Event(), asyncio.Event()

    async def load():
        started.set()
        await finish.wait()
        return {"places": [{"name": "小岛咖啡"}]}

    model = AsyncMock(side_effect=load)
    first = asyncio.create_task(reuse(model))
    await started.wait()
    rest = [asyncio.create_task(reuse(model)) for _ in range(4)]
    await asyncio.sleep(0)
    finish.set()
    results = await asyncio.gather(first, *rest)
    assert model.await_count == 1
    results[0]["places"][0]["name"] = "用户修改"
    assert results[1]["places"][0]["name"] == "小岛咖啡"
    assert (await reuse(model))["places"][0]["name"] == "小岛咖啡"
    assert model.await_count == 1
    assert not any(key.endswith(":lock") for key in shared.rows)


async def test_cancelled_consumer_does_not_cancel_other_users(shared):
    started, finish = asyncio.Event(), asyncio.Event()

    async def load():
        started.set()
        await finish.wait()
        return {"result": "ok"}

    model = AsyncMock(side_effect=load)
    cancelled = asyncio.create_task(reuse(model))
    await started.wait()
    other = asyncio.create_task(reuse(model))
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    finish.set()
    assert await other == {"result": "ok"}
    model.assert_awaited_once()


async def test_fresh_checks_fetch_independently_and_update_regular_cache(shared):
    load = AsyncMock(side_effect=[{"text": "公告"}, {"text": "取消"}, {"text": "取消"}])
    assert await reuse(load) == {"text": "公告"}
    fresh = await asyncio.gather(reuse(load, fresh=True), reuse(load, fresh=True))
    assert fresh == [{"text": "取消"}, {"text": "取消"}]
    assert await reuse(load) == {"text": "取消"}
    assert load.await_count == 3


async def test_failure_timeout_and_corruption_are_recoverable(shared):
    fail = AsyncMock(side_effect=RuntimeError("provider failed"))
    with pytest.raises(RuntimeError):
        await reuse(fail)
    assert not shared.rows

    async def slow():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await reuse(slow, timeout=0.03)
    assert not shared.rows
    healthy = AsyncMock(return_value={"ok": True})
    await reuse(healthy)
    key = next(iter(shared.rows))
    shared.rows[key] = "broken JSON"
    assert await reuse(healthy) == {"ok": True}
    assert healthy.await_count == 2


async def test_other_worker_lease_is_followed_without_duplicate_fetch(shared):
    load = AsyncMock(return_value={"ok": True})
    await reuse(load)
    key = next(iter(shared.rows))
    shared.rows.clear()
    shared.rows[key + ":lock"] = "another-worker"
    pending = asyncio.create_task(reuse(load, timeout=1))
    await asyncio.sleep(0.02)
    shared.rows[key] = json.dumps({"ok": "other-worker"})
    assert await pending == {"ok": "other-worker"}
    assert shared.rows[key + ":lock"] == "another-worker"  # token-safe cleanup
    load.assert_awaited_once()


async def test_missing_redis_keeps_public_fetch_available(monkeypatch):
    monkeypatch.setattr(cache, "get_redis", AsyncMock(side_effect=OSError()))
    load = AsyncMock(return_value={"ok": True})
    assert await reuse(load) == {"ok": True}
    load.assert_awaited_once()


async def test_optional_cache_serialization_failure_cannot_drop_provider_result(shared):
    from datetime import UTC, datetime

    value = {"provider_timestamp": datetime.now(UTC)}
    load = AsyncMock(return_value=value)
    result = await reuse(load)
    assert result == value and result is not value
    assert not shared.rows  # A non-JSON SDK field is usable without caching it.


async def test_unavailable_lease_cleanup_has_its_own_deadline(shared, monkeypatch):
    async def blocked(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(shared, 'eval', blocked)
    async with asyncio.timeout(2):
        assert await reuse(AsyncMock(return_value={'ok': True})) == {'ok': True}


async def test_identity_isolation_and_short_empty_cache_lifetime(shared):
    load = AsyncMock(return_value={"places": []})
    for region in ["镇江", "南京"]:
        await cache.public_cached(
            "test",
            [region, "咖啡"],
            load,
            valid=lambda v: isinstance(v, dict),
            ttl=lambda v: 30 if not v["places"] else 1800,
        )
    assert load.await_count == 2
    assert list(shared.ttls.values()).count(30) == 2
    assert all("镇江" not in key and "咖啡" not in key for key in shared.rows)


async def test_public_event_extraction_is_reused_but_expiry_is_checked_now(
    shared, monkeypatch
):
    from app.services.offline import cleversee_discovery as discovery
    from tests.test_cleversee_facts import EVENT, TEXT, NOW
    from datetime import timedelta

    model = AsyncMock(return_value={"events": [EVENT]})
    monkeypatch.setattr(discovery, "get_chat_model", lambda: object())
    monkeypatch.setattr(
        discovery,
        "get_prompt_text",
        AsyncMock(return_value="{city} {now} {source_url} {source_text}"),
    )
    monkeypatch.setattr(discovery, "invoke_json", model)
    kwargs = dict(city="镇江", url="https://organizer.example/event")
    started = NOW + timedelta(days=1, hours=7)
    assert len(await discovery.extract_events(TEXT, now=started, **kwargs)) == 1
    assert (
        await discovery.extract_events(TEXT, now=started + timedelta(hours=3), **kwargs)
        == []
    )
    assert model.await_count == 1  # Same-day cache cannot freeze an active event.
    await discovery.extract_events(TEXT + "本场取消", now=started, **kwargs)
    assert model.await_count == 2  # Changed announcement invalidates extraction.
    monkeypatch.setattr(
        discovery,
        "get_prompt_text",
        AsyncMock(return_value="Web新版 {city} {now} {source_url} {source_text}"),
    )
    await discovery.extract_events(TEXT, now=started, **kwargs)
    assert model.await_count == 3  # Live Web publication invalidates extraction.


def test_regional_query_shares_small_gps_changes_and_separates_city_cell_intent():
    from app.services.offline.cleversee_discovery import regional_query
    from app.services.offline.activity_discovery import category_for

    cafe, park = category_for("咖啡馆"), category_for("公园")
    first = regional_query("镇江市", "镇江市润州区", cafe, (32.2111, 119.4311))
    assert first == regional_query("镇江市", "镇江市润州区", cafe, (32.2119, 119.4319))
    assert first != regional_query("镇江市", "镇江市润州区", cafe, (32.2211, 119.4311))
    assert first != regional_query("镇江市", "镇江市润州区", park, (32.2111, 119.4311))
    assert first != regional_query("南京市", "南京市", cafe, (32.2111, 119.4311))
    assert "32.2111" not in first[0] and "16公里" in first[0]
