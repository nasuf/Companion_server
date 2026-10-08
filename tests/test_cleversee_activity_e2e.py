"""Web version → provider wire fixtures → real HTTP/PG/Redis/media → journey.

SDK and external HTTP/model responses are controlled. Identity/radius/event
validation, image decoding/deduplication, storage, API and lifecycle are real.
"""

import asyncio
import io
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image, ImageDraw

from app.config import settings
from app.api.public import offline
from app.services.offline import (
    activity_generation as generation,
    activity_images as images,
    activity_service as service,
)
from app.services.offline import (
    repository as repo,
    cleversee_discovery as discovery,
    image_quality,
)
from app.services.offline.discovery_facts import (
    native_place,
    session_key,
    validate_event,
)
from app.services.offline.providers import cleversee
from app.services.offline.providers.search import SearchResult
from app.services.offline.chat_emit import build_activity_component_card
from app.services.offline.geocode import wgs84_to_gcj02
from app.services.prompting.registry import PROMPT_DEFINITION_MAP
from app.services.prompting import store
from tests.test_offline_quality_e2e import journey, post  # noqa: F401
from tests.test_proactive_activity_e2e import flow  # noqa: F401

RELEASE = json.loads(
    (
        Path(__file__).parents[1]
        / "scripts/prompt_releases/20261008_cleversee_activity.json"
    ).read_text()
)
DETAIL_RELEASE = json.loads(
    (
        Path(__file__).parents[1]
        / "scripts/prompt_releases/20261008_activity_detail_quality.json"
    ).read_text()
)
RELEASE["prompts"] = list(
    {
        entry["key"]: entry for entry in RELEASE["prompts"] + DETAIL_RELEASE["prompts"]
    }.values()
)


@pytest.fixture
async def native(journey, monkeypatch, tmp_path):  # noqa: F811 — imported pytest fixture
    j = journey
    for entry in RELEASE["prompts"]:
        key = entry["key"]
        definition = PROMPT_DEFINITION_MAP[key]
        await j.db.prompttemplate.upsert(
            where={"key": key},
            data={
                "create": dict(
                    key=key,
                    stage=definition.stage,
                    category=definition.category,
                    title=definition.title,
                    description=definition.description,
                    content=definition.default_text,
                    defaultContent=definition.default_text,
                ),
                "update": dict(content=definition.default_text, isEnabled=True),
            },
        )
        before = await j.db.prompttemplate.find_unique(where={"key": key})
        history = await j.db.prompttemplateversion.find_many(where={"promptKey": key})
        r = await j.client.put(
            "/admin-api/prompts/" + key,
            json={
                "content": entry["content"],
                "expected_updated_at": before.updatedAt.isoformat(),
            },
        )
        assert r.status_code == 200, r.text
        after = await j.db.prompttemplate.find_unique(where={"key": key})
        versions = await j.db.prompttemplateversion.find_many(where={"promptKey": key})
        added = [v for v in versions if v.id not in {old.id for old in history}]
        assert len(added) == 1 and added[0].changeType == "manual_save"
        assert (
            before.defaultContent == after.defaultContent
            and before.isEnabled == after.isEnabled
        )
        assert (
            after.content
            == await j.redis.get(store._redis_key(key))
            == await store.get_prompt_text(key)
        )
        stale = await j.client.put(
            "/admin-api/prompts/" + key,
            json={
                "content": "stale",
                "expected_updated_at": before.updatedAt.isoformat(),
            },
        )
        assert stale.status_code == 409
    monkeypatch.setattr(offline, "is_activity_enabled", AsyncMock(return_value=True))
    for key, value in [
        ("offline_search_provider", "cleversee"),
        ("offline_tavily_fallback", False),
        ("ali_cloud_access_key_id", "fixture-ak"),
        ("ali_cloud_access_key_secret", "fixture-sk"),
        ("ali_cleversee_api_key", "fixture-api"),
    ]:
        monkeypatch.setattr(settings, key, value)

    class PublicCache:
        async def get(self, key):
            return await j.redis.get("native:" + j.user + ":" + key)

        async def set(self, key, value, **kwargs):
            return await j.redis.set("native:" + j.user + ":" + key, value, **kwargs)

        async def eval(self, script, count, key, *args):
            return await j.redis.eval(
                script, count, "native:" + j.user + ":" + key, *args
            )

    cache = PublicCache()
    monkeypatch.setattr(cleversee, "get_redis", AsyncMock(return_value=cache))
    monkeypatch.setattr(
        "app.services.offline.public_cache.get_redis", AsyncMock(return_value=cache)
    )
    monkeypatch.setattr(cleversee, "_slots", asyncio.Semaphore(2))
    monkeypatch.setattr(images.storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(images, "_public_url", AsyncMock(return_value=True))
    monkeypatch.setattr(image_quality, "get_redis", AsyncMock(return_value=cache))
    monkeypatch.setattr(
        image_quality, "_call_doubao_vision", AsyncMock(return_value='{"kind":"photo"}')
    )
    monkeypatch.setattr(
        generation.repo, "list_user_tags", AsyncMock(return_value=["咖啡"])
    )
    monkeypatch.setattr(
        generation.repo, "memory_brief", AsyncMock(return_value="用户喜欢看展")
    )
    monkeypatch.setattr(
        service,
        "generate_activity_invite_message",
        AsyncMock(return_value="这地方可以看看，有空再去"),
    )
    geocode = AsyncMock(
        side_effect=AssertionError("Native coordinates must never be overwritten")
    )
    monkeypatch.setattr(service, "geocode_address", geocode)
    lat, lng = wgs84_to_gcj02(32.21, 119.43)
    await j.db.execute_raw(
        "UPDATE users SET location_city='镇江市',location_region='润州区',location_latitude=32.21,location_longitude=119.43 WHERE id=$1",
        j.user,
    )
    j.raw = dict(
        id="B_FIXTURE_" + j.user,
        name="小岛咖啡(伯先路店)",
        cityName="镇江市",
        address="伯先路12号",
        types="餐饮服务|咖啡厅",
        latitude=str(lat),
        longitude=str(lng),
        images=[{"url": f"https://photos.fixture.test/{i}.jpg"} for i in range(4)],
    )
    blobs = []
    for index in range(3):
        image = Image.new("RGB", (360, 270), "white")
        draw = ImageDraw.Draw(image)
        for x in range(8):
            for y in range(8):
                if (x * 11 + y * 7 + index * 17) % (3 + index) == 0:
                    draw.rectangle(
                        (x * 45, y * 33, x * 45 + 38, y * 33 + 27), fill="black"
                    )
        buf = io.BytesIO()
        image.save(buf, "JPEG")
        blobs.append(buf.getvalue())
    # Duplicate URL content is intentionally included; gallery must refill.
    blobs.insert(1, blobs[0])
    j.calls = []
    j.event_rows = []
    j.source_text = ""
    j.pages = []
    j.selection = None
    j.sdk_calls = []

    async def transport(request):
        j.calls.append((request.method, request.url.path))
        if request.url.host == "photos.fixture.test":
            return httpx.Response(
                200,
                content=blobs[int(request.url.path[1:-4])],
                headers={"content-type": "image/jpeg"},
            )
        if request.url.path == "/search/unified":
            return httpx.Response(200, json={"pageItems": j.pages})
        if request.url.path == "/search/multimodal":
            return httpx.Response(200, json={"imageItems": []})
        if request.url.path == "/readpage/basic":
            return httpx.Response(
                200, json={"data": {"text": j.source_text, "statusCode": 200}}
            )
        if request.url.path == "/qa/chat":
            return httpx.Response(
                200,
                json={
                    "content": "Hallucinated coordinates: 1,2",
                    "cards": [{"cardData": {"places": []}}],
                },
            )
        raise AssertionError(str(request.url))

    original = httpx.AsyncClient

    def client(**kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(transport))
        return original(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)

    async def sdk(body):
        j.sdk_calls.append(body)
        assert body["searchModel"] == "normal" and 1 <= body["limit"] <= 25
        return {
            "data": [
                j.raw,
                {**j.raw, "id": "other", "cityName": "南京市"},
                {**j.raw, "id": "list", "name": "镇江这些咖啡店"},
            ]
        }

    monkeypatch.setattr(cleversee, "_sdk_query", sdk)

    async def model(_model, prompt):
        if "候选原始ID" in prompt:
            return json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": j.selection or "poi:" + j.raw["id"],
                            "title": "虚构别处",
                            "summary": "免费全天营业",
                            "place_lat": 1,
                            "starts_at": "1999-01-01",
                            "image_urls": ["https://wrong.example/1.jpg"],
                        }
                    ]
                }
            )
        return "可以去这家坐一会儿，按自己的节奏来。"

    monkeypatch.setattr(generation, "invoke_text", model)
    monkeypatch.setattr(generation, "get_chat_model", lambda: object())
    monkeypatch.setattr(generation, "get_utility_model", lambda: object())
    monkeypatch.setattr(
        generation, "invoke_json", AsyncMock(return_value={"supported": True})
    )
    monkeypatch.setattr(discovery, "get_chat_model", lambda: object())
    monkeypatch.setattr(
        discovery,
        "invoke_json",
        AsyncMock(side_effect=lambda *a, **k: {"events": j.event_rows}),
    )
    monkeypatch.setattr(
        generation,
        "tavily_search",
        AsyncMock(side_effect=AssertionError("Tavily must be off")),
    )
    monkeypatch.setattr(
        images,
        "tavily_place_images",
        AsyncMock(side_effect=AssertionError("Tavily must be off")),
    )
    return j


async def test_same_region_reuses_public_candidates_and_gallery_without_user_history_leak(
    native, monkeypatch
):
    j = native
    calls = AsyncMock(wraps=generation.invoke_text)
    monkeypatch.setattr(generation, "invoke_text", calls)
    from app.services.offline.activity_discovery import ACTIVITY_PLACE_CATEGORIES

    cafe = next(c for c in ACTIVITY_PLACE_CATEGORIES if c.name == "咖啡与茶饮")
    monkeypatch.setattr(discovery, "ordered_categories", lambda *args: [cafe])
    monkeypatch.setattr(discovery, "_events", AsyncMock(return_value=[]))
    before = await discovery.discover(
        city="镇江市",
        search_anchor="镇江市润州区",
        tags=["咖啡"],
        recent=[],
        center=(32.2111, 119.4311),
    )
    assert len(before) == 1 and len(j.sdk_calls) == 1
    # Small GPS movement in the same regional cell reuses raw facts, yet this
    # user's recent place is removed after retrieving those shared facts.
    repeat = await discovery.discover(
        city="镇江市",
        search_anchor="镇江市润州区",
        tags=["咖啡"],
        recent=[{"location_name": j.raw["name"]}],
        center=(32.2119, 119.4319),
    )
    assert repeat == [] and len(j.sdk_calls) == 1
    other = await discovery.discover(
        city="镇江市",
        search_anchor="镇江市润州区",
        tags=["咖啡"],
        recent=[],
        center=(32.2119, 119.4319),
    )
    assert len(other) == 1 and len(j.sdk_calls) == 1
    first = await generation._native_card(
        before,
        user_id=j.user,
        city="镇江市",
        search_anchor="镇江市润州区",
        tags=["咖啡"],
        memory="用户甲喜欢安静",
        recent=[],
    )
    assert len(first["image_urls"]) == 3
    assert "用户甲喜欢安静" in calls.call_args_list[0].args[1]
    calls.reset_mock()
    paid_calls = len(j.calls)
    second = await generation._native_card(
        other,
        user_id="another-user",
        city="镇江市",
        search_anchor="镇江市润州区",
        tags=["咖啡"],
        memory="用户乙喜欢阅读",
        recent=[],
    )
    assert first["image_urls"] == second["image_urls"]
    assert "用户乙喜欢阅读" in calls.call_args_list[0].args[1]
    assert "用户甲喜欢安静" not in calls.call_args_list[0].args[1]
    assert len(j.calls) == paid_calls and len(j.sdk_calls) == 1
    # Cache and selector mutation cannot contaminate the public POI payload.
    before[0]["title"] = "用户甲临时修改"
    assert other[0]["title"] == j.raw["name"]


@pytest.mark.parametrize("photo_count", [0, 1, 3])
async def test_native_album_cooldown_avoids_repeat_enrichment_and_repairs_missing_files(
    native, monkeypatch, photo_count
):
    j = native
    raw = {**j.raw, "images": j.raw["images"][-photo_count:] if photo_count else []}
    card = native_place(raw, "镇江市", center=(32.21, 119.43))
    # An eligible page is supplied even when the native gallery is sufficient.
    source = SearchResult(
        title=j.raw["name"],
        url="https://source.fixture.test/cafe",
        content="镇江市伯先路12号 " + j.raw["name"],
        images=[
            {**i, "poi_id": card["native_poi_id"], "evidence": "native_poi"}
            for i in card["native_images"]
        ],
    )
    supplement = SearchResult(
        title=j.raw["name"],
        url="https://source.fixture.test/inside",
        content="镇江市伯先路12号 " + j.raw["name"],
    )
    page = AsyncMock(return_value={})
    monkeypatch.setattr(images, "page_image_evidence", page)
    first = await images.persist_activity_images(
        user_id=j.user, card=card, city="镇江市", search_results=[source, supplement]
    )
    assert len(first) == photo_count
    if photo_count == 3:
        page.assert_not_awaited()
        assert not any(path == "/search/multimodal" for _, path in j.calls)
    calls = len(j.calls)
    second_card = native_place(raw, "镇江市", center=(32.21, 119.43))
    second = await images.persist_activity_images(
        user_id="other-user",
        card=second_card,
        city="镇江市",
        search_results=[source, supplement],
    )
    assert second == first and len(j.calls) == calls
    if first:
        images.storage.storage_path(card["image_provenance"][0]["storage_key"]).unlink()
        repaired = await images.persist_activity_images(
            user_id="third-user",
            card=second_card,
            city="镇江市",
            search_results=[source, supplement],
        )
        assert len(repaired) == photo_count and len(j.calls) > calls
        assert all(
            images.storage.storage_path(i["storage_key"]).is_file()
            for i in second_card["image_provenance"]
        )


async def test_generate_native_gallery_arrive_archive_reuses_all_facts(native):
    j = native
    r = await j.client.post(
        "/offline/activities/recommend", params={"workspace_id": j.workspace}
    )
    assert r.status_code == 200, r.text
    # API wrapper contains the same item that subsequent detail reads return.
    payload = r.json()
    a = payload.get("activity", payload)
    assert (
        a["title"] == j.raw["name"]
        and len(a["image_urls"]) == 3
        and "全天" not in a["summary"]
    )
    stored = await repo.get_activity(a["id"], j.user)
    assert (
        stored["place_lat"] == float(j.raw["latitude"]) and stored["starts_at"] is None
    )
    card = build_activity_component_card(stored, status_label="待确定")
    assert (
        card["title"] == a["title"] and card["payload"]["image_urls"] == a["image_urls"]
    )
    detail = await j.client.get("/offline/activities/" + a["id"])
    assert detail.json()["image_urls"] == a["image_urls"]
    assert (await post(j, a, "accept")).status_code == 200
    assert (
        await post(j, a, "arrive", {"lat": 31.2, "lng": 119.43, "accuracy_m": 10})
    ).status_code == 422
    old = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    stale = await post(
        j,
        a,
        "arrive",
        {"lat": 32.21, "lng": 119.43, "accuracy_m": 10, "observed_at": old},
    )
    assert (
        stale.status_code == 422
        and stale.json()["detail"]["reason"] == "stale_location"
    )
    assert (
        await post(
            j,
            a,
            "arrive",
            {
                "lat": 32.21,
                "lng": 119.43,
                "accuracy_m": 10,
                "observed_at": datetime.now(UTC).isoformat(),
            },
        )
    ).status_code == 200
    assert (await post(j, a, "archive")).status_code == 200
    review = await j.client.get("/offline/activities/" + a["id"] + "/review")
    assert review.status_code == 200 and review.json()["image_urls"] == a["image_urls"]
    assert review.json()["fragments"] == [] and "推荐介绍" not in review.json()["story"]
    assert not any("tavily" in path for _, path in j.calls)
    generation.tavily_search.assert_not_awaited()
    images.tavily_place_images.assert_not_awaited()


async def test_event_sessions_cancellation_and_atomic_acceptance(native, monkeypatch):
    j = native
    day = (
        (datetime.now(UTC) + timedelta(days=1))
        .astimezone(timezone(timedelta(hours=8)))
        .date()
    )
    quote = f"{day:%Y年%m月%d日}19:00—21:00"
    identity = f"{day.year}年秋日音乐会在小岛咖啡(伯先路店)举办"
    event = dict(
        title="秋日音乐会",
        venue_name=j.raw["name"],
        city="镇江市",
        year=day.year,
        starts_at=f"{day}T19:00:00+08:00",
        ends_at=f"{day}T21:00:00+08:00",
        time_precision="datetime",
        daily_hours=[],
        status="scheduled",
        identity_evidence=identity,
        year_evidence=identity,
        schedule_evidence=quote,
    )
    j.source_text = identity + "。时间：" + quote
    j.event_rows = [event]
    facts = validate_event(
        event,
        j.source_text,
        city="镇江市",
        source_url="https://source.fixture.test/event",
    )
    place = native_place(j.raw, "镇江市")

    async def create(e):
        return await repo.create_activity(
            {
                **place,
                "user_id": j.user,
                "agent_id": j.agent,
                "workspace_id": j.workspace,
                "conversation_id": j.conversation,
                "status": "pending",
                "title": e["title"],
                "starts_at": e["starts_at"],
                "ends_at": e["ends_at"],
                "place_key": session_key(j.raw["id"], e),
                "discovery_metadata": {
                    **place["discovery_metadata"],
                    "kind": "event",
                    "time_precision": "datetime",
                    "event": e,
                },
            }
        )

    a = await create(facts)
    b = await create(facts)
    refreshed = await discovery.refresh_event(a)
    assert refreshed, (
        j.calls,
        a["starts_at"],
        facts,
        await discovery.extract_events(
            j.source_text, city="镇江市", url=facts["official_url"], allow_inactive=True
        ),
    )
    results = await asyncio.gather(post(j, a, "accept"), post(j, b, "accept"))
    assert sorted(r.status_code for r in results) == [200, 409]
    accepted = a if results[0].status_code == 200 else b
    assert (
        await post(
            j, accepted, "arrive", {"lat": 32.21, "lng": 119.43, "accuracy_m": 10}
        )
    ).status_code == 409
    # The same venue can host another session without conflicting with the first.
    tomorrow = day + timedelta(days=1)
    quote2 = f"{tomorrow:%Y年%m月%d日}19:00—21:00"
    event2 = {
        **event,
        "starts_at": f"{tomorrow}T19:00:00+08:00",
        "ends_at": f"{tomorrow}T21:00:00+08:00",
        "schedule_evidence": quote2,
    }
    j.source_text += "。第二场时间：" + quote2
    j.event_rows = [event, event2]
    facts2 = validate_event(
        event2,
        j.source_text,
        city="镇江市",
        source_url="https://source.fixture.test/event",
    )
    c = await create(facts2)
    assert (await post(j, c, "accept")).status_code == 200
    original_source = j.source_text
    j.source_text = ""
    assert (await post(j, c, "accept")).status_code == 503
    assert (await repo.get_activity(c["id"], j.user))["status"] == "accepted"
    j.source_text = original_source
    proof = f"{day.year}年秋日音乐会取消"
    j.source_text += "。" + proof
    j.event_rows = [{**event, "status": "cancelled", "status_evidence": proof}, event2]
    cancelled = await post(j, accepted, "accept")
    assert cancelled.status_code == 409
    current = await repo.get_activity(accepted["id"], j.user)
    assert current["discovery_metadata"]["event"]["status"] == "cancelled"
    stale = await repo.save_discovery_metadata(
        accepted["id"], j.user, {**current["discovery_metadata"], "event": facts}
    )
    assert stale["event"]["status"] == "cancelled"
    assert (
        await repo.mark_arrived(
            accepted["id"], j.user, lat=32.21, lng=119.43, verified=True
        )
        is None
    )
    # Deleting a planned outing is not a memory of attending it.
    await j.db.execute_raw(
        "INSERT INTO memories_user (id,user_id,workspace_id,content,importance,level,main_category,sub_category,updated_at) VALUES ($1,$2,$3,'去年常去小岛咖啡馆',0.7,2,'生活','其他',CURRENT_TIMESTAMP)",
        "historical-" + j.user,
        j.user,
        j.workspace,
    )
    removed = await j.client.delete("/offline/activities/" + c["id"])
    assert removed.status_code == 200
    assert (await repo.get_activity(c["id"], j.user))["status"] == "cancelled"
    assert await j.db.query_raw(
        "SELECT id FROM memories_user WHERE id=$1", "historical-" + j.user
    )


async def test_current_event_is_discovered_and_bound_to_venue(native):
    j = native
    day = (
        (datetime.now(UTC) + timedelta(days=1))
        .astimezone(timezone(timedelta(hours=8)))
        .date()
    )
    identity = f"{day.year}年秋日市集在{j.raw['name']}举办"
    quote = f"{day:%Y年%m月%d日}10:00—18:00"
    event = dict(
        title="秋日市集",
        venue_name=j.raw["name"],
        city="镇江市",
        year=day.year,
        starts_at=f"{day}T10:00:00+08:00",
        ends_at=f"{day}T18:00:00+08:00",
        time_precision="datetime",
        daily_hours=[],
        status="scheduled",
        identity_evidence=identity,
        year_evidence=identity,
        schedule_evidence=quote,
    )
    j.source_text = identity + "。活动时间：" + quote
    j.event_rows = [event]
    j.selection = session_key(j.raw["id"], event)
    j.pages = [
        dict(
            title=identity,
            link="https://source.fixture.test/market",
            mainText=j.source_text,
        )
    ]
    r = await j.client.post(
        "/offline/activities/recommend", params={"workspace_id": j.workspace}
    )
    assert r.status_code == 200, r.text
    a = r.json()
    assert (
        a["kind"] == "event"
        and a["title"] == "秋日市集"
        and a["location_name"] == j.raw["name"]
    )
    assert (
        a["time_precision"] == "datetime"
        and "10:00—18:00" in a["schedule_label"]
        and len(a["image_urls"]) == 3
    )
    assert (await post(j, a, "accept")).status_code == 200


async def test_empty_wrong_city_and_qa_prose_never_create_fictional_place(
    native, monkeypatch
):
    j = native
    monkeypatch.setattr(
        cleversee,
        "_sdk_query",
        AsyncMock(return_value={"data": [{**j.raw, "cityName": "南京市"}]}),
    )
    r = await j.client.post(
        "/offline/activities/recommend", params={"workspace_id": j.workspace}
    )
    assert (
        r.status_code == 503 and r.json()["detail"]["reason"] == "no_suitable_activity"
    )
    assert await repo.list_activities(j.user, j.workspace) == []
    generation.tavily_search.assert_not_awaited()


async def test_unverified_generated_business_claims_fall_back_to_place_facts(
    native, monkeypatch
):
    j = native
    # Bound page text can contain stale offers, advertisements and URLs. It is
    # generation context only and must not leak into the rejected-copy fallback.
    j.pages = [
        dict(
            title="镇江市 " + j.raw["name"],
            link="https://source.fixture.test/cafe",
            mainText="镇江市 "
            + j.raw["name"]
            + " "
            + j.raw["address"]
            + " 广告：https://example.test/promotion",
        )
    ]
    monkeypatch.setattr(
        generation, "invoke_json", AsyncMock(return_value={"supported": False})
    )
    original = generation.invoke_text

    async def unsafe_copy(model, prompt):
        if "候选原始ID" in prompt:
            return await original(model, prompt)
        return json.dumps(
            {
                "text": "🌿 免费全天开放\n这里有窗边座位和免费体验，" * 5
                + "\n\n💡 无需预约\n现在就可以参加。\n\n🧭 实用信息\n每人免费。"
            }
        )

    monkeypatch.setattr(generation, "invoke_text", unsafe_copy)
    r = await j.client.post(
        "/offline/activities/recommend", params={"workspace_id": j.workspace}
    )
    assert r.status_code == 200
    a = r.json()
    assert len(a["description"]) >= 120 and len(a["description"].split("\n\n")) >= 3
    assert j.raw["name"] in a["description"] and j.raw["address"] in a["description"]
    assert "免费" not in a["description"] and "https://" not in a["description"]
    stored = await repo.get_activity(a["id"], j.user)
    assert stored["discovery_metadata"]["copy_status"] == "fallback_rejected"
    detail = await j.client.get("/offline/activities/" + a["id"])
    assert detail.json()["description"] == a["description"]


async def test_empty_album_reselects_same_category_and_serves_real_authenticated_images(
    native, monkeypatch
):
    j = native
    first = native_place({**j.raw, "images": []}, "镇江市")
    second = native_place(
        {
            **j.raw,
            "id": "second-" + j.raw["id"],
            "name": "青石咖啡馆",
            "address": "伯先路18号",
        },
        "镇江市",
    )
    j.selection = first["candidate_id"]
    monkeypatch.setattr(discovery, "discover", AsyncMock(return_value=[first, second]))
    result = await j.client.post(
        "/offline/activities/recommend", params={"workspace_id": j.workspace}
    )
    assert result.status_code == 200, result.text
    a = result.json()
    assert a["location_name"] == second["location_name"] and len(a["image_urls"]) == 3
    assert first["location_name"] not in a["summary"] + a["description"]
    assert a["summary"] != "可以去" + a["location_name"] + "看看"
    assert len(a["description"]) >= 120
    for uri in a["image_urls"]:
        image = await j.client.get(uri)
        assert image.status_code == 200 and image.headers["content-type"].startswith(
            "image/"
        )
        picture = Image.open(io.BytesIO(image.content))
        picture.verify()
    detail = await j.client.get("/offline/activities/" + a["id"])
    assert (
        detail.json()["summary"] == a["summary"]
        and detail.json()["description"] == a["description"]
    )
    stored = await repo.get_activity(a["id"], j.user)
    component = build_activity_component_card(stored, status_label="待确定")
    assert (
        component["title"] == a["title"]
        and component["payload"]["image_urls"] == a["image_urls"]
    )
    generation.tavily_search.assert_not_awaited()
    images.tavily_place_images.assert_not_awaited()


async def test_expired_events_agree_in_deep_links_and_lists_but_started_journeys_survive(
    native,
):
    j = native
    place = native_place(j.raw, "镇江市")
    end = datetime.now(UTC) - timedelta(minutes=1)
    metadata = {
        **place["discovery_metadata"],
        "kind": "event",
        "time_precision": "datetime",
        "event": dict(
            title="昨晚音乐会",
            status="scheduled",
            starts_at=(end - timedelta(hours=2)).isoformat(),
            ends_at=end.isoformat(),
        ),
    }

    async def create(reached):
        activity = await repo.create_activity(
            {
                **place,
                "title": "昨晚音乐会",
                "user_id": j.user,
                "agent_id": j.agent,
                "workspace_id": j.workspace,
                "conversation_id": j.conversation,
                "status": "accepted",
                "starts_at": metadata["event"]["starts_at"],
                "ends_at": end.isoformat(),
                "expires_at": end.isoformat(),
                "discovery_metadata": metadata,
            }
        )
        if reached:
            await j.db.execute_raw(
                "UPDATE offline_activity_recommendations SET reached=TRUE,arrival_confirmed_at=CURRENT_TIMESTAMP WHERE id=$1",
                activity["id"],
            )
        return activity

    expired = await create(False)
    started = await create(True)
    direct = await j.client.get("/offline/activities/" + expired["id"])
    assert direct.status_code == 200 and direct.json()["status"] == "expired"
    listing = await repo.list_activities(j.user, j.workspace)
    assert next(a for a in listing if a["id"] == expired["id"])["status"] == "expired"
    assert next(a for a in listing if a["id"] == started["id"])["status"] == "accepted"
    assert (await post(j, expired, "accept")).status_code == 409
    assert (await post(j, started, "archive")).status_code == 200


async def test_flutter_native_recommendation_gps_and_review_over_real_http(native):
    import os
    import shutil
    import socket
    import uvicorn
    from app.services.auth import create_jwt

    if not os.getenv("OFFLINE_FLUTTER_E2E"):
        pytest.skip(
            "Set OFFLINE_FLUTTER_E2E=1 and OFFLINE_FLUTTER_PROJECT for cross-client E2E"
        )
    assert shutil.which("flutter")
    j = native
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(j.app, log_level="error", lifespan="off"))
    running = asyncio.create_task(server.serve(sockets=[sock]))
    process = None
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.02)
        assert server.started
        env = {
            **os.environ,
            "OFFLINE_E2E_API": f"http://127.0.0.1:{port}",
            "OFFLINE_E2E_TOKEN": create_jwt(j.user, role="user"),
            "OFFLINE_E2E_WORKSPACE": j.workspace,
            "OFFLINE_E2E_PLACE": j.raw["name"],
        }
        process = await asyncio.create_subprocess_exec(
            "flutter",
            "test",
            "--no-pub",
            "test/offline_cleversee_api_e2e_test.dart",
            "-r",
            "expanded",
            cwd=Path(os.environ["OFFLINE_FLUTTER_PROJECT"]),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=120)
        assert process.returncode == 0, output.decode()
        assert "All tests passed" in output.decode()
        generation.tavily_search.assert_not_awaited()
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        server.should_exit = True
        await running
