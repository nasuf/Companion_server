"""P2 到达校验/预言的纯逻辑单测（不触 DB）。

覆盖 spec §3.3/§4.5 的 ≤200m 直线距离门槛与无坐标降级策略，以及预言池取值。
"""

import pytest
from fastapi import HTTPException

from app.config import settings
from app.services.offline import activity_service, prophecy
from app.services.offline.geocode import haversine_m, make_place_key

_BASE_LAT, _BASE_LNG = 31.2304, 121.4737  # 上海人广附近
_METERS_PER_DEG_LAT = 111_320.0


def _place(lat, lng):
    return {"place_lat": lat, "place_lng": lng}


def _lat_offset_by_meters(meters: float) -> float:
    return _BASE_LAT + meters / _METERS_PER_DEG_LAT


def test_same_point_within_radius():
    # 同点 0m，不抛
    activity_service._verify_arrival_distance(
        _place(_BASE_LAT, _BASE_LNG), _BASE_LAT, _BASE_LNG
    )


def test_boundary_just_inside_200m():
    activity_service._verify_arrival_distance(
        _place(_BASE_LAT, _BASE_LNG), _lat_offset_by_meters(190), _BASE_LNG
    )


def test_boundary_just_outside_200m():
    with pytest.raises(HTTPException) as exc:
        activity_service._verify_arrival_distance(
            _place(_BASE_LAT, _BASE_LNG), _lat_offset_by_meters(260), _BASE_LNG
        )
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "too_far"
    assert exc.value.detail["distance_m"] > 200


def test_no_geocode_allows_when_not_required(monkeypatch):
    monkeypatch.setattr(settings, "offline_arrival_require_geocode", False)
    # 无坐标 + 不强制 -> 放行（不抛）
    activity_service._verify_arrival_distance(
        _place(None, None), _BASE_LAT, _BASE_LNG
    )


def test_no_geocode_blocks_when_required(monkeypatch):
    monkeypatch.setattr(settings, "offline_arrival_require_geocode", True)
    with pytest.raises(HTTPException) as exc:
        activity_service._verify_arrival_distance(
            _place(None, None), _BASE_LAT, _BASE_LNG
        )
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "no_geocode"


def test_haversine_meter_accuracy():
    # ~200m 北移，haversine 应落在 199~201
    d = haversine_m(_BASE_LAT, _BASE_LNG, _lat_offset_by_meters(200), _BASE_LNG)
    assert 199 <= d <= 201


def test_prophecy_always_from_pool():
    for _ in range(30):
        assert prophecy.pick_prophecy() in prophecy.PROPHECY_POOL


def test_place_key_stable_and_normalized():
    a = make_place_key("佛山植物园", "广东省佛山市禅城区东鄱南路1号", "佛山")
    b = make_place_key("佛山植物园 ", "广东省佛山市禅城区东鄱南路1号", "佛山")
    assert a == b and a is not None
    assert make_place_key(None, None, None) is None


async def test_manual_confirmation_cannot_bypass_gps_for_located_place(monkeypatch):
    from unittest.mock import AsyncMock
    activity = {"id": "a", "user_id": "u", "status": "accepted", "reached": False,
                "place_lat": _BASE_LAT, "place_lng": _BASE_LNG}
    monkeypatch.setattr(activity_service.repo, "get_activity", AsyncMock(return_value=activity))
    monkeypatch.setattr(activity_service.repo, "find_other_reached_activity", AsyncMock(return_value=None))
    mark = AsyncMock()
    monkeypatch.setattr(activity_service.repo, "mark_arrived", mark)
    with pytest.raises(HTTPException) as exc:
        await activity_service.arrive_activity("u", "a", manual_confirmation=True)
    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "location_required"
    with pytest.raises(HTTPException) as exc:
        await activity_service.arrive_activity(
            "u", "a", lat=_lat_offset_by_meters(260), lng=_BASE_LNG, accuracy_m=10,
            manual_confirmation=True,
        )
    assert exc.value.detail["reason"] == "too_far"
    mark.assert_not_awaited()


@pytest.mark.parametrize("source", ["manual", "scheduled"])
async def test_recommendation_without_coordinates_is_never_created(monkeypatch, source):
    from unittest.mock import AsyncMock
    service = activity_service
    monkeypatch.setattr(service.repo, "resolve_user_context", AsyncMock(return_value={
        "conversation_id": "c", "agent_id": "ag", "workspace_id": "w",
        "user_location_city": "镇江市"}))
    monkeypatch.setattr(service, "generate_activity_card", AsyncMock(return_value={
        "title": "折光咖啡", "location_name": "折光咖啡", "address": "某路1号", "discovery_metadata": {}}))
    monkeypatch.setattr(service, "geocode_address", AsyncMock(return_value=None))
    create = AsyncMock()
    monkeypatch.setattr(service.repo, "create_activity", create)
    emit = AsyncMock()
    monkeypatch.setattr(service, "emit_assistant", emit)
    if source == "manual":
        with pytest.raises(HTTPException) as exc:
            await service.create_recommendation_for_user(user_id="u", workspace_id="w", source=source)
        assert exc.value.detail["reason"] == "no_suitable_activity"
    else:
        assert await service.create_recommendation_for_user(user_id="u", workspace_id="w", source=source) is None
    create.assert_not_awaited()
    emit.assert_not_awaited()


async def test_generation_skips_ungeocodable_place_for_the_next_candidate(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services.offline import activity_generation as gen
    from app.services.offline.providers.search import SearchResult
    monkeypatch.setattr(gen.settings, "offline_search_provider", "tavily")
    monkeypatch.setattr(gen, "get_chat_model", lambda: object())
    lost = SearchResult(title="折光咖啡", url="https://example.com/lost", content="镇江市某路1号 折光咖啡")
    found = SearchResult(title="镇江博物馆", url="https://example.com/museum", content="镇江市润州区伯先路85号 镇江博物馆")
    monkeypatch.setattr(gen.repo, "list_user_tags", AsyncMock(return_value=[]))
    monkeypatch.setattr(gen.repo, "memory_brief", AsyncMock(return_value=""))
    monkeypatch.setattr(gen.repo, "list_recent_activity_fingerprints", AsyncMock(return_value=[]))
    monkeypatch.setattr(gen, "_search_activity_candidates", AsyncMock(return_value=([lost, found], [lost, found], "q")))
    monkeypatch.setattr(gen, "get_prompt_text", AsyncMock(return_value="{sources_json} {avoid_text}"))
    monkeypatch.setattr(gen, "invoke_text", AsyncMock(return_value="{}"))
    monkeypatch.setattr(gen, "persist_activity_images", AsyncMock(return_value=[]))
    monkeypatch.setattr(gen, "_recommendation_copy", AsyncMock(return_value=""))

    async def geocode(address, city=None):
        return (32.21, 119.43) if address == "镇江博物馆" else None

    monkeypatch.setattr(gen, "geocode_address", geocode)
    card = await gen.generate_activity_card(user_id="u", workspace_id=None, city="镇江市", source="manual")
    assert card["location_name"] == "镇江博物馆"
    assert (card["place_lat"], card["place_lng"]) == (32.21, 119.43)
