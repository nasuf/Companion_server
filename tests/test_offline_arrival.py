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
