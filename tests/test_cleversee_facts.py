from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings
from app.services.offline.activity_discovery import category_for
from app.services.offline.discovery_facts import (
    native_place,
    validate_event,
    session_key,
    event_is_available,
)
from app.services.offline.event_schedule import schedule_label
from app.services.offline.geocode import wgs84_to_gcj02

NOW = datetime(2026, 10, 8, 12, tzinfo=timezone(timedelta(hours=8)))
TEXT = "2026年秋日音乐会在小岛音乐厅举办。时间：2026年10月9日19:00—21:00。"
EVENT = dict(
    title="秋日音乐会",
    venue_name="小岛音乐厅",
    city="镇江",
    year=2026,
    starts_at="2026-10-09T19:00:00+08:00",
    ends_at="2026-10-09T21:00:00+08:00",
    time_precision="datetime",
    daily_hours=[],
    status="scheduled",
    identity_evidence="2026年秋日音乐会在小岛音乐厅举办",
    year_evidence="2026年秋日音乐会",
    schedule_evidence="2026年10月9日19:00—21:00",
)


def poi(**change):
    lat, lng = wgs84_to_gcj02(32.21, 119.43)
    return {
        **dict(
            id="B1",
            name="小岛咖啡(伯先路店)",
            cityName="镇江市",
            address="伯先路12号",
            types="餐饮服务|咖啡厅",
            latitude=str(lat),
            longitude=str(lng),
            images=[],
        ),
        **change,
    }


@pytest.mark.parametrize(
    "change",
    [
        dict(id=""),
        dict(name="镇江这些咖啡店"),
        dict(cityName="南京市"),
        dict(latitude="nan"),
        dict(longitude="inf"),
        dict(latitude="91"),
        dict(address=""),
        dict(name="咖啡馆"),
    ],
)
def test_native_poi_rejects_unusable_facts(change):
    assert (
        native_place(
            poi(**change),
            "镇江",
            category=category_for("咖啡馆"),
            center=(32.21, 119.43),
        )
        is None
    )


def test_native_distance_is_calculated_from_phone_center_and_exact_branch():
    found = native_place(
        poi(distanceMeter="0"),
        "镇江",
        center=(32.21, 119.43),
        exact_name="小岛咖啡(伯先路店)",
    )
    assert found["discovery_metadata"]["distance_m"] == 0
    assert native_place(poi(distanceMeter="1"), "镇江", center=(31.21, 119.43)) is None
    assert native_place(poi(), "镇江", exact_name="小岛咖啡(万达店)") is None
    assert native_place(poi(), "镇江", category=category_for("图书馆")) is None


@pytest.mark.parametrize(
    "name",
    [
        "图书馆",
        "书店",
        "博物馆",
        "美术馆",
        "咖啡馆",
        "茶馆",
        "公园",
        "饭馆",
        "小吃",
        "苍蝇馆",
        "音乐厅",
        "市集",
    ],
)
def test_catalog_is_an_open_regional_query_policy(name):
    c = category_for(name)
    assert c is not None
    assert native_place(poi(name="小岛" + name, types=name), "镇江", category=c)


@pytest.mark.parametrize(
    "change",
    [
        dict(year=2025),
        dict(status="cancelled"),
        dict(city="南京"),
        dict(venue_name="另一音乐厅"),
        dict(starts_at="2026-10-10T19:00:00+08:00"),
        dict(ends_at="2026-10-09T22:00:00+08:00"),
        dict(starts_at="2026-10-09T19:00:00Z"),
        dict(schedule_evidence="凭空生成的日期"),
        dict(identity_evidence="不在原文里"),
    ],
)
def test_event_requires_edition_date_time_venue_and_verbatim_evidence(change):
    assert (
        validate_event(
            {**EVENT, **change},
            TEXT,
            city="镇江",
            source_url="https://organizer.example/event",
            now=NOW,
        )
        is None
    )


def test_event_sessions_time_precision_expiry_and_early_arrival():
    event = validate_event(
        EVENT, TEXT, city="镇江", source_url="https://organizer.example/event", now=NOW
    )
    assert event and schedule_label(event) == "2026/10/09 19:00—21:00"
    a = dict(
        starts_at=event["starts_at"],
        ends_at=event["ends_at"],
        discovery_metadata={"kind": "event", "event": event},
    )
    assert event_is_available(a, now=NOW)
    assert not event_is_available(a, arrival=True, now=NOW)
    assert event_is_available(a, arrival=True, now=NOW + timedelta(days=1, hours=6))
    assert not event_is_available(a, now=NOW + timedelta(days=1, hours=9))
    assert session_key("B1", event) != session_key(
        "B1", {**event, "starts_at": "2026-10-10T19:00:00+08:00"}
    )


def test_multi_day_event_daily_closing_is_not_midnight_or_whole_day():
    text = (
        "2026年秋日市集在小岛音乐厅举办，展期2026年10月8日—10月9日，每日10:00—21:00。"
    )
    raw = {
        **EVENT,
        "title": "秋日市集",
        "starts_at": "2026-10-08T00:00:00+08:00",
        "ends_at": "2026-10-09T23:59:59+08:00",
        "identity_evidence": "2026年秋日市集在小岛音乐厅举办",
        "year_evidence": "2026年秋日市集",
        "schedule_evidence": "2026年10月8日—10月9日，每日10:00—21:00",
        "time_precision": "date",
        "daily_hours": ["10:00", "21:00"],
    }
    event = validate_event(
        raw, text, city="镇江", source_url="https://organizer.example/market", now=NOW
    )
    assert event and "00:00" not in schedule_label(event)
    activity = dict(
        starts_at=event["starts_at"],
        ends_at=event["ends_at"],
        discovery_metadata={"kind": "event", "event": event},
    )
    assert not event_is_available(activity, arrival=True, now=NOW.replace(hour=22))
    assert event_is_available(
        activity, now=NOW.replace(hour=22)
    )  # Tomorrow remains open.
    assert not event_is_available(activity, now=NOW.replace(day=9, hour=22))


def test_cancellation_must_cite_this_edition_not_other_or_old_events():
    raw = {**EVENT, "status": "cancelled", "status_evidence": "2026年秋日音乐会取消"}
    assert validate_event(
        raw,
        TEXT + "2026年秋日音乐会取消",
        city="镇江",
        source_url="https://organizer.example/event",
        now=NOW,
        allow_inactive=True,
    )
    for quote in ["2025年秋日音乐会取消", "2026年别的音乐会取消"]:
        assert not validate_event(
            {**raw, "status_evidence": quote},
            TEXT + quote,
            city="镇江",
            source_url="https://organizer.example/event",
            now=NOW,
            allow_inactive=True,
        )


def test_provider_coordinate_system_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "cleversee_poi_coordinate_system", "unknown")
    assert native_place(poi(), "镇江") is None


def test_county_level_city_uses_native_district_without_accepting_other_counties():
    raw = poi(cityName="镇江市", districtName="丹阳市")
    assert native_place(raw, "丹阳市")
    assert not native_place(raw, "句容市")


def test_search_intent_does_not_relabel_a_named_noodle_shop_as_a_cafe():
    found = native_place(
        poi(name="金山寺素面馆", types="餐饮服务|咖啡厅"),
        "镇江",
        category=category_for("咖啡馆"),
    )
    assert found["category"] == "餐饮小店"


def test_footer_year_cannot_relabel_old_event_as_current():
    assert not validate_event(
        {**EVENT, "year_evidence": "版权所有2026"},
        TEXT + "版权所有2026",
        city="镇江",
        source_url="https://organizer.example/event",
        now=NOW,
    )


def test_start_only_announcement_never_invents_finish_time_or_expires_at_opening():
    quote = "2026年10月9日19:00开演"
    raw = {
        **EVENT,
        "schedule_evidence": quote,
        "end_time_known": False,
        "ends_at": "2026-10-09T23:59:59+08:00",
    }
    event = validate_event(
        raw,
        TEXT + quote,
        city="镇江",
        source_url="https://organizer.example/event",
        now=NOW,
    )
    assert event and schedule_label(event) == "2026/10/09 19:00"
    assert not validate_event(
        {**raw, "ends_at": raw["starts_at"]},
        TEXT + quote,
        city="镇江",
        source_url="https://organizer.example/event",
        now=NOW,
    )


def test_whitespace_normalization_preserves_quotes_but_ellipsis_is_not_evidence():
    raw = {**EVENT, "schedule_evidence": "2026 年 10 月 9 日 19:00—21:00"}
    assert validate_event(
        raw, TEXT, city="镇江", source_url="https://organizer.example/event", now=NOW
    )
    assert not validate_event(
        {**raw, "identity_evidence": "2026年秋日音乐会……小岛音乐厅"},
        TEXT,
        city="镇江",
        source_url="https://organizer.example/event",
        now=NOW,
    )
