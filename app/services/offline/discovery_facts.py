"""Validate native identities and cited event sessions independently of the LLM."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime, timedelta, timezone

from app.config import settings
from app.services.offline.activity_discovery import category_for
from app.services.offline.content import canonical_url, concrete_place_name, normalized
from app.services.offline.geocode import haversine_m, wgs84_to_gcj02


def native_place(
    raw: dict, city: str, *, category=None, center=None, exact_name: str = ""
) -> dict | None:
    name, identity = str(raw.get("name") or "").strip(), str(raw.get("id") or "")
    actual_city, address = str(raw.get("cityName") or ""), str(raw.get("address") or "")
    expected = normalized(city).removesuffix("市")
    if (
        not identity
        or not concrete_place_name(name, city)
        or not address
        or not expected
        or expected
        not in {
            normalized(actual_city).removesuffix("市"),
            normalized(raw.get("districtName")).removesuffix("市"),
        }
    ):
        return None

    def local_name(value):
        return normalized(value).removeprefix(expected).removeprefix("市")

    if exact_name and local_name(name) != local_name(exact_name):
        # Branches, internal entrances and nearby businesses are distinct identities.
        return None
    types = str(raw.get("types") or "")
    if category and not any(
        normalized(k) in normalized(name + " " + types) for k in category.keywords
    ):
        return None
    # A provider may tag a noodle shop as a cafe. A concrete category in its
    # native name wins when presenting it; search intent is not a venue fact.
    category = category_for(name) or category
    try:
        lat, lng = float(raw["latitude"]), float(raw["longitude"])
        if (
            not math.isfinite(lat + lng)
            or not (-90 <= lat <= 90 and -180 <= lng <= 180)
            or (lat == 0 and lng == 0)
        ):
            return None
    except (ValueError, TypeError, KeyError):
        return None
    crs = settings.cleversee_poi_coordinate_system
    if crs not in {"gcj02", "wgs84"}:
        return None
    distance = None
    if center:
        point = wgs84_to_gcj02(*center) if crs == "gcj02" else center
        distance = haversine_m(*point, lat, lng)
        if distance > settings.offline_discovery_radius_m:
            return None
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return dict(
        candidate_id="poi:" + identity,
        location_name=name,
        city=city,
        address=address,
        category=category.name if category else types,
        suitable=category.suitable if category else "走走看看",
        place_lat=lat,
        place_lng=lng,
        native_poi_id=identity,
        native_images=raw.get("images") or [],
        official_url="https://www.amap.com/detail/" + identity,
        title=name,
        summary=f"可以去{name}看看",
        description=f"{name}，{address}。",
        discovery_metadata=dict(
            provider="cleversee",
            kind="place",
            poi_id=identity,
            coordinate_system=crs,
            coordinate_source="native_poi",
            distance_m=round(distance) if distance is not None else None,
            opening_hours=str(
                metadata.get("dailyOpeningHours")
                or metadata.get("weeklyOpeningDays")
                or ""
            )[:240],
            price_info=str(metadata.get("averageSpend") or "")[:120],
            checked_at=datetime.now(UTC).isoformat(),
        ),
    )


def session_key(venue_id: str, event: dict) -> str:
    dates = [
        datetime.fromisoformat(event[k].replace("Z", "+00:00"))
        .astimezone(UTC)
        .isoformat()
        for k in ("starts_at", "ends_at")
    ]
    raw = "|".join([venue_id, normalized(event["title"]), *dates])
    return "event:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _dates(text: str, year: int) -> set[str]:
    result = set()
    # Explicit year + same-year/same-month shorthand are supported. Ambiguous
    # month-only dates and inferred years from publication timestamps are not.
    month = None
    pattern = r"(?:(20\d{2})[年./-])?(\d{1,2})[月./-](\d{1,2})(?:日|号)?|(?<=[至到—～~-])(\d{1,2})(?:日|号)"
    for match in re.finditer(pattern, re.sub(r"\s+", "", text)):
        if match[4]:
            if month is None:
                continue
            y, m, d = year, month, int(match[4])
        else:
            y, m, d = int(match[1] or year), int(match[2]), int(match[3])
            month = m
        try:
            result.add(datetime(y, m, d).date().isoformat())
        except ValueError:
            pass
    english = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    for m, d, y in re.findall(
        r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:,?\s+(20\d{2}))?", text
    ):
        if m[:3].lower() in english:
            try:
                result.add(
                    datetime(int(y or year), english[m[:3].lower()], int(d))
                    .date()
                    .isoformat()
                )
            except ValueError:
                pass
    return result


def validate_event(
    raw: dict,
    text: str,
    *,
    city: str,
    source_url: str,
    now: datetime | None = None,
    allow_inactive: bool = False,
) -> dict | None:
    try:
        title, venue = str(raw["title"]).strip(), str(raw["venue_name"]).strip()
        year = int(raw["year"])
        quote = str(raw["schedule_evidence"]).strip()
        year_quote = str(raw["year_evidence"]).strip()
        identity_quote = str(raw["identity_evidence"]).strip()
        status = raw.get("status")
        precision = raw.get("time_precision")
        end_known = raw.get("end_time_known", True)
        start, end = (
            datetime.fromisoformat(str(raw[k]).replace("Z", "+00:00"))
            for k in ("starts_at", "ends_at")
        )
        current = now or datetime.now(UTC)
        if (
            status
            not in (
                {"scheduled", "cancelled", "postponed"}
                if allow_inactive
                else {"scheduled"}
            )
            or precision not in {"date", "datetime"}
            or not title
            or not concrete_place_name(venue, city)
            or not canonical_url(source_url)
            or not all(
                q and re.sub(r"\s+", "", q) in re.sub(r"\s+", "", text)
                for q in (quote, year_quote, identity_quote)
            )
            or str(year) not in year_quote
            or normalized(title) not in normalized(identity_quote)
            or normalized(venue) not in normalized(identity_quote)
            or normalized(city).removesuffix("市")
            not in normalized(str(raw.get("city") or ""))
            or start.utcoffset() != timedelta(hours=8)
            or end.utcoffset() != timedelta(hours=8)
            or start.year != year
            or end.year not in {year, year + 1}
            or not isinstance(end_known, bool)
            or end <= current
            or end <= start
            or start > current + timedelta(days=settings.offline_event_horizon_days)
        ):
            return None
        mentioned = _dates(quote, year)
        if normalized(title) not in normalized(year_quote) and not (
            re.search(r"20\d{2}[年./-]\d{1,2}[月./-]\d{1,2}", year_quote)
            and start.date().isoformat() in _dates(year_quote, year)
        ):
            return None  # A footer/publication year is not the event edition.
        if (
            start.date().isoformat() not in mentioned
            or end.date().isoformat() not in mentioned
        ):
            return None
        if status != "scheduled":
            proof = str(raw.get("status_evidence") or "")
            if (
                not proof
                or proof not in text
                or str(year) not in proof
                or normalized(title) not in normalized(proof)
                or not re.search(
                    r"取消|延期|改期|暂停|cancell?ed|postponed", proof, re.I
                )
            ):
                return None
        elif re.search(
            r"取消|延期|改期|暂停|待定|cancell?ed|postponed",
            quote + identity_quote,
            re.I,
        ):
            return None
        if precision == "datetime":
            clocks = {
                (int(h), int(m))
                for h, m in re.findall(r"(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)", quote)
            }
            if (start.hour, start.minute) not in clocks:
                return None
            if end_known and (end.hour, end.minute) not in clocks:
                return None
            if not end_known and (
                end.date() != start.date()
                or (end.hour, end.minute, end.second) != (23, 59, 59)
            ):
                return None
        elif (
            start.hour,
            start.minute,
            start.second,
            end.hour,
            end.minute,
            end.second,
        ) != (0, 0, 0, 23, 59, 59):
            return None
        daily = raw.get("daily_hours") or []
        if daily:
            if (
                len(daily) != 2
                or not all(
                    re.fullmatch(r"\d{2}:\d{2}", h) and h in quote for h in daily
                )
                or daily[1] <= daily[0]
            ):
                return None
            for clock in daily:
                datetime.strptime(clock, "%H:%M")
        return dict(
            title=title,
            venue_name=venue,
            city=city,
            starts_at=start.isoformat(),
            ends_at=end.isoformat(),
            time_precision=precision,
            end_time_known=end_known,
            daily_hours=daily,
            status=status,
            official_url=canonical_url(source_url),
            schedule_evidence=quote,
            year_evidence=year_quote,
            identity_evidence=identity_quote,
            description=(
                str(raw.get("description") or "")[:600]
                if str(raw.get("description") or "").strip() in text
                else quote
            ),
            checked_at=current.isoformat(),
            status_evidence=str(raw.get("status_evidence") or ""),
        )
    except (ValueError, TypeError, KeyError, OverflowError):
        return None


def event_is_available(
    activity: dict, *, arrival: bool = False, now: datetime | None = None
) -> bool:
    metadata = activity.get("discovery_metadata") or {}
    if metadata.get("kind") != "event":
        return True
    event = metadata.get("event") or {}
    try:
        current = now or datetime.now(UTC)
        start, end = (
            datetime.fromisoformat(str(activity[k]).replace("Z", "+00:00")).astimezone(
                timezone(timedelta(hours=8))
            )
            for k in ("starts_at", "ends_at")
        )
        if event.get("status") != "scheduled" or end <= current:
            return False
        daily = event.get("daily_hours") or []
        local = current.astimezone(start.tzinfo)
        if daily and local.date() == end.date() and local.strftime("%H:%M") >= daily[1]:
            return False
        if arrival:
            # Allow arrival up to an hour before a timed session; planning is
            # allowed earlier, but it is not evidence of attending the event.
            if current < start - timedelta(hours=1):
                return False
            if daily:
                if not daily[0] <= local.strftime("%H:%M") < daily[1]:
                    return False
        return True
    except (ValueError, TypeError, KeyError):
        return False
