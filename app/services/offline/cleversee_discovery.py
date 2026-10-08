"""Regional discovery: immutable POI facts + separately verified event sessions."""

from __future__ import annotations

import asyncio
import logging
import hashlib
import math
from datetime import UTC, datetime
from urllib.parse import urlsplit

from app.services.llm.models import invoke_json
from app.services.offline.llm import get_offline_chat_model as get_chat_model
from app.services.offline.activity_discovery import ordered_categories
from app.services.offline.content import normalized
from app.services.offline.discovery_facts import (
    native_place,
    session_key,
    validate_event,
)
from app.services.offline.providers import cleversee
from app.services.offline.public_cache import public_cached
from app.config import settings
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


async def extract_events(
    text: str,
    *,
    city: str,
    url: str,
    now: datetime | None = None,
    allow_inactive: bool = False,
) -> list[dict]:
    current = now or datetime.now(UTC)
    template = await get_prompt_text("offline.event_extract")
    prompt = template.format(
        city=city, now=current.isoformat(), source_url=url, source_text=text[:18000]
    )
    try:

        async def load():
            return await invoke_json(get_chat_model(), prompt)

        # Public source extraction is reusable; its time/venue validation is not.
        # A fresh announcement or Web prompt version produces a different key.
        result = await public_cached(
            "event-extraction",
            [
                city,
                url,
                hashlib.sha256(text[:18000].encode()).hexdigest(),
                hashlib.sha256(str(template).encode()).hexdigest(),
                current.date().isoformat(),
                allow_inactive,
            ],
            load,
            valid=lambda value: (
                isinstance(value, dict) and isinstance(value.get("events"), list)
            ),
            ttl=lambda value: 180 if value["events"] else 30,
            timeout=16,
        )
        rows = result.get("events", []) if isinstance(result, dict) else []
        return [
            event
            for row in rows[:12]
            if isinstance(row, dict)
            and (
                event := validate_event(
                    row,
                    text,
                    city=city,
                    source_url=url,
                    now=current,
                    allow_inactive=allow_inactive,
                )
            )
        ]
    except Exception as exc:
        logger.info("[offline-events] extraction unavailable (%s)", type(exc).__name__)
        return []


async def _events(city: str, center, *, on_event=None) -> list[dict]:
    now = datetime.now(UTC)
    # Month queries recall announcements better than an exact date-range string;
    # the explicit horizon is applied by validate_event, never by search ranking.
    base = f"{city} {now:%Y年%m月}"
    batches = await asyncio.gather(
        *(
            cleversee.web_search(
                base + " " + kind + " 活动时间 举办地点", max_results=8
            )
            for kind in ("音乐会 演唱会 演出", "市集 跳蚤市场 展览 讲座")
        )
    )
    pages = []
    seen_pages = set()
    for row in range(max(map(len, batches), default=0)):
        for batch in batches:
            if row < len(batch) and batch[row].url not in seen_pages:
                seen_pages.add(batch[row].url)
                pages.append(batch[row])

    def authority(page):
        host = (urlsplit(page.url).hostname or "").lower()
        return (
            0
            if host.endswith(".gov.cn") or host in {"arts.cctv.com", "www.cctv.com"}
            else 1
        )

    pages = [
        p
        for p in pages
        if not any(
            word in p.url
            for word in (
                "baike.baidu.com",
                "wikipedia.org",
                "page.sm.cn",
                "bilibili.com",
            )
        )
    ]
    pages.sort(key=authority)
    output = []
    # Diverse candidate pages are useful, but only verified event/venue pairs
    # can become recommendations. Official sources are read before extraction.
    for page in pages[:3]:
        body = await cleversee.read_page(page.url)
        if not body:
            continue
        for event in await extract_events(
            page.title + "\n" + body, city=city, url=page.url, now=now
        ):
            name = event["venue_name"]
            raw_places = await cleversee.places(f"{city} {name} 地址 经纬度", limit=10)
            matches = [
                p
                for raw in raw_places
                if (p := native_place(raw, city, center=center, exact_name=name))
            ]
            if not matches:
                raw_places = await cleversee.qa_places(
                    f"{city} {name} 的准确地点卡片",
                    lat=center[0] if center else None,
                    lng=center[1] if center else None,
                )
                matches = [
                    p
                    for raw in raw_places
                    if (p := native_place(raw, city, center=center, exact_name=name))
                ]
            # A non-empty response is not proof of the requested venue. An
            # ambiguous same-name venue is withheld instead of choosing row 1.
            unique = {p["native_poi_id"]: p for p in matches}
            if len(unique) != 1:
                continue
            place = next(iter(unique.values()))
            candidate_id = session_key(place["native_poi_id"], event)
            expiry = datetime.fromisoformat(event["ends_at"])
            if event.get("daily_hours"):
                hour, minute = map(int, event["daily_hours"][1].split(":"))
                expiry = expiry.replace(hour=hour, minute=minute, second=0)
            output.append(
                {
                    **place,
                    "candidate_id": candidate_id,
                    "title": event["title"],
                    "summary": f"{event['title']}在{place['location_name']}举办",
                    "category": "限时活动",
                    "description": event["description"] or event["schedule_evidence"],
                    "starts_at": event["starts_at"],
                    "ends_at": event["ends_at"],
                    "expires_at": expiry.isoformat(),
                    "official_url": event["official_url"],
                    "discovery_metadata": {
                        **place["discovery_metadata"],
                        "kind": "event",
                        "time_precision": event["time_precision"],
                        "event": event,
                        "session_key": candidate_id,
                    },
                }
            )
            if on_event:
                on_event(output[-1])
    return output


def regional_query(
    city: str, search_anchor: str, category, center=None
) -> tuple[str, tuple | None]:
    """Stable public regional query, followed by exact per-user radius checks."""
    query = f"{normalized(search_anchor) or normalized(city)} {category.query_hint}"
    if not center:
        return query, None
    # Include the cell's margin in search; final distance checks still use the
    # actual device position rather than this shared query center.
    lat, lng = (math.floor(float(value) * 100) / 100 + 0.005 for value in center)
    radius_km = math.ceil(settings.offline_discovery_radius_m / 1000) + 1
    query += f" 距离坐标纬度{lat:.3f}经度{lng:.3f}{radius_km}公里以内"
    return query, (lat, lng)


async def discover(
    *, city: str, search_anchor: str, tags: list[str], recent: list[dict], center=None
) -> list[dict]:
    categories = [
        c for c in ordered_categories(recent, tags, city) if c.family != "限时"
    ][:6]
    output = []

    async def category_places(category):
        query, query_center = regional_query(city, search_anchor, category, center)
        rows = await cleversee.places(query)
        candidates = [
            p
            for raw in rows
            if (p := native_place(raw, city, category=category, center=center))
        ]
        if not candidates:
            rows = await cleversee.places(f"{city} {category.query_hint}")
            candidates = [
                p
                for raw in rows
                if (p := native_place(raw, city, category=category, center=center))
            ]
        if not candidates and category.family == "文化":
            rows = await cleversee.qa_places(
                f"{search_anchor}附近的{category.query_hint}，返回具体地点卡片",
                lat=query_center[0] if query_center else None,
                lng=query_center[1] if query_center else None,
            )
            candidates = [
                p
                for raw in rows
                if (p := native_place(raw, city, category=category, center=center))
            ]
        # Preserve category ordering regardless of network completion ordering.
        return categories.index(category), candidates[:5]

    tasks = [asyncio.create_task(category_places(c)) for c in categories]
    event_bucket = []
    event_task = asyncio.create_task(
        _events(city, center, on_event=event_bucket.append)
    )
    batches = {}
    try:
        async with asyncio.timeout(42):
            for task in asyncio.as_completed(tasks + [event_task]):
                try:
                    result = await task
                except Exception as exc:
                    logger.info(
                        "[offline-discovery] one intent failed (%s)", type(exc).__name__
                    )
                    continue
                if isinstance(result, tuple):
                    batches[result[0]] = result[1]
                else:
                    batches[-1] = result
    except TimeoutError:
        logger.info(
            "[offline-discovery] deadline reached, retaining completed searches"
        )
    finally:
        for task in tasks + [event_task]:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, event_task, return_exceptions=True)
    if event_bucket:
        batches[-1] = event_bucket
    # Prefer a verified event when the most recent recommendation wasn't one.
    event_first = not recent or recent[0].get("kind") != "event"
    order = (
        ([-1] if event_first else [])
        + list(range(len(categories)))
        + ([] if event_first else [-1])
    )
    seen = set()
    history = {normalized(str(r.get("location_name") or "")) for r in recent[:8]}
    sessions = {r.get("place_key") for r in recent}
    for index in order:
        for card in batches.get(index, []):
            identity = card["candidate_id"]
            is_event = card["discovery_metadata"]["kind"] == "event"
            if (
                identity in seen
                or (is_event and identity in sessions)
                or (not is_event and normalized(card["location_name"]) in history)
            ):
                continue
            seen.add(identity)
            output.append(card)
    return output[:24]


async def refresh_event(activity: dict) -> dict | None:
    """A fresh, source-bound session check; failure never invents cancellation."""
    metadata = activity.get("discovery_metadata") or {}
    if metadata.get("kind") != "event":
        return None
    old = metadata.get("event") or {}
    try:
        async with asyncio.timeout(24):
            text = await cleversee.read_page(old.get("official_url", ""), fresh=True)
            if not text:
                return None
            events = await extract_events(
                text,
                city=activity["city"],
                url=old["official_url"],
                allow_inactive=True,
            )
            matching = [
                e
                for e in events
                if normalized(e["title"]) == normalized(old.get("title"))
                and normalized(e["venue_name"]) == normalized(old.get("venue_name"))
                and all(
                    datetime.fromisoformat(e[k].replace("Z", "+00:00"))
                    == datetime.fromisoformat(activity[k].replace("Z", "+00:00"))
                    for k in ("starts_at", "ends_at")
                )
            ]
            if matching:
                return {**metadata, "event": matching[0]}
    except Exception as exc:
        logger.info("[offline-events] refresh unavailable (%s)", type(exc).__name__)
    return None
