"""Placeholder values shared by the activity prompt spec."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from app.config import settings

try:
    from zoneinfo import ZoneInfo

    _TZ = ZoneInfo(getattr(settings, "schedule_timezone", "Asia/Shanghai"))
except Exception:  # pragma: no cover - missing tz database
    _TZ = None

_SKIP_DISTRICT = ("景区", "园区", "校区", "小区", "社区")
_ADMIN_BREAKS = "省市州盟"


def filled(value: Any, *, empty: str = "（未提供）") -> str:
    text = str(value or "").strip()
    return text or empty


def format_moment(value: Any) -> str:
    moment = _as_datetime(value)
    if moment is None:
        return ""
    if _TZ is not None:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        moment = moment.astimezone(_TZ)
    return moment.strftime("%Y-%m-%d %H:%M")


def district_from_address(address: str) -> str:
    """Last 区/县 name, stopping at a city marker so 市 is not included."""
    text = address or ""
    found = ""
    for index, char in enumerate(text):
        if char not in "区县":
            continue
        han: list[str] = []
        cursor = index - 1
        while cursor >= 0 and "\u4e00" <= text[cursor] <= "\u9fff" and len(han) < 6:
            if text[cursor] in _ADMIN_BREAKS:
                break
            han.append(text[cursor])
            cursor -= 1
        if not 2 <= len(han) <= 6:
            continue
        district = "".join(reversed(han)) + char
        if any(district.endswith(skip) for skip in _SKIP_DISTRICT):
            continue
        found = district
    return found


def clip_text(text: str, limit: int) -> str:
    """Keep a user-visible passage within `limit` characters, on a break."""
    body = (text or "").strip()
    if limit <= 0 or len(body) <= limit:
        return body
    clipped = body[:limit]
    floor = int(limit * 0.6)
    for separator in ("\n", "。", "！", "？"):
        index = clipped.rfind(separator)
        if index >= floor:
            end = index if separator == "\n" else index + 1
            return clipped[:end].strip()
    return clipped.strip()


def location_fields(
    activity: dict[str, Any],
    *,
    now: datetime | None = None,
    city_fallback: str = "",
) -> dict[str, str]:
    raw_address = str(activity.get("address") or activity.get("location_name") or "")
    scene = filled(
        activity.get("summary") or activity.get("description") or activity.get("vibe"),
        empty="（未提供）",
    )
    current = now or datetime.now(UTC)
    return {
        "city": filled(activity.get("city") or city_fallback),
        "district": filled(district_from_address(raw_address)),
        "activity_name": filled(activity.get("title"), empty="这次外出"),
        "address": filled(raw_address),
        "type": filled(activity.get("category"), empty="线下活动"),
        "scene_description": scene[:400],
        "arrival_time": filled(format_moment(activity.get("arrival_confirmed_at"))),
        "current_time": filled(format_moment(current)),
    }


def parse_text_field(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        if not match:
            return ""
        try:
            return json.loads(f'"{match.group(1)}"').strip()
        except (json.JSONDecodeError, TypeError):
            return match.group(1).strip()
    if isinstance(data, dict):
        return str(data.get("text") or "").strip()
    return ""


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None
