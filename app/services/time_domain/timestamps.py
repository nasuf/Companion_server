"""Shared timestamp parsing: naive database timestamps are UTC."""
from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from app.config import settings


def parse_time(value: object) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(dt, datetime):
            return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


EndBoundary = Literal["exclusive", "inclusive_second", "inclusive_microsecond"]


def exclusive_end(value: datetime, boundary: EndBoundary) -> datetime:
    """Convert a parser endpoint to a UTC SQL upper bound without rounding.

    Calendar parsers historically use either the final second or microsecond;
    elapsed offsets already end at the next interval. Never infer this from the
    clock digits: an elapsed offset can also end at xx:59:59.
    """
    stamp = parse_time(value)
    if stamp is None:
        raise ValueError("invalid_interval_end")
    step = {"exclusive": timedelta(0), "inclusive_second": timedelta(seconds=1),
            "inclusive_microsecond": timedelta(microseconds=1)}[boundary]
    return stamp + step


def local_clock(value: object, *, timezone_name: str | None = None) -> str:
    dt = parse_time(value)
    if dt is None:
        return "未知"
    try:
        zone = ZoneInfo(timezone_name or settings.schedule_timezone)
    except (ValueError, ZoneInfoNotFoundError):
        return "未知"
    return dt.astimezone(zone).strftime("%Y-%m-%d %H:%M %Z")
