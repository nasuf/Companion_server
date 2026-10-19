"""Shared timestamp parsing: naive database timestamps are UTC."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app.config import settings


def parse_time(value: object) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(dt, datetime):
            return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def local_clock(value: object) -> str:
    dt = parse_time(value)
    if dt is None:
        return "未知"
    return dt.astimezone(ZoneInfo(settings.schedule_timezone)).strftime("%Y-%m-%d %H:%M %Z")
