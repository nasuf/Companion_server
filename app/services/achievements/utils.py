"""Shared text and time helpers for achievement evaluation."""

from __future__ import annotations

import json
import unicodedata
from datetime import datetime, time, timedelta, timezone
from typing import Any

LOCAL_TZ = timezone(timedelta(hours=8))
QUESTION_END = ("?", "？")


def _field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _json(value: dict | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        value = datetime.fromisoformat(normalized)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _local(value: datetime | None = None) -> datetime:
    return _aware(value or _now()).astimezone(LOCAL_TZ)


def _day_bounds(local_day: datetime | None = None) -> tuple[datetime, datetime]:
    local_dt = local_day or _local()
    start_local = datetime.combine(local_dt.date(), time.min, tzinfo=LOCAL_TZ)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def count_chars(text: str) -> int:
    """Count Unicode letters and numbers for achievement length rules."""
    return sum(1 for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def _normalized_message(text: str) -> str:
    return "".join(ch for ch in text.strip() if unicodedata.category(ch)[0] in {"L", "N"})


def _first_counted_char(text: str) -> str:
    for ch in text.strip():
        if unicodedata.category(ch)[0] in {"L", "N"}:
            return ch
    return ""


def _has_symbol_or_punctuation(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in {"P", "S"} for ch in text if not ch.isspace())


def _has_emoji(text: str) -> bool:
    return any(ord(ch) >= 0x1F000 or unicodedata.category(ch) == "So" for ch in text)
