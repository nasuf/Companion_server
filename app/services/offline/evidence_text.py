"""One rendering of dated user evidence, shared by collection and matching.

A memory such as "用户今天出来走走" is only true on the day it was said. Without
the date, writers turned a 4-day-old memory into "今天不是正打算出来走走嘛",
and reviewers that saw the same undated text could not object. The format
matches the chat history the models already read: ``[YYYY-MM-DD HH:MM] ``.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings

_TZ = ZoneInfo(settings.schedule_timezone)
_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?\] ")


def dated(text: str, when: Any, *, with_time: bool) -> str:
    """Prefix the local date the user said it; undated text stays unchanged."""
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            when = None
    if not isinstance(when, datetime):
        return text
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    local = when.astimezone(_TZ)
    return f"[{local:%Y-%m-%d %H:%M}] {text}" if with_time else f"[{local:%Y-%m-%d}] {text}"


def undated(text: str) -> str:
    return _STAMP.sub("", text, count=1)


def mask_quotes(items: list[str], quotes: list[str]) -> list[str]:
    """Stored diagnostics must not repeat the user's own words (health, mood...).

    Reviewers quote the draft, and drafts quote the evidence; the admin view
    needs the kind of problem, not the private sentence.
    """
    bodies = sorted({undated(quote) for quote in quotes if len(undated(quote)) >= 2}, key=len, reverse=True)
    masked = []
    for item in items:
        for body in bodies:
            item = item.replace(body, "〔用户原文〕")
        masked.append(item)
    return masked
