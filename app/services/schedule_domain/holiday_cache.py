"""In-process cache for holidays loaded from DB.

Design: keep `is_holiday()` synchronous (existing call-sites are sync),
preload the entire table at app startup into a module-level dict, and
provide async `reload()` that admin save / refresh cron triggers.

Data volume is bounded (< 1000 rows covers ~10 years) so a full in-memory
copy is cheap and simplifies the read path.

Lunardate remains the ultimate fallback when a date is outside whatever
years the DB has loaded (e.g. a year beyond the latest data).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import date
from threading import RLock

from app.services.schedule_domain.holiday_repo import (
    HolidayEntry,
    list_holidays,
)

logger = logging.getLogger(__name__)

_lock = RLock()
_by_date: dict[date, list[HolidayEntry]] = {}
_by_name: dict[str, list[date]] = {}
_workday_swaps: set[date] = set()
_loaded: bool = False


async def reload() -> dict[str, int]:
    """Pull all holidays from DB and replace the in-process cache atomically.

    Returns stats for logging. Safe to call concurrently — last writer wins.
    """
    global _loaded
    entries = await list_holidays()

    by_date: dict[date, list[HolidayEntry]] = defaultdict(list)
    by_name: dict[str, list[date]] = defaultdict(list)
    workday_swaps: set[date] = set()
    for e in entries:
        by_date[e.date].append(e)
        if e.is_workday_swap:
            workday_swaps.add(e.date)
        else:
            by_name[e.name].append(e.date)
            # Also record canonical name (e.g. "春节假期" has metadata
            # {canonical_name: "春节"}) so time_parser resolves "春节"
            canonical = (e.metadata or {}).get("canonical_name") if e.metadata else None
            if canonical and canonical != e.name:
                by_name[canonical].append(e.date)

    with _lock:
        _by_date.clear()
        _by_date.update(by_date)
        _by_name.clear()
        _by_name.update(by_name)
        _workday_swaps.clear()
        _workday_swaps.update(workday_swaps)
        _loaded = True

    stats = {
        "total": len(entries),
        "unique_dates": len(by_date),
        "unique_names": len(by_name),
        "workday_swaps": len(workday_swaps),
    }
    logger.info(f"Holiday cache reloaded: {stats}")
    return stats


def is_loaded() -> bool:
    """True once the cache has been populated at least once."""
    return _loaded


def get_by_date(d: date) -> HolidayEntry | None:
    """Return the primary (first) holiday on `d`, or None.

    "Primary" prefers non-workday-swap entries; multi-country on same day
    falls back to insertion order.
    """
    with _lock:
        entries = _by_date.get(d)
    if not entries:
        return None
    # Prefer the non-swap entry (real holiday beats "调休上班")
    for e in entries:
        if not e.is_workday_swap:
            return e
    return entries[0]


def is_workday_swap(d: date) -> bool:
    with _lock:
        return d in _workday_swaps


def list_dates_for_name(name: str) -> list[date]:
    """Used by time_parser to resolve "春节" → list of actual ISO dates."""
    with _lock:
        return sorted(_by_name.get(name, []))


def all_known_names() -> list[str]:
    """Used by time_parser to scan message for any holiday name mention."""
    with _lock:
        return sorted(_by_name.keys())


def invalidate_background() -> None:
    """Schedule a reload without blocking the caller.

    Use this from synchronous paths (e.g. admin API response handlers
    that don't want to hold up the 200 response while the cache warms up).
    The reload runs as a background task on the running event loop; if
    no loop is running, we fall back to scheduling via asyncio.run.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Not in an async context — run synchronously (tests / CLI).
        asyncio.run(reload())
        return
    loop.create_task(reload())
