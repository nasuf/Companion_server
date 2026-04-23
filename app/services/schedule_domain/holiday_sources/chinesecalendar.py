"""Fetch CN legal holidays + workday swaps from the `chinesecalendar` library.

The library ships offline data; each release covers one more year of
State Council announcements. When a target year is beyond the library's
coverage, `get_holiday_detail()` raises `NotImplementedError`; we surface
that as `available=False` so the caller can degrade gracefully.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from app.services.schedule_domain.holiday_repo import (
    SOURCE_CHINESE_CALENDAR,
    HolidayEntry,
)

logger = logging.getLogger(__name__)


def fetch_cn_year(year: int) -> dict[str, Any]:
    """Return `{available, entries, error}`.

    `entries` is a flat list of HolidayEntry (type='legal', source='chinesecalendar');
    workday-swap rows have `is_workday_swap=True` and type='custom' since they're
    not real holidays.
    """
    try:
        import chinese_calendar as cc
    except ImportError as e:
        return {"available": False, "entries": [], "error": f"chinesecalendar not installed: {e}"}

    entries: list[HolidayEntry] = []
    try:
        # First pass: collect raw (date, name) pairs + workday swaps.
        raw_holidays: list[tuple[date, str]] = []
        cur = date(year, 1, 1)
        last = date(year, 12, 31)
        while cur <= last:
            on_holiday, name = cc.get_holiday_detail(cur)
            if on_holiday and name:
                raw_holidays.append((cur, _normalize_cn_name(name)))
            if cur.weekday() >= 5 and cc.is_workday(cur):
                entries.append(HolidayEntry(
                    date=cur,
                    name="调休上班",
                    type="custom",
                    country_code="CN",
                    is_workday_swap=True,
                    source=SOURCE_CHINESE_CALENDAR,
                ))
            cur += timedelta(days=1)
    except NotImplementedError:
        return {"available": False, "entries": [], "error": f"chinesecalendar has no data for {year}"}
    except Exception as e:
        logger.warning(f"chinesecalendar fetch failed for {year}: {e}")
        return {"available": False, "entries": [], "error": str(e)}

    # Second pass: for consecutive days sharing the same name, keep the first
    # as the canonical main day and rename subsequent days to "{name}假期" —
    # matches the legacy hardcoded convention (spec §10 triggers only on the
    # first day's "春节"/"元旦" name match).
    entries.extend(_collapse_consecutive_holidays(raw_holidays))

    return {"available": True, "entries": entries, "error": None}


def _collapse_consecutive_holidays(
    raw: list[tuple[date, str]],
) -> list[HolidayEntry]:
    """Rename day 2+ of a consecutive same-name run to `{name}假期`."""
    out: list[HolidayEntry] = []
    prev_name: str | None = None
    prev_date: date | None = None
    for d, name in raw:
        is_continuation = (
            prev_name is not None
            and prev_date is not None
            and name == prev_name
            and (d - prev_date).days == 1
        )
        display_name = f"{name}假期" if is_continuation else name
        out.append(HolidayEntry(
            date=d,
            name=display_name,
            type="legal",
            country_code="CN",
            is_workday_swap=False,
            source=SOURCE_CHINESE_CALENDAR,
            metadata={"canonical_name": name} if is_continuation else None,
        ))
        prev_name = name
        prev_date = d
    return out


# chinesecalendar 返回的 holiday name 是英文/拼音 (e.g. "New Year's Day"),
# 我们映射到用户习惯的中文名。Key 按 case-insensitive 匹配（库在不同版本
# 大小写不一致，例如 "Tomb-sweeping Day" vs "Tomb-Sweeping Day"）。无匹
# 配时原样返回。
_NAME_MAP = {
    "new year's day": "元旦",
    "spring festival": "春节",
    "tomb-sweeping day": "清明节",
    "labour day": "劳动节",
    "labor day": "劳动节",
    "dragon boat festival": "端午节",
    "national day": "国庆节",
    "mid-autumn festival": "中秋节",
}


def _normalize_cn_name(raw: str) -> str:
    return _NAME_MAP.get(raw.lower(), raw)
