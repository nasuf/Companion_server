"""Algorithmic international observances that no public API reliably returns.

Covers two classes:
  1. Fixed-date UN-observed international days (妇女节 3-8, 儿童节 6-1,
     国际劳动妇女节 3-8 等)
  2. Movable Western holidays computed from day-of-month rules (母亲节 =
     5 月第 2 个周日; 父亲节 = 6 月第 3 个周日; 感恩节 = 11 月第 4 个
     周四 — 这些也由 nager 提供, 但算法版用作无网络时的兜底)

We intentionally keep overlap with nager minimal by only emitting names
nager does not return for US (妇女节, 儿童节, 重阳节等 UN-style observances).
"""

from __future__ import annotations

from datetime import date, timedelta

from app.services.schedule_domain.holiday_repo import (
    SOURCE_UN_OBSERVED,
    HolidayEntry,
)

# spec: 每年固定公历日期的国际纪念日
_FIXED_DATES = [
    (3, 8, "妇女节"),
    (6, 1, "儿童节"),
    (5, 1, "国际劳动节"),  # CN 也是劳动节, 但这里标注 INTL
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the `n`-th occurrence of `weekday` (Monday=0) in `year-month`."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def compute_un_observed(year: int) -> list[HolidayEntry]:
    """Return all algorithmically-computed international observances for the year."""
    entries: list[HolidayEntry] = []

    for month, day, name in _FIXED_DATES:
        entries.append(HolidayEntry(
            date=date(year, month, day),
            name=name,
            type="international",
            country_code="INTL",
            source=SOURCE_UN_OBSERVED,
        ))

    # 母亲节 = 5 月第 2 个周日 (周日 = weekday 6)
    entries.append(HolidayEntry(
        date=_nth_weekday(year, 5, 6, 2),
        name="母亲节",
        type="international",
        country_code="INTL",
        source=SOURCE_UN_OBSERVED,
    ))
    # 父亲节 = 6 月第 3 个周日
    entries.append(HolidayEntry(
        date=_nth_weekday(year, 6, 6, 3),
        name="父亲节",
        type="international",
        country_code="INTL",
        source=SOURCE_UN_OBSERVED,
    ))

    return entries
