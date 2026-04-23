"""本地源: 算法生成的国际常见纪念日 + 可算的西方节日.

不依赖任何外部 API, 无网络也能用. 主要兜住 nager.date US 覆盖不到的:
  1. 固定日期国际纪念日 (妇女节 3-8, 儿童节 6-1, 国际劳动节 5-1)
     — 这些不是美国联邦假日, nager US 不返回
  2. 可算的西方节日 (母亲节 = 5 月第 2 个周日; 父亲节 = 6 月第 3 个周日)
     — nager 也有, 本地源作为备份
"""

from __future__ import annotations

from datetime import date, timedelta

from app.services.schedule_domain.holiday_repo import (
    SOURCE_LOCAL,
    HolidayEntry,
)

# 每年固定公历日期的国际纪念日
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


def compute_local(year: int) -> list[HolidayEntry]:
    """Return all algorithmically-computed international observances for the year."""
    entries: list[HolidayEntry] = []

    for month, day, name in _FIXED_DATES:
        entries.append(HolidayEntry(
            date=date(year, month, day),
            name=name,
            type="international",
            country_code="INTL",
            source=SOURCE_LOCAL,
        ))

    # 母亲节 = 5 月第 2 个周日 (周日 = weekday 6)
    entries.append(HolidayEntry(
        date=_nth_weekday(year, 5, 6, 2),
        name="母亲节",
        type="international",
        country_code="INTL",
        source=SOURCE_LOCAL,
    ))
    # 父亲节 = 6 月第 3 个周日
    entries.append(HolidayEntry(
        date=_nth_weekday(year, 6, 6, 3),
        name="父亲节",
        type="international",
        country_code="INTL",
        source=SOURCE_LOCAL,
    ))

    return entries
