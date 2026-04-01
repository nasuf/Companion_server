"""时间基础服务。

PRD §9.2: 为所有模块提供统一的时间查询能力，包括当前时间、节假日、时区。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.config import settings
from app.data.holidays_cn import HOLIDAYS, WORKDAY_SWAPS

_TZ = ZoneInfo(settings.schedule_timezone)

_WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@dataclass(frozen=True)
class TimeInfo:
    now: datetime
    date: date
    weekday: str  # "星期一"
    is_weekend: bool


@dataclass(frozen=True)
class HolidayInfo:
    name: str
    date: date
    type: str  # legal/traditional/international
    days_away: int  # 距今天数，0=今天


def get_current_time() -> TimeInfo:
    """返回当前本地时间信息。"""
    now = datetime.now(_TZ)
    d = now.date()
    return TimeInfo(
        now=now,
        date=d,
        weekday=_WEEKDAY_CN[d.weekday()],
        is_weekend=d.weekday() >= 5,
    )


def is_holiday(d: date | None = None) -> HolidayInfo | None:
    """判断给定日期是否为节假日。"""
    d = d or datetime.now(_TZ).date()
    key = d.isoformat()
    entry = HOLIDAYS.get(key)
    if not entry:
        return None
    return HolidayInfo(
        name=entry["name"],
        date=d,
        type=entry["type"],
        days_away=0,
    )


def is_workday_swap(d: date | None = None) -> bool:
    """判断是否为调休上班日。"""
    d = d or datetime.now(_TZ).date()
    return d.isoformat() in WORKDAY_SWAPS


_next_holiday_cache: tuple[date, HolidayInfo | None] | None = None


def get_next_holiday(after: date | None = None, limit_days: int = 90) -> HolidayInfo | None:
    """返回未来最近的节假日（最多查90天）。结果按天缓存。"""
    global _next_holiday_cache
    start = after or datetime.now(_TZ).date()
    if _next_holiday_cache and _next_holiday_cache[0] == start:
        return _next_holiday_cache[1]

    result = None
    for i in range(1, limit_days + 1):
        d = start + timedelta(days=i)
        entry = HOLIDAYS.get(d.isoformat())
        if entry:
            result = HolidayInfo(name=entry["name"], date=d, type=entry["type"], days_away=i)
            break

    _next_holiday_cache = (start, result)
    return result


def build_time_context() -> str:
    """构建时间上下文文本，供prompt注入。

    包含：当前时间、今日节假日、即将到来的节假日。
    """
    ti = get_current_time()
    parts = [f"当前时间：{ti.now.strftime('%Y年%m月%d日 %H:%M')} {ti.weekday}"]

    today_holiday = is_holiday(ti.date)
    if today_holiday:
        parts.append(f"今天是{today_holiday.name}")

    if is_workday_swap(ti.date):
        parts.append("今天是调休上班日")

    if not today_holiday:
        next_h = get_next_holiday(ti.date)
        if next_h and next_h.days_away <= 7:
            if next_h.days_away == 1:
                parts.append(f"明天是{next_h.name}")
            else:
                parts.append(f"{next_h.days_away}天后是{next_h.name}")

    return "；".join(parts)
