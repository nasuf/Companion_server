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


# ── Part 5 §2.1: NTP 校准 ──
# 仅做校准结果记录, 不强行修改 OS 时钟 (容器环境通常无权限).
# 若漂移 > 阈值, 通过日志告警, 由运维介入.
_NTP_DRIFT_SECONDS: float = 0.0
_NTP_LAST_SYNC: datetime | None = None
_NTP_DRIFT_WARN_THRESHOLD = 1.0  # 1s


def get_ntp_drift() -> tuple[float, datetime | None]:
    """返回 (与 NTP 服务器的偏差秒数, 上次校准时间)."""
    return _NTP_DRIFT_SECONDS, _NTP_LAST_SYNC


def calibrate_against_ntp(server: str = "pool.ntp.org", timeout: float = 3.0) -> float | None:
    """同步与 NTP 服务器对比, 返回偏差秒数 (正=本地慢于 NTP).

    spec Part 5 §2.1: NTP 校准. 失败返回 None.
    """
    global _NTP_DRIFT_SECONDS, _NTP_LAST_SYNC
    try:
        import ntplib  # type: ignore[import-untyped]
        import time

        client = ntplib.NTPClient()
        response = client.request(server, version=3, timeout=timeout)
        ntp_time = response.tx_time
        local_time = time.time()
        drift = ntp_time - local_time
        _NTP_DRIFT_SECONDS = float(drift)
        _NTP_LAST_SYNC = datetime.now(_TZ)
        return drift
    except Exception:
        return None

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


def _lunar_holiday_today(d: date) -> tuple[str, str] | None:
    """Part 5 §2.1: 农历节日动态计算 (春节 = 正月初一; 中秋 = 八月十五).

    返回 (name, type) 或 None. lunardate 不可用时返回 None.
    """
    try:
        from lunardate import LunarDate
    except ImportError:
        return None
    try:
        lunar = LunarDate.fromSolarDate(d.year, d.month, d.day)
        # 春节: 正月初一
        if lunar.month == 1 and lunar.day == 1:
            return ("春节", "legal")
        # 中秋: 八月十五 (仅作为传统节日, spec §5.1 不主动触发)
        if lunar.month == 8 and lunar.day == 15:
            return ("中秋节", "legal")
        # 元宵: 正月十五
        if lunar.month == 1 and lunar.day == 15:
            return ("元宵节", "traditional")
        # 端午: 五月初五
        if lunar.month == 5 and lunar.day == 5:
            return ("端午节", "legal")
        # 七夕: 七月初七
        if lunar.month == 7 and lunar.day == 7:
            return ("七夕节", "traditional")
        # 重阳: 九月初九
        if lunar.month == 9 and lunar.day == 9:
            return ("重阳节", "traditional")
    except Exception:
        return None
    return None


def is_holiday(d: date | None = None) -> HolidayInfo | None:
    """判断给定日期是否为节假日。

    优先级:
    1. holidays_cn.py 静态表 (含调休、国际节日, 准确度更高)
    2. lunardate 动态计算 (覆盖静态表过期的年份)
    """
    d = d or datetime.now(_TZ).date()
    key = d.isoformat()
    entry = HOLIDAYS.get(key)
    if entry:
        return HolidayInfo(
            name=entry["name"],
            date=d,
            type=entry["type"],
            days_away=0,
        )

    # 静态表无命中 → 农历兜底
    lunar_hit = _lunar_holiday_today(d)
    if lunar_hit:
        name, htype = lunar_hit
        return HolidayInfo(name=name, date=d, type=htype, days_away=0)
    return None


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
