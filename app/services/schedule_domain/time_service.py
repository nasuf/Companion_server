"""时间基础服务。

PRD §9.2: 为所有模块提供统一的时间查询能力，包括当前时间、节假日、时区。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.schedule_domain import holiday_cache

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
    timestamp_ms: int  # spec §2.2: Unix 毫秒时间戳


@dataclass(frozen=True)
class HolidayInfo:
    name: str
    date: date
    type: str  # legal/traditional/international
    days_away: int  # 距今天数，0=今天


def _now_corrected() -> datetime:
    """spec §2.1: 应用 NTP drift 后的当前北京时间, 用于关键比较点.

    日级别比较 (date(), 节假日扫描) 不需要走此修正 — drift 是亚秒级.
    时间中枢 + 提醒 occur_time 比较等亚秒精度敏感场景应走此函数; 散落各模块
    的 datetime.now() 直接调用不修正, 见 CLAUDE.md §6 偏离表.
    """
    return datetime.now(_TZ) + timedelta(seconds=_NTP_DRIFT_SECONDS)


def ensure_aware(dt: datetime | None) -> datetime | None:
    """统一成 tz-aware (假定 naive 是 UTC).

    防 'can't compare offset-naive and offset-aware datetimes' — 多个数据
    源 (Prisma datetime / Redis ISO / LLM 输出 ISO / datetime.fromisoformat)
    返回 naive 还是 aware 不一致, 比较点散落各处. 在边界统一规范化, 比每个
    比较点临时打补丁干净.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_current_time() -> TimeInfo:
    """spec §2.2 getCurrentDateTime(): 返回当前本地时间信息 (含 NTP 修正)."""
    now = _now_corrected()
    d = now.date()
    return TimeInfo(
        now=now,
        date=d,
        weekday=_WEEKDAY_CN[d.weekday()],
        is_weekend=d.weekday() >= 5,
        timestamp_ms=int(now.timestamp() * 1000),
    )


def get_current_timestamp_ms() -> int:
    """spec §2.2 getCurrentTimestamp(): Unix 毫秒时间戳 (含 NTP 修正)."""
    return int(_now_corrected().timestamp() * 1000)


async def resolve_implicit_time(
    agent_id: str,
    ai_status: dict | None = None,
) -> tuple[datetime, str]:
    """spec §3.2 隐性时间解析: 给"询问当前状态"意图返 (current_time, current_activity_str).

    `ai_status` 已加载时复用 (caller 在 data_fetch_phase 已 fetch), 否则现场加载.
    替代散落 chat 内的 `format_schedule_context(get_current_status(schedule))` inline 写法.
    """
    from app.services.schedule_domain.schedule import (
        get_cached_schedule, get_current_status, format_schedule_context,
    )
    if ai_status is None:
        schedule = await get_cached_schedule(agent_id)
        ai_status = get_current_status(schedule) if schedule else None
    activity = format_schedule_context(ai_status) if ai_status else "(未知)"
    return _now_corrected(), activity


def _lunar_holiday_today(d: date) -> tuple[str, str] | None:
    """Part 5 §2.1: 农历节日动态计算.

    返回 (name, type) 或 None. lunardate 不可用时返回 None.

    注: type 字段仅供 prompt 节日上下文展示, **主动触发判定走 spec §5.1
    name 白名单 (special_dates.py: ["春节","元旦"])**, 不依赖 type. 此处把
    中秋/端午标 type='legal' 是为对齐"国务院法定节日"语义, 不代表会主动触发.
    覆盖度: 春节 / 元宵 / 端午 / 七夕 / 中秋 / 重阳 (6 个), 除夕 / 腊八 /
    小年 等依赖 holidays_cn.py 静态 DB (覆盖到 2027). 见 CLAUDE.md §6.
    """
    try:
        from lunardate import LunarDate
    except ImportError:
        return None
    try:
        lunar = LunarDate.fromSolarDate(d.year, d.month, d.day)
        if lunar.month == 1 and lunar.day == 1:
            return ("春节", "legal")
        if lunar.month == 8 and lunar.day == 15:
            return ("中秋节", "legal")
        if lunar.month == 1 and lunar.day == 15:
            return ("元宵节", "traditional")
        if lunar.month == 5 and lunar.day == 5:
            return ("端午节", "legal")
        if lunar.month == 7 and lunar.day == 7:
            return ("七夕节", "traditional")
        if lunar.month == 9 and lunar.day == 9:
            return ("重阳节", "traditional")
    except Exception:
        return None
    return None


def is_holiday(d: date | None = None) -> HolidayInfo | None:
    """判断给定日期是否为节假日。

    优先级:
    1. DB (holiday_cache 模块内进程缓存, 启动时预加载, admin 保存后失效)
    2. lunardate 动态计算 (覆盖 DB 没覆盖的未来年份)
    """
    d = d or datetime.now(_TZ).date()
    cached = holiday_cache.get_by_date(d)
    if cached:
        return HolidayInfo(
            name=cached.name,
            date=d,
            type=cached.type,
            days_away=0,
        )

    lunar_hit = _lunar_holiday_today(d)
    if lunar_hit:
        name, htype = lunar_hit
        return HolidayInfo(name=name, date=d, type=htype, days_away=0)
    return None


def is_workday_swap(d: date | None = None) -> bool:
    """判断是否为调休上班日。"""
    d = d or datetime.now(_TZ).date()
    return holiday_cache.is_workday_swap(d)


def classify_day_kind(d: date, holiday_info: HolidayInfo | None = None) -> str:
    """Spec Part 5 §3.2: 当日属性分类 — 节假日·X / 调休上班日 / 周末 / 工作日.

    holiday_info 省略时自查一次; 调用方若已持有节假日对象可直接传入
    (作息生成热路径用这条避免重复 DB/cache 查询).
    """
    info = holiday_info if holiday_info is not None else is_holiday(d)
    if info is not None:
        return f"节假日·{info.name}"
    if is_workday_swap(d):
        return "调休上班日"
    if d.weekday() >= 5:
        return "周末"
    return "工作日"


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
        entry = holiday_cache.get_by_date(d)
        if entry:
            result = HolidayInfo(name=entry.name, date=d, type=entry.type, days_away=i)
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
