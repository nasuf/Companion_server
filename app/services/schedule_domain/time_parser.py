"""时间解析器。

PRD §9.3.2: 识别用户消息中的显式时间表述，转换为标准时间范围。
规则引擎优先，覆盖常见中文时间表达。
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time

from app.data.holidays_cn import HOLIDAY_NAME_DATES
from app.services.schedule_domain.time_service import _TZ

_WEEKDAY_CN = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# Part 5 §3.1 相对时间段默认映射 (严格对齐 spec 表格).
# 深夜 = 次日 00:00-06:00, 在调用方决定是否要把日期 +1.
_PERIOD_HOURS = {
    "凌晨": (0, 6),
    "早上": (6, 9),
    "早晨": (6, 9),
    "上午": (9, 12),
    "中午": (12, 14),
    "下午": (14, 18),
    "傍晚": (17, 19),  # spec 未列, 按常识保留
    "晚上": (18, 24),
    "深夜": (0, 6),    # 实际语义指次日 0-6, 解析时由调用方决定 +1 天
}

# 标记需要"次日"语义的时段
_NEXT_DAY_PERIODS = {"深夜"}

# 相对日期词（按长度降序，确保"大前天"优先于"前天"匹配）
_RELATIVE_DAYS: list[tuple[str, int]] = sorted(
    [("今天", 0), ("明天", 1), ("后天", 2), ("大后天", 3),
     ("昨天", -1), ("前天", -2), ("大前天", -3)],
    key=lambda x: len(x[0]), reverse=True,
)

_FUZZY_WORDS: list[tuple[str, int, int]] = [
    ("小时候", -20 * 365, -5 * 365),
    ("以前", -365 * 3, 0),
    ("之前", -365, 0),
    ("当时", -365, 0),
]

# Pre-compiled patterns for hot path
_WEEK_PAT = re.compile(r"(上上?|下下?|这)(?:个)?周([一二三四五六日天])")
_DATE_PAT = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")
_YEAR_PAT = re.compile(r"(去年|前年|今年)(?:(\d{1,2})月)?")
_HOUR_PAT = re.compile(r"(?:(早上|上午|中午|下午|晚上|凌晨))?(\d{1,2})[点时](?:(\d{1,2})分?)?")
_QUICK_TIME_PAT = re.compile(
    r"[今昨明前后]天|[上下这]周|[上下这]个月"
    r"|\d{1,2}月\d{1,2}[日号]|\d{1,2}[点时]"
    r"|去年|前年|今年|大[前后]天"
    r"|早上|上午|中午|下午|晚上|凌晨"
    r"|小时候|以前"
)

_MONTH_MAP = [("这个月", 0), ("上个月", -1), ("下个月", 1)]

_PM_PERIODS = {"下午", "晚上"}


@dataclass
class ParsedTime:
    original_text: str
    type: str  # absolute / relative / fuzzy
    start: datetime
    end: datetime
    confidence: float
    is_future: bool


@dataclass
class TimeExtract:
    """Part 5 §3.1 双时间字段输出.

    spec §6.1 落库映射:
    - statement_time → memories.statement_time 列
    - event_times    → memories.occur_time 列 (取列表第一个或最相关的一条)

    statement_time: 用户说这句话的时间 (消息到达时刻)
    event_times:    事件时间范围列表 (可能多条: "昨天跟明天")
    """
    statement_time: datetime
    event_times: list[ParsedTime]


def parse_with_statement_time(
    message: str,
    now: datetime | None = None,
) -> TimeExtract:
    """spec Part 5 §3.1: 返回 (statement_time, event_time 列表).

    statement_time 取自调用时的 now (若未提供则用当前时间).
    调用方落库时, 把 event_times[0].start 写入 memories.occur_time,
    把 statement_time 写入 memories.statement_time.
    """
    ts = now or datetime.now(_TZ)
    return TimeExtract(
        statement_time=ts,
        event_times=parse_time_expressions(message, now=ts),
    )


def parse_time_expressions(
    message: str,
    now: datetime | None = None,
) -> list[ParsedTime]:
    """解析消息中的显式时间表述。返回所有匹配结果。"""
    now = now or datetime.now(_TZ)
    today = now.date()
    results: list[ParsedTime] = []
    used_spans: list[tuple[int, int]] = []

    def _add(text: str, start: datetime, end: datetime, typ: str, conf: float, span: tuple[int, int] | None = None) -> None:
        if span:
            for us, ue in used_spans:
                if span[0] < ue and span[1] > us:
                    return
            used_spans.append(span)
        results.append(ParsedTime(
            original_text=text,
            type=typ,
            start=start,
            end=end,
            confidence=conf,
            is_future=start > now,
        ))

    def _day_range(d: date) -> tuple[datetime, datetime]:
        return (
            datetime.combine(d, time.min, tzinfo=_TZ),
            datetime.combine(d, time.max, tzinfo=_TZ),
        )

    # --- 1. 相对日期（长词优先，避免"大前天"被"前天"抢先匹配）---
    for word, delta in _RELATIVE_DAYS:
        idx = message.find(word)
        if idx != -1:
            span = (idx, idx + len(word))
            d = today + timedelta(days=delta)
            s, e = _day_range(d)
            _add(word, s, e, "relative", 0.95, span)

    # --- 2. 相对周 ---
    for m in _WEEK_PAT.finditer(message):
        prefix, wd = m.group(1), m.group(2)
        target_wd = _WEEKDAY_CN.get(wd)
        if target_wd is None:
            continue
        current_wd = today.weekday()
        diff = target_wd - current_wd
        if prefix == "这":
            pass
        elif prefix == "下":
            diff += 7
        elif prefix == "下下":
            diff += 14
        elif prefix == "上":
            diff -= 7
        elif prefix == "上上":
            diff -= 14
        d = today + timedelta(days=diff)
        s, e = _day_range(d)
        _add(m.group(), s, e, "relative", 0.9, m.span())

    # --- 3. 相对月 ---
    for word, delta in _MONTH_MAP:
        idx = message.find(word)
        if idx != -1:
            span = (idx, idx + len(word))
            year = today.year
            month = today.month + delta
            if month < 1:
                month += 12
                year -= 1
            elif month > 12:
                month -= 12
                year += 1
            _, last_day = monthrange(year, month)
            s = datetime(year, month, 1, tzinfo=_TZ)
            e = datetime(year, month, last_day, 23, 59, 59, tzinfo=_TZ)
            _add(word, s, e, "relative", 0.85, span)

    # --- 4. X月X日/号 ---
    for m in _DATE_PAT.finditer(message):
        month, day = int(m.group(1)), int(m.group(2))
        year = today.year
        try:
            d = date(year, month, day)
        except ValueError:
            continue
        if d < today and (today - d).days > 180:
            d = date(year + 1, month, day)
        s, e = _day_range(d)
        _add(m.group(), s, e, "absolute", 0.9, m.span())

    # --- 5. 去年/前年/今年 + 可选月份 ---
    for m in _YEAR_PAT.finditer(message):
        prefix = m.group(1)
        month_str = m.group(2)
        year = today.year
        if prefix == "去年":
            year -= 1
        elif prefix == "前年":
            year -= 2
        if month_str:
            month = int(month_str)
            _, last_day = monthrange(year, month)
            s = datetime(year, month, 1, tzinfo=_TZ)
            e = datetime(year, month, last_day, 23, 59, 59, tzinfo=_TZ)
        else:
            s = datetime(year, 1, 1, tzinfo=_TZ)
            e = datetime(year, 12, 31, 23, 59, 59, tzinfo=_TZ)
        _add(m.group(), s, e, "absolute", 0.85, m.span())

    # --- 6. 时间点 X点/时 ---
    for m in _HOUR_PAT.finditer(message):
        period = m.group(1)
        hour = int(m.group(2))
        minute = int(m.group(3)) if m.group(3) else 0
        if period and hour <= 12 and period in _PM_PERIODS and hour < 12:
            hour += 12
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            dt = datetime.combine(today, time(hour, minute), tzinfo=_TZ)
            _add(m.group(), dt, dt + timedelta(hours=1), "absolute", 0.8, m.span())

    # --- 7. 时间段词 ---
    for period_name, (h_start, h_end) in _PERIOD_HOURS.items():
        idx = message.find(period_name)
        if idx == -1:
            continue
        span = (idx, idx + len(period_name))
        # spec §3.1: "深夜" 语义 = 次日 00:00-06:00
        anchor_day = today + timedelta(days=1) if period_name in _NEXT_DAY_PERIODS else today
        # 处理 18-24 这种 end=24 不是合法 hour, 改用次日 00:00 作为闭区间
        if h_end >= 24:
            s = datetime.combine(anchor_day, time(h_start, 0), tzinfo=_TZ)
            e = datetime.combine(anchor_day + timedelta(days=1), time(0, 0), tzinfo=_TZ) - timedelta(seconds=1)
        else:
            s = datetime.combine(anchor_day, time(h_start, 0), tzinfo=_TZ)
            e = datetime.combine(anchor_day, time(h_end - 1, 59, 59), tzinfo=_TZ)
        _add(period_name, s, e, "relative", 0.6, span)

    # --- 8. 节日名称 ---
    for holiday_name, dates in HOLIDAY_NAME_DATES.items():
        idx = message.find(holiday_name)
        if idx == -1:
            continue
        span = (idx, idx + len(holiday_name))
        best_date_str = _nearest_holiday_date(dates, today)
        if best_date_str:
            d = date.fromisoformat(best_date_str)
            s, e = _day_range(d)
            prefix_text = message[max(0, idx - 3):idx]
            if "去年" in prefix_text:
                d_prev = date(d.year - 1, d.month, d.day)
                s, e = _day_range(d_prev)
            _add(holiday_name, s, e, "absolute", 0.85, span)

    # --- 9. 模糊时间 ---
    for word, d_start, d_end in _FUZZY_WORDS:
        idx = message.find(word)
        if idx != -1:
            span = (idx, idx + len(word))
            s = now + timedelta(days=d_start)
            e = now + timedelta(days=d_end)
            _add(word, s, e, "fuzzy", 0.5, span)

    return results


def _nearest_holiday_date(dates: list[str], today: date) -> str | None:
    """从候选日期中找距今最近的一个（优先当年或最近过去年份）。"""
    best, best_diff = None, float("inf")
    for ds in dates:
        d = date.fromisoformat(ds)
        diff = abs((d - today).days)
        if diff < best_diff:
            best, best_diff = ds, diff
    return best


def has_explicit_time(message: str) -> bool:
    """快速判断消息是否包含显式时间表述（无需完整解析）。"""
    return bool(_QUICK_TIME_PAT.search(message))
