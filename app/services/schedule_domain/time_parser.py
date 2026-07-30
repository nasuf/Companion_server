"""时间解析器。

Spec Part 5 §3.1: 识别用户消息中的显式时间表述，转换为标准时间范围。
纯规则引擎，不调大模型。

spec 明确模糊时间词（"小时候/几年前/以前/当时"）不做处理 — 无法
落成确定的时间范围，勉强解析成 20 年跨度会污染 occur_time 过滤。
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time

from app.services.schedule_domain import holiday_cache
from app.services.schedule_domain.time_service import _TZ, _now_corrected

_WEEKDAY_CN = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# Part 5 §3.1 相对时间段默认映射 (严格对齐 spec 表格 7 项).
# 深夜 = 次日 00:00-06:00, 在调用方决定是否要把日期 +1.
_PERIOD_HOURS = {
    "凌晨": (0, 6),
    "早上": (6, 9),
    "早晨": (6, 9),
    "上午": (9, 12),
    "中午": (12, 14),
    "下午": (14, 18),
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

# Pre-compiled patterns for hot path
_WEEK_PAT = re.compile(r"(上上?|下下?|这|本)(?:个)?周([一二三四五六日天])")
_WEEK_RANGE_PAT = re.compile(r"(上上|下下|上|下|这|本)(?:个)?周(?![一二三四五六日天末])")
_WEEKEND_PAT = re.compile(r"(上上|下下|上|下|这|本)?(?:个)?周末")
_DATE_PAT = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")
_YEAR_PAT = re.compile(r"(去年|前年|今年)(?:(\d{1,2})月)?")
_HOUR_PAT = re.compile(r"(?:(早上|上午|中午|下午|晚上|凌晨))?(\d{1,2})[点时](?:(\d{1,2})分?)?")
# 相对偏移: "3天后 / 2小时前 / 15分钟之后 / 两个月前 / 半年前 / 十五天前"
#
# 「个」必须可选: 原来的模式是 数字+单位+前后, 于是"两月前"能解析而**"两个月前"
# 不能** —— 恰好把没人这么说的写法支持了, 把日常说法漏了。同理"半个月前"。
# 「年」原本整个缺失, 而"一年前"和"去年"是两回事 (前者是相对今天, 后者是自然年)。
_REL_OFFSET_PAT = re.compile(
    r"([一二三四五六七八九十百两半\d]{1,4})\s*个?\s*(分钟|小时|天|周|月|年)(?:之)?(前|后)"
)
_QUICK_TIME_PAT = re.compile(
    r"[今昨明前后]天|[上下这本]周|周末|[上下这]个月"
    r"|\d{1,2}月\d{1,2}[日号]|\d{1,2}[点时]"
    r"|去年|前年|今年|大[前后]天"
    r"|早上|上午|中午|下午|晚上|凌晨"
    # 必须与 _REL_OFFSET_PAT 保持同步 —— 这是 has_explicit_time 的快速闸门,
    # 它漏了的表达连解析都不会被调用。
    r"|[一二三四五六七八九十百两半\d]{1,4}\s*个?\s*(?:分钟|小时|天|周|月|年)(?:之)?[前后]"
)

_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}


def _parse_cn_number(s: str) -> int | None:
    """解析中文或阿拉伯数字（0-99 足够覆盖常见相对时间表达）。"""
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    # X十
    if len(s) == 2 and s[1] == "十" and s[0] in _CN_DIGIT:
        return _CN_DIGIT[s[0]] * 10
    # 十X
    if len(s) == 2 and s[0] == "十" and s[1] in _CN_DIGIT:
        return 10 + _CN_DIGIT[s[1]]
    # X十Y
    if (len(s) == 3 and s[1] == "十"
        and s[0] in _CN_DIGIT and s[2] in _CN_DIGIT):
        return _CN_DIGIT[s[0]] * 10 + _CN_DIGIT[s[2]]
    # 单字
    if len(s) == 1 and s in _CN_DIGIT:
        return _CN_DIGIT[s]
    return None

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

    statement_time 取自调用时的 now (若未提供则用 NTP 修正后当前时间, 跟
    pipeline.py 提醒未来校验比较侧一致).
    调用方落库时, 把 event_times[0].start 写入 memories.occur_time,
    把 statement_time 写入 memories.statement_time.
    """
    ts = now or _now_corrected()
    return TimeExtract(
        statement_time=ts,
        event_times=parse_time_expressions(message, now=ts),
    )


def parse_time_expressions(
    message: str,
    now: datetime | None = None,
) -> list[ParsedTime]:
    """解析消息中的显式时间表述。返回所有匹配结果。"""
    now = now or _now_corrected()
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

    # --- 1b. N 天后 / N 小时前 / 十五分钟后 等相对偏移 ---
    for m in _REL_OFFSET_PAT.finditer(message):
        num_str, unit, direction = m.group(1), m.group(2), m.group(3)
        if num_str == "半":
            # 换成更小单位的整数而不是让 _parse_cn_number 返回小数 —— 改它的返回
            # 类型会波及所有调用方, 而"半年=6个月"本身就比"0.5 年"更准确。
            half = {"年": (6, "月"), "月": (15, "天"), "天": (12, "小时"),
                    "小时": (30, "分钟")}.get(unit)
            if half is None:
                continue
            amount, unit = half
        else:
            amount = _parse_cn_number(num_str)
        if amount is None:
            continue
        if direction == "前":
            amount = -amount
        target = now
        if unit == "分钟":
            target = now + timedelta(minutes=amount)
            s, e = target, target + timedelta(minutes=1)
        elif unit == "小时":
            target = now + timedelta(hours=amount)
            s, e = target, target + timedelta(hours=1)
        elif unit == "天":
            d = today + timedelta(days=amount)
            s, e = _day_range(d)
        elif unit == "周":
            d = today + timedelta(weeks=amount)
            s, e = _day_range(d)
        else:
            # 月/年按近似天数折算。这两个单位天生是模糊的 —— 说"两个月前"的人不是
            # 指那一天, 所以 end 往后放宽一个单位, 检索时才圈得住。
            #
            # 只放宽 end 不动 start 是刻意的: 记忆录入侧 (pipeline) 只取 .start 存
            # occur_time, 检索侧 (hybrid) 才用整个区间。这样加宽只影响检索, 不会让
            # 提醒提前触发, 也不会把 occur_time 记偏。
            #
            # 放宽幅度固定 30 天而不是"一个单位": 按单位放宽的话"一年前"会得到
            # 整整 365 天的窗口 (一年前一直到今天), 那不是容错是把整段时间全捞。
            # 说"一年前"的人指的是那个时间点前后, 一个月的容差足够。
            step = 30 if unit == "月" else 365
            d = today + timedelta(days=step * amount)
            s, _ = _day_range(d)
            _, e = _day_range(d + timedelta(days=30))
        _add(m.group(), s, e, "relative", 0.88, m.span())

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

    # --- 2b. 整周 / 周末范围 ---
    def _week_offset(prefix: str | None) -> int:
        if prefix == "上上":
            return -2
        if prefix == "上":
            return -1
        if prefix == "下":
            return 1
        if prefix == "下下":
            return 2
        return 0

    monday = today - timedelta(days=today.weekday())
    for m in _WEEK_RANGE_PAT.finditer(message):
        prefix = m.group(1)
        start_day = monday + timedelta(weeks=_week_offset(prefix))
        end_day = start_day + timedelta(days=6)
        s = datetime.combine(start_day, time.min, tzinfo=_TZ)
        e = datetime.combine(end_day, time.max, tzinfo=_TZ)
        _add(m.group(), s, e, "relative", 0.82, m.span())

    for m in _WEEKEND_PAT.finditer(message):
        prefix = m.group(1)
        start_day = monday + timedelta(weeks=_week_offset(prefix), days=5)
        end_day = start_day + timedelta(days=1)
        s = datetime.combine(start_day, time.min, tzinfo=_TZ)
        e = datetime.combine(end_day, time.max, tzinfo=_TZ)
        _add(m.group(), s, e, "relative", 0.82, m.span())

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
        # spec §3.1 "晚上 12 点" 语义对齐"深夜次日 00:00", 而不是中午
        if period == "晚上" and hour == 12:
            anchor_day = today + timedelta(days=1)
            dt = datetime.combine(anchor_day, time(0, minute), tzinfo=_TZ)
            _add(m.group(), dt, dt + timedelta(hours=1), "absolute", 0.85, m.span())
            continue
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
    for holiday_name in holiday_cache.all_known_names():
        idx = message.find(holiday_name)
        if idx == -1:
            continue
        span = (idx, idx + len(holiday_name))
        dates = holiday_cache.list_dates_for_name(holiday_name)
        best = _nearest_holiday_date(dates, today)
        if best:
            d_found = best
            s, e = _day_range(d_found)
            prefix_text = message[max(0, idx - 3):idx]
            if "去年" in prefix_text:
                d_prev = date(d_found.year - 1, d_found.month, d_found.day)
                s, e = _day_range(d_prev)
            _add(holiday_name, s, e, "absolute", 0.85, span)

    # spec §3.1 明确不处理模糊时间词（小时候 / 以前 / 之前 / 当时）。

    return results


def _nearest_holiday_date(dates: list[date], today: date) -> date | None:
    """从候选日期中找距今最近的一个（优先当年或最近过去年份）。"""
    best: date | None = None
    best_diff = float("inf")
    for d in dates:
        diff = abs((d - today).days)
        if diff < best_diff:
            best, best_diff = d, diff
    return best


def has_explicit_time(message: str) -> bool:
    """快速判断消息是否包含显式时间表述（无需完整解析）。"""
    return bool(_QUICK_TIME_PAT.search(message))


# RECORD_REQUEST 上下文专用宽松正则: 允许 "X 分钟/小时/天/周" 省略 "前/后" 字.
# 严格 _REL_OFFSET_PAT 要求 "前|后" 防 "我等了一分钟" 误匹配, 但 RECORD_REQUEST
# intent 已确认是用户设提醒, 大概率 "X 分钟" = "X 分钟后". 此 helper 仅在调用方
# 显式表态"上下文是 reminder 设置"时使用.
_LOOSE_OFFSET_PAT = re.compile(
    r"([一二三四五六七八九十百两\d]{1,4})\s*(秒|分钟|小时|天|周)"
)


def parse_loose_offset(
    message: str, now: datetime, *, default_direction: str = "后",
) -> datetime | None:
    """RECORD_REQUEST 等"已知用户在设时间"的上下文里, 宽松提取 "X 时长" 当作
    `+ default_direction` (通常为 "后"). 返回 future datetime 或 None.

    跟 `parse_time_expressions` 的区别:
    - 严格 parser 要求"前/后" 后缀, 防止"我等了一分钟" 这种非时间表达被误匹配
    - 本 helper **不要求**后缀, 假设调用方上下文已确认意图

    支持中文数字 (一/两/十/十五/二十) + 阿拉伯数字 (1/30/180).
    """
    if default_direction not in ("前", "后"):
        return None
    m = _LOOSE_OFFSET_PAT.search(message)
    if not m:
        return None
    amount = _parse_cn_number(m.group(1))
    if amount is None or amount <= 0:
        return None
    unit = m.group(2)
    sign = -1 if default_direction == "前" else 1
    if unit == "秒":
        delta = timedelta(seconds=sign * amount)
    elif unit == "分钟":
        delta = timedelta(minutes=sign * amount)
    elif unit == "小时":
        delta = timedelta(hours=sign * amount)
    elif unit == "天":
        delta = timedelta(days=sign * amount)
    elif unit == "周":
        delta = timedelta(weeks=sign * amount)
    else:
        return None
    return now + delta
