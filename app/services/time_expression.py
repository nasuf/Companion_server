"""时间表达式生成器。

PRD §9.4: 将时间点/段转化为符合AI人格的自然语言表达。
纯规则实现，不调用LLM。
"""

from __future__ import annotations

from datetime import datetime, date

from app.services.trait_model import get_dim
from app.services.time_service import _TZ

_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

_PERIOD_MAP = [
    (0, 6, "凌晨"),
    (6, 9, "早上"),
    (9, 12, "上午"),
    (12, 13, "中午"),
    (13, 18, "下午"),
    (18, 23, "晚上"),
    (23, 24, "深夜"),
]


def _get_period(hour: int) -> str:
    for start, end, name in _PERIOD_MAP:
        if start <= hour < end:
            return name
    return "深夜"


def format_time_naturally(
    target: datetime,
    now: datetime | None = None,
    seven_dim: dict | None = None,
) -> str:
    """将时间点转为人格化自然语言。"""
    now = now or datetime.now(_TZ)
    today = now.date()
    target_date = target.date() if isinstance(target, datetime) else target
    diff_days = (target_date - today).days

    is_lively = seven_dim and get_dim(seven_dim, "活泼度") >= 0.7
    is_precise = seven_dim and get_dim(seven_dim, "理性度") >= 0.7

    # 日期部分
    date_part = _format_date_part(target_date, today, diff_days, is_lively)

    # 时间部分（仅当target是datetime且非整天时）
    if isinstance(target, datetime) and target.hour != 0:
        time_part = _format_time_part(target.hour, target.minute, is_lively, is_precise)
        return f"{date_part}{time_part}"

    return date_part


def _format_date_part(target_date: date, today: date, diff: int, lively: bool) -> str:
    if diff == 0:
        return "今儿" if lively else "今天"
    elif diff == 1:
        return "明儿" if lively else "明天"
    elif diff == -1:
        return "昨儿" if lively else "昨天"
    elif diff == 2:
        return "后天"
    elif diff == -2:
        return "前天"
    elif diff == 3:
        return "大后天"
    elif diff == -3:
        return "大前天"

    # 本周内
    if -7 < diff < 7 and target_date.isocalendar()[1] == today.isocalendar()[1]:
        wd = _WEEKDAY_CN[target_date.weekday()]
        return f"这{wd}"

    # 上周/下周
    target_week = target_date.isocalendar()[1]
    today_week = today.isocalendar()[1]
    wd = _WEEKDAY_CN[target_date.weekday()]
    if target_week == today_week + 1 and target_date.year == today.year:
        return f"下{wd}"
    if target_week == today_week - 1 and target_date.year == today.year:
        return f"上{wd}"

    # 同年
    if target_date.year == today.year:
        return f"{target_date.month}月{target_date.day}日"

    # 去年
    if target_date.year == today.year - 1:
        return f"去年{target_date.month}月"

    return f"{target_date.year}年{target_date.month}月"


def _format_time_part(hour: int, minute: int, lively: bool, precise: bool) -> str:
    period = _get_period(hour)
    display_hour = hour % 12 or 12

    if lively:
        return f"{period}{display_hour}点那会儿"

    if minute:
        return f"{period}{display_hour}点{minute}分"
    return f"{period}{display_hour}点"
