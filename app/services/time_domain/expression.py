"""时间表达式生成器。

PRD §9.4: prompt 里的时刻与时段文字（format_clock / describe_time_scene）。
纯规则实现，不调用LLM。
"""

from __future__ import annotations

from datetime import datetime

from app.services.time_domain.clock import _TZ

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


def format_clock(moment: datetime) -> str:
    """prompt 里的"某个时刻": 09-29 周二 14:30 (UTC+8)。"""
    local = moment.astimezone(_TZ)
    return f"{local.strftime('%m-%d')} {_WEEKDAY_CN[local.weekday()]} {local.strftime('%H:%M')}"


def describe_time_scene(moment: datetime) -> str:
    """当前时间场景: 周二下午（工作日）/ 周六晚上（周末）/ 周四上午（节假日·国庆节）."""
    from app.services.time_domain.clock import classify_day_kind

    local = moment.astimezone(_TZ)
    weekday = _WEEKDAY_CN[local.weekday()]
    return f"{weekday}{_get_period(local.hour)}（{classify_day_kind(local.date())}）"
