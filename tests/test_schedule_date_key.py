"""ai_daily_schedules 的日期键: 落库与回查必须落在同一行.

背景: 列是 `@db.Date`, 而驱动会把带时区的 datetime 先归一到 UTC 再截断. 本地
午夜 2026-07-29 00:00+08:00 变成 UTC 2026-07-28 16:00, 截断后是 07-28 —— 每天
的作息都写到前一天那行上, 撞唯一键把它覆盖掉.

这个错位藏了很久, 因为聊天热路径读的是 Redis (键按本地日期字符串拼), 而 DB 的
读侧当时也带着时区, 两边一样错正好抵消. 于是"只修一侧"反而会把兜底查询彻底打断.
下面两个测试就是钉住这一点.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parents[1]
_WRITE_SITE = _ROOT / "app" / "services" / "schedule_domain" / "schedule.py"
_READ_SITE = _ROOT / "app" / "api" / "public" / "conversations.py"


def _replace_calls_with_tzinfo_none(source: str) -> list[ast.Call]:
    """找出所有 `.replace(..., tzinfo=None)` 调用."""
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "replace"):
            continue
        for kw in node.keywords:
            if kw.arg == "tzinfo" and isinstance(kw.value, ast.Constant):
                if kw.value.value is None:
                    found.append(node)
    return found


def _midnight_kwargs(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


def test_both_sites_strip_tzinfo_when_building_the_date_key():
    """写侧和读侧都必须把 tzinfo 抹掉, 否则两者查/写的不是同一行."""
    for label, path in (("落库", _WRITE_SITE), ("回查", _READ_SITE)):
        calls = _replace_calls_with_tzinfo_none(path.read_text(encoding="utf-8"))
        midnight = [c for c in calls if {"hour", "minute", "second"} <= _midnight_kwargs(c)]
        assert midnight, (
            f"{label}处 ({path.name}) 找不到 `.replace(hour=0, ..., tzinfo=None)`。"
            "带时区的午夜会被归一到 UTC 后截断到前一天, 导致日期错位一天。"
        )


def test_tz_aware_midnight_would_land_on_the_previous_day():
    """记录这个坑本身: 带时区的本地午夜, 其 UTC 日期是前一天."""
    local_midnight = datetime(2026, 7, 29, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert local_midnight.astimezone(ZoneInfo("UTC")).date().isoformat() == "2026-07-28"

    naive_midnight = local_midnight.replace(tzinfo=None)
    assert naive_midnight.date().isoformat() == "2026-07-29"
