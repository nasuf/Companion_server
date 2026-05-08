"""Spec Part 5 §3.1 显性时间解析器单测.

覆盖: 时段映射 / 相对偏移 / 周/月/年 / 节日名 / 双字段输出 / 模糊词 negative.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.services.schedule_domain.time_parser import (
    parse_time_expressions,
    parse_with_statement_time,
    _PERIOD_HOURS,
)
from app.services.schedule_domain.time_service import _TZ


def _now() -> datetime:
    """固定一个非边界时刻 (周三 14:30) 让相对计算稳定."""
    return datetime(2026, 4, 22, 14, 30, tzinfo=_TZ)


# ── P0-1 回归: 删除"傍晚" ───────────────────────────────────────────


def test_period_no_dusk():
    """spec §3.1 表只列 7 个时段词, "傍晚"不在其中."""
    assert "傍晚" not in _PERIOD_HOURS


def test_all_periods_match_spec_table():
    """7 个时段词全在表里, 范围对齐 spec."""
    expected = {
        "凌晨": (0, 6),
        "早上": (6, 9),
        "早晨": (6, 9),
        "上午": (9, 12),
        "中午": (12, 14),
        "下午": (14, 18),
        "晚上": (18, 24),
        "深夜": (0, 6),
    }
    assert _PERIOD_HOURS == expected


# ── P1-3: "晚上 12 点" 边界 ─────────────────────────────────────────


def test_evening_12_maps_to_next_day_midnight():
    """spec §3.1: "晚上 12 点" 应 == 深夜次日 00:00, 不能跟"中午 12 点"撞."""
    results = parse_time_expressions("晚上12点开会", now=_now())
    starts = [r.start for r in results if r.start.hour == 0]
    assert starts, f"没找到 00:00 的解析: {[(r.original_text, r.start) for r in results]}"
    next_day = (_now().date() + timedelta(days=1))
    assert any(s.date() == next_day and s.hour == 0 for s in starts)


# ── 相对偏移 ───────────────────────────────────────────────────────


def test_relative_offset_cn_digit():
    """中文数字"三天后/十五分钟前/两周后"."""
    now = _now()
    cases = [
        ("三天后", "天", 3),
        ("十五分钟前", "分钟", -15),
        ("两周后", "周", 2),
    ]
    for text, _unit, expected_amount in cases:
        results = parse_time_expressions(text, now=now)
        assert results, f"无解析: {text}"
        r = results[0]
        diff_seconds = (r.start - now).total_seconds()
        if _unit == "天":
            # 天级应该 ~= expected_amount 天 (取当天 00:00 所以可能略小于 24h)
            assert abs(diff_seconds / 86400 - expected_amount) < 1.5
        elif _unit == "分钟":
            assert abs(diff_seconds / 60 - expected_amount) < 2
        elif _unit == "周":
            assert abs(diff_seconds / 86400 - expected_amount * 7) < 1.5


def test_relative_weekday_combinations():
    """下下周三 / 上上周一 / 这周五."""
    now = _now()  # 周三
    today = now.date()
    cases = [
        # 今天周三 (4/22): 下下周三 = +14 = 5/6
        ("下下周三", today + timedelta(days=14)),
        # 上上周一 = 上上周的周一 = 比"这周一" (4/20) 再往前 14 天 = 4/6
        ("上上周一", today + timedelta(days=-16)),
        # 这周五 = 周三 + 2 = 4/24
        ("这周五", today + timedelta(days=2)),
        # 本周五 同义于这周五
        ("本周五", today + timedelta(days=2)),
    ]
    for text, expected_date in cases:
        results = parse_time_expressions(text, now=now)
        assert results, f"无解析: {text}"
        assert results[0].start.date() == expected_date, (
            f"{text} expected {expected_date}, got {results[0].start.date()}"
        )


def test_relative_week_and_weekend_ranges():
    """下周 / 周末 这类范围词必须解析成日期范围，供日程查询复用。"""
    now = _now()  # 周三
    results = parse_time_expressions("下周忙吗", now=now)
    assert results, "无解析: 下周"
    assert results[0].start.date().isoformat() == "2026-04-27"
    assert results[0].end.date().isoformat() == "2026-05-03"

    weekend = parse_time_expressions("周末有空吗", now=now)
    assert weekend, "无解析: 周末"
    assert weekend[0].start.date().isoformat() == "2026-04-25"
    assert weekend[0].end.date().isoformat() == "2026-04-26"


def test_absolute_year_with_month():
    """去年3月 / 今年12月 / 前年."""
    now = _now()  # 2026 年
    cases = [
        ("去年3月", 2025, 3),
        ("今年12月", 2026, 12),
        ("前年", 2024, None),
    ]
    for text, expected_year, expected_month in cases:
        results = parse_time_expressions(text, now=now)
        assert results, f"无解析: {text}"
        r = results[0]
        assert r.start.year == expected_year, f"{text} year"
        if expected_month is not None:
            assert r.start.month == expected_month, f"{text} month"


# ── 模糊词 negative ───────────────────────────────────────────────


def test_fuzzy_words_not_parsed():
    """spec §3.1 明确不解析: 小时候 / 几年前 / 以前 / 当时."""
    fuzzy_msgs = ["小时候我喜欢", "几年前去过", "以前住北京", "当时还小"]
    for msg in fuzzy_msgs:
        results = parse_time_expressions(msg, now=_now())
        assert results == [], f"模糊词被解析了: {msg} → {results}"


# ── 双字段输出 ────────────────────────────────────────────────────


def test_double_field_output():
    """spec §3.1: parse_with_statement_time 返 (statement_time, event_times)."""
    now = _now()
    yesterday = now.date() - timedelta(days=1)
    extract = parse_with_statement_time("我昨天去了医院", now=now)
    assert extract.statement_time == now
    assert len(extract.event_times) == 1
    assert extract.event_times[0].start.date() == yesterday
    # spec: 未指定具体时间点 → event_time 全天
    assert extract.event_times[0].start.hour == 0
    assert extract.event_times[0].end.hour == 23


def test_multi_event_times():
    """昨天跟明天都有事 → 2 个 event_times, 顺序按 message 出现."""
    now = _now()
    extract = parse_with_statement_time("昨天跟明天都有事", now=now)
    days = sorted({r.start.date() for r in extract.event_times})
    assert (now.date() - timedelta(days=1)) in days
    assert (now.date() + timedelta(days=1)) in days
