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


def test_measure_word_does_not_block_parsing():
    """「两个月前」必须能解析.

    原来的模式是 数字+单位+前后, 中间不许有量词 —— 于是"两月前"能解析而**日常
    说法"两个月前"不能**。支持了没人用的写法, 漏了所有人用的写法。
    """
    now = _now()
    for text in ("两个月前", "2个月前", "半个月前", "三个星期前".replace("星期", "周")):
        assert parse_time_expressions(text, now=now), f"无解析: {text}"

    # 带不带量词应当落在同一天 (量词不改变语义)
    a = parse_time_expressions("两月前", now=now)[0]
    b = parse_time_expressions("两个月前", now=now)[0]
    assert a.start.date() == b.start.date()


def test_year_unit_is_supported():
    """「一年前」原本整个不解析 —— 年不在单位表里.

    它和"去年"不是一回事: 前者相对今天, 后者是自然年 (1/1-12/31)。
    """
    now = _now()
    for text in ("一年前", "1年前", "两年前", "半年前"):
        assert parse_time_expressions(text, now=now), f"无解析: {text}"

    one = parse_time_expressions("一年前", now=now)[0]
    assert 360 <= (now - one.start).days <= 370


def test_year_literal_is_not_read_as_an_offset():
    """「2024年前」是"2024 年之前", 不是"2024 年那么久以前".

    加「年」单位时踩过的坑: 不拦的话"2024年前我住在北京"会算出公元 3 年写进
    occur_time —— 一条彻底错误的日期, 而且静默。
    """
    now = _now()
    for text in ("2024年前", "2020年前的事", "1998年前"):
        assert not parse_time_expressions(text, now=now), f"误判成时间跨度: {text}"

    # 合理范围内的仍要能解析
    for text in ("十年前", "100年前", "3年前"):
        assert parse_time_expressions(text, now=now), f"误伤: {text}"


def test_half_maps_to_a_smaller_whole_unit():
    """半年 = 6 个月, 半个月 = 15 天 —— 比 0.5 年更准, 也不用把数字解析改成浮点."""
    now = _now()
    half_year = parse_time_expressions("半年前", now=now)[0]
    assert 170 <= (now - half_year.start).days <= 195

    half_month = parse_time_expressions("半个月前", now=now)[0]
    assert 12 <= (now - half_month.start).days <= 18


def test_coarse_units_get_a_tolerance_window():
    """月/年天生模糊, 说"两个月前"的人不是指那一天.

    窗口固定 30 天而不是"一个单位" —— 后者会让"一年前"得到整整 365 天的区间,
    等于把过去一年全捞出来, 那不是容错。
    """
    now = _now()
    for text in ("两个月前", "一年前", "半年前"):
        r = parse_time_expressions(text, now=now)[0]
        span = (r.end - r.start).days
        assert 25 <= span <= 35, f"{text} 窗口 {span} 天, 应在 30 天左右"

    # 天/周是精确表达, 不放宽
    for text in ("三天前", "两周前"):
        r = parse_time_expressions(text, now=now)[0]
        assert (r.end - r.start).days == 0, f"{text} 不该被放宽"


def test_widening_does_not_move_the_start():
    """放宽只动 end.

    记忆录入侧 (pipeline) 只取 .start 存 occur_time, 提醒也按 start 调度; 检索侧
    才用整个区间。动了 start 就会让 occur_time 记偏、提醒提前响。
    """
    now = _now()
    r = parse_time_expressions("两个月前", now=now)[0]
    assert 55 <= (now - r.start).days <= 65, "start 应当仍是那个时间点本身"


def test_quick_gate_matches_the_parser():
    """has_explicit_time 的快速闸门漏了的表达, 解析器根本不会被调用."""
    from app.services.schedule_domain.time_parser import has_explicit_time

    for text in ("两个月前", "半年前", "一年前", "半个月前"):
        assert has_explicit_time(text), f"闸门漏了: {text}"
        assert parse_time_expressions(text), f"闸门放行但解析不出: {text}"


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
