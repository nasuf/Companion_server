"""occur_time 三层解析器 + 拓宽日期 pattern 的守卫。

背景: 2026-08 生产诊断发现事件类记忆 (生活/情绪) 里 user 侧只有 6-7% 填了
occur_time —— 抽取产出的是剥掉时间词的摘要, 规则引擎在 content 上抓不到东西。
杠杆是 statement_time 兜底 (event ≈ 说到它的时刻), 但结构化事件记忆 (送礼/红包/
时间胶囊) content 里埋着 ISO 日期, 旧 parser 认不出, 兜底会把事件日错标成说话日。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.schedule_domain.time_parser import (
    VAGUE_PAST_RE,
    has_explicit_time,
    parse_time_expressions,
    resolve_occur_time,
)

ST = datetime(2026, 8, 27, 10, 0, tzinfo=timezone.utc)  # 说话时刻


class TestWidenedDatePatterns:
    """拓宽后的 parser 要认出 ISO / 带年份 / 带空格 的日期 —— 旧 _DATE_PAT 只认
    紧凑的 '8月9日'。"""

    @pytest.mark.parametrize("text,expect", [
        ("2026-08-23 我给你送了礼物", (2026, 8, 23)),
        ("2026/8/9 的事", (2026, 8, 9)),
        ("用户于 2026 年 8 月 9 日埋下时间胶囊", (2026, 8, 9)),  # 带空格+年份
        ("2026年8月9日", (2026, 8, 9)),
    ])
    def test_iso_and_year_dates_parse(self, text, expect):
        assert has_explicit_time(text) is True
        parsed = [p for p in parse_time_expressions(text, now=ST) if p.type != "fuzzy"]
        assert parsed, f"no parse for {text!r}"
        best = max(parsed, key=lambda p: p.confidence)
        assert (best.start.year, best.start.month, best.start.day) == expect

    def test_spaced_plain_date_parses(self):
        # 旧 _DATE_PAT 要求 '8月9日' 无空格; 拓宽后 '8 月 9 日' 也认
        parsed = [p for p in parse_time_expressions("生日是 8 月 9 日", now=ST)]
        assert any((p.start.month, p.start.day) == (8, 9) for p in parsed)

    def test_year_date_not_double_added_with_wrong_year(self):
        # '2026年8月9日' 里的 '8月9日' 子串不能被 _DATE_PAT 用当前年重复加一条
        parsed = [p for p in parse_time_expressions("2026年8月9日", now=ST)
                  if p.type != "fuzzy"]
        years = {p.start.year for p in parsed}
        assert years == {2026}, f"unexpected years {years}"


class TestResolveOccurTime:
    """三层: 显性日期 > statement_time 兜底 > None(不猜)。"""

    def test_tier_a_embedded_iso_date_wins_over_statement(self):
        # 事件日埋在 content 里, 必须用它, 不能退到说话日
        r = resolve_occur_time(
            "2026-08-23 我给小芜送了礼物", statement_time=ST,
            main_category="生活", sub_category=None)
        assert r is not None and (r.year, r.month, r.day) == (2026, 8, 23)

    def test_tier_a_time_capsule_case(self):
        r = resolve_occur_time(
            "用户于 2026 年 8 月 9 日埋下了时间胶囊", statement_time=ST,
            main_category="生活")
        assert r is not None and (r.month, r.day) == (8, 9)

    def test_tier_b_statement_fallback_for_recent_event(self):
        # 没有显性日期的近期事件 → occur_time = 说话日
        r = resolve_occur_time("用户分手了", statement_time=ST, main_category="生活")
        assert r is not None and r.date() == ST.date()

    def test_tier_b_uses_now_when_statement_missing(self):
        r = resolve_occur_time("用户换了工作", statement_time=None, main_category="生活")
        assert r is not None  # 退到 _now_corrected(), 不返回 None

    def test_tier_a_anchors_calendar_date_at_utc_noon(self):
        # occur_time 列是 timestamp without tz, ORM 把 aware 转 UTC 落库。日历日期
        # 必须锚在 UTC 正午, 否则 +08 午夜转 UTC 会退到前一天 (生产实测踩过)。
        r = resolve_occur_time("2026-08-20 领了红包", statement_time=ST, main_category="生活")
        assert r is not None
        assert (r.year, r.month, r.day) == (2026, 8, 20)
        assert r.hour == 12 and r.utcoffset().total_seconds() == 0

    @pytest.mark.parametrize("mc", ["身份", "偏好", "思维"])
    def test_non_dateable_categories_return_none(self, mc):
        assert resolve_occur_time("用户喜欢吃酸菜鱼", statement_time=ST, main_category=mc) is None

    def test_reminder_subcategory_never_falls_back(self):
        # 提醒必须由上游给精确未来时间, 绝不能退到"现在"
        assert resolve_occur_time(
            "交季度报告", statement_time=ST,
            main_category="生活", sub_category="提醒") is None

    def test_profile_seed_returns_none(self):
        # 建号往事叙事: 事件 ≠ 建号时刻, 不猜
        assert resolve_occur_time(
            "18岁高考后去大理旅行", statement_time=ST,
            main_category="生活", provenance="profile_seed") is None

    @pytest.mark.parametrize("text", [
        "用户小时候在苏州长大", "以前经常去爬山", "读书时喜欢打篮球",
    ])
    def test_vague_past_returns_none(self, text):
        assert resolve_occur_time(text, statement_time=ST, main_category="生活") is None

    def test_vague_past_without_explicit_date_only(self):
        # 远过去词 + 显性日期时, 显性日期优先(Tier A 在远过去检查之前)
        r = resolve_occur_time(
            "小时候的事，不过具体是 2015-06-01", statement_time=ST, main_category="生活")
        assert r is not None and r.year == 2015

    @pytest.mark.parametrize("text", [
        "晚上和朋友视频聊周末约饭",   # "周末" 是提到的计划, 不是事件时间
        "回家后拼了一周前的拼图",     # "一周前" 描述拼图来历, 不是事件时间
        "提前准备好明天的工牌",       # "明天" 是对象, 事件是今天的准备动作
        "下午去超市采购",             # 纯时段, 不当日历日期
    ])
    def test_relative_words_fall_to_statement_not_tier_a(self, text):
        # 摘要里的相对词/时段不能被 Tier A 抓走 → occur_time = 说话日
        r = resolve_occur_time(text, statement_time=ST, main_category="生活")
        assert r is not None and r.date() == ST.date()


def test_vague_past_re_is_single_source_of_truth():
    """timeline 与写入侧兜底必须用同一份远过去判定, 不能各存一份。"""
    from app.services.memory.retrieval import timeline
    assert timeline._VAGUE_PAST_RE is VAGUE_PAST_RE
