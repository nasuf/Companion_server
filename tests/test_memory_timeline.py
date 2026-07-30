"""聚合类时间问题的事件时间线.

LongMemEval 实测: 普通时间题 k=5 就全中 (8/10), 只有需要"穷举某主题下所有事件"的
聚合题要 k=26~42。这个模块就是为那两成题准备的 —— 所以最要紧的测试是**它不能在
另外八成上乱开**, 那会白白挤占注入预算。
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from app.services.memory.retrieval.timeline import (
    build_timeline,
    format_timeline,
    is_aggregate_time_question,
)

_NOW = datetime(2026, 7, 30, 12, 0)


def _row(content: str, *, occur=None, statement=None) -> dict:
    return {"content": content, "occur_time": occur, "statement_time": statement}


class TestDetection:
    def test_aggregate_questions_are_caught(self):
        for q in (
            "我上次去博物馆是什么时候",
            "我最近一次健身是几号",
            "这个月我去过几次电影院",
            "我学吉他多久了",
            "搬家和面试哪个先",
            "我们认识多长时间了",
        ):
            assert is_aggregate_time_question(q), f"漏判: {q}"

    def test_ordinary_messages_do_not_trigger(self):
        """误判的代价是白占注入预算, 挤掉真正相关的记忆."""
        for q in (
            "今天天气真好",
            "我有点难过",
            "你在干嘛呢",
            "帮我记一下明天要交房租",
            "推荐个电影吧",
            "昨天我去游泳了",
        ):
            assert not is_aggregate_time_question(q), f"误判: {q}"

    def test_empty_input_is_safe(self):
        assert is_aggregate_time_question("") is False


class TestBuild:
    def test_prefers_occur_time_over_statement_time(self):
        """occur_time 是事件真正发生的时刻, 有就该用它."""
        rows = [_row("去了博物馆", occur=_NOW - timedelta(days=100), statement=_NOW)]
        e = build_timeline(rows)[0]
        assert e.dated_by == "occur_time"
        assert (_NOW - e.at).days == 100

    def test_falls_back_to_statement_time(self):
        """occur_time 覆盖率只有 12%, 光靠它时间线是空的.

        实测两者都有的记忆中位只差 2 天 —— 用户讲刚发生的事时不带时间词。
        """
        rows = [_row("去了博物馆", statement=_NOW - timedelta(days=3))]
        e = build_timeline(rows)[0]
        assert e.dated_by == "statement_time"

    def test_undated_rows_are_dropped(self):
        assert build_timeline([_row("没有任何时间信息")]) == []

    def test_vague_past_memories_are_excluded(self):
        """「我小时候在苏州长大」用 statement_time 会被标成今天.

        时间线里一条错误日期会让 LLM 算出完全错误的间隔 —— 比少一条严重得多。
        """
        rows = [
            _row("我小时候在苏州长大", statement=_NOW),
            _row("以前我常去那家面馆", statement=_NOW),
            _row("昨天去了博物馆", statement=_NOW),
        ]
        out = build_timeline(rows)
        assert len(out) == 1
        assert "博物馆" in out[0].text

    def test_entries_are_sorted_by_time(self):
        rows = [
            _row("第三件", statement=_NOW),
            _row("第一件", statement=_NOW - timedelta(days=30)),
            _row("第二件", statement=_NOW - timedelta(days=10)),
        ]
        assert [e.text for e in build_timeline(rows)] == ["第一件", "第二件", "第三件"]

    def test_limit_keeps_the_most_recent(self):
        """聚合题问的多是"上次""最近几次", 久远条目边际价值更低."""
        rows = [
            _row(f"事件{i}", statement=_NOW - timedelta(days=100 - i))
            for i in range(50)
        ]
        out = build_timeline(rows, limit=10)
        assert len(out) == 10
        assert out[-1].text == "事件49"

    def test_empty_input(self):
        assert build_timeline([]) == []


class TestFormat:
    def test_never_exceeds_the_token_budget(self):
        """整套方案成立的前提就是它便宜.

        第一版按"取 40 条"排版, 实测 2529 token —— 中文 40 字一行约 56 token, 光
        按条数算会把预算翻三倍。改成按预算裁, 这条测试就是那次的回归护栏。
        """
        from app.services.memory.retrieval.context_selector import estimate_tokens
        from app.services.memory.retrieval.timeline import TIMELINE_TOKEN_BUDGET

        rows = [
            _row("这是一条很长的记忆内容" * 8, statement=_NOW - timedelta(days=i))
            for i in range(200)
        ]
        text = format_timeline(build_timeline(rows, limit=200))
        assert estimate_tokens(text) <= TIMELINE_TOKEN_BUDGET, \
            f"渲染后 {estimate_tokens(text)} token, 超出 {TIMELINE_TOKEN_BUDGET}"

    def test_budget_trimming_keeps_the_recent_end(self):
        """裁掉的应当是久远条目 —— 聚合题问的多是"上次""最近几次"."""
        rows = [
            _row(f"事件{i}发生了一些具体的事情内容", statement=_NOW - timedelta(days=200 - i))
            for i in range(200)
        ]
        text = format_timeline(build_timeline(rows, limit=200))
        assert "事件199" in text
        assert "事件0发生" not in text

    def test_lines_stay_in_chronological_order_after_trimming(self):
        """按预算是从近往前收的, 输出仍必须是时间正序 —— 乱序会让 LLM 算错先后."""
        rows = [
            _row(f"事件{i}", statement=_NOW - timedelta(days=10 - i)) for i in range(10)
        ]
        lines = format_timeline(build_timeline(rows)).splitlines()
        dates = [ln.lstrip("约")[:10] for ln in lines]
        assert dates == sorted(dates)

    def test_inferred_dates_are_marked(self):
        """statement_time 推出来的日期要标"约" —— 让 LLM 知道它不是确证的."""
        rows = [_row("去了博物馆", statement=_NOW)]
        assert format_timeline(build_timeline(rows)).startswith("约")

        rows = [_row("去了博物馆", occur=_NOW)]
        assert not format_timeline(build_timeline(rows)).startswith("约")

    def test_empty_timeline_renders_empty(self):
        """空串让调用方能直接判空跳过整段注入."""
        assert format_timeline([]) == ""


class TestWiring:
    """接线检查: 模块本身对了但没接上, 等于没做."""

    def test_vector_search_selects_the_date_columns(self):
        """时间线要日期, 而主向量检索原来只取 created_at.

        漏了这两列的话时间线永远是空的 —— 而且是静默的空, 不报错。
        """
        import inspect

        from app.services.memory.retrieval import vector_search

        sql = inspect.getsource(vector_search)
        assert "m.occur_time" in sql
        assert "m.statement_time" in sql

    def test_hybrid_returns_a_timeline_key(self):
        """空结果也要带这个键, 调用方才不用判 key 是否存在."""
        from app.services.memory.retrieval.hybrid import _EMPTY_RESULT

        assert "timeline" in _EMPTY_RESULT

    def test_prompt_section_is_registered(self):
        from app.services.prompting.registry import PROMPT_DEFINITIONS

        assert any(d.key == "chat.timeline_section" for d in PROMPT_DEFINITIONS)

    def test_prompt_template_has_the_placeholder(self):
        from app.services.prompting.defaults import CHAT_TIMELINE_SECTION_PROMPT

        assert "{timeline}" in CHAT_TIMELINE_SECTION_PROMPT

    def test_build_system_prompt_accepts_timeline(self):
        import inspect

        from app.services.chat.prompt_builder import build_system_prompt

        assert "timeline" in inspect.signature(build_system_prompt).parameters

    @pytest.mark.asyncio
    async def test_section_is_omitted_without_a_timeline(self):
        """非聚合问题不该多出这一段 —— 它会挤占其他记忆的预算."""
        from app.services.chat.prompt_builder import _build_timeline_section

        assert await _build_timeline_section(None) is None
        assert await _build_timeline_section("") is None
