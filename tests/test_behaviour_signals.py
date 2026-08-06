"""行为事实层的守卫.

这一层存在的全部理由是"喂给画像的数字必须是对的" —— 模型会忠实地总结错误的数据,
产出一句听起来合理的假判断, 而整条链路不报任何错。

所以这里既测算得对不对, 也钉住开发时踩过的三个真实错误: 时区方向反了、把每条用户
消息都会写的衰减重置事件当成"回应了主动消息"、以及 date 列取回来是字符串导致相减
抛异常被吞掉。
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory import behaviour_signals as signals
from app.services.memory.behaviour_signals import (
    LOCAL_UTC_OFFSET_HOURS,
    MIN_MESSAGES_FOR_TIMING,
    BehaviouralFact,
    _as_date,
    _describe_hour_band,
    collect_behavioural_facts,
    format_facts_for_prompt,
)


class _StripDocstrings(ast.NodeTransformer):
    """递归去掉所有 docstring, 只留可执行代码。

    必须递归: `ast.unparse` 会原样保留嵌套函数的 docstring, 而这些注释里恰恰会
    提到"我们刻意不用 AT TIME ZONE"这类字样, 让禁用字面量的检查误报。
    """

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


def _code_only(source: str) -> str:
    """源码去掉注释与 docstring 后的可执行部分。

    不能靠按三引号切分字符串 —— SQL 字面量本身就是三引号包的, 那样会把 SQL 一起
    切掉, 于是「SQL 里有没有某个片段」的断言全部落空。更糟的是断言「不存在」的
    那些会静默通过, 看起来一片绿。
    """
    tree = _StripDocstrings().visit(ast.parse(textwrap.dedent(source)))
    return ast.unparse(ast.fix_missing_locations(tree))


def _body_without_docstring(func) -> str:
    return _code_only(inspect.getsource(func))


class TestTimezone:
    """开发时的真实事故: 用 AT TIME ZONE 算出"活跃峰值在凌晨 5 点", 真实是 21 点。"""

    def test_local_hour_adds_the_offset_rather_than_shifting_the_other_way(self):
        expr = signals._local_hour_expr("m.created_at")
        assert f"+ INTERVAL '{LOCAL_UTC_OFFSET_HOURS} hours'" in expr
        assert "AT TIME ZONE" not in expr, (
            "AT TIME ZONE 对 timestamp without time zone 的方向跟直觉相反 —— "
            "它会减 8 小时, 于是晚上 21 点被算成凌晨 5 点"
        )

    def test_local_date_uses_the_same_offset(self):
        """日期和小时必须用同一个偏移, 否则"活跃天数"和"活跃时段"会对不上。"""
        assert (
            f"+ INTERVAL '{LOCAL_UTC_OFFSET_HOURS} hours'"
            in signals._local_date_expr("m.created_at")
        )

    def test_no_module_in_the_reflection_package_uses_at_time_zone(self):
        """整个包都不许出现这个字面量 —— 一处写对不够, 下次新增聚合会再踩。"""
        package = Path(signals.__file__).parent
        offenders = [
            path.name for path in package.glob("*.py")
            if "AT TIME ZONE" in _code_only(path.read_text())
        ]
        assert not offenders, f"{offenders} 里用了 AT TIME ZONE"

    @pytest.mark.parametrize("hour,band", [
        (2, "凌晨"), (7, "清晨"), (10, "上午"),
        (13, "中午"), (16, "下午"), (21, "晚上"), (23, "深夜"),
    ])
    def test_hour_bands(self, hour, band):
        assert _describe_hour_band(hour) == band


class TestProactiveResponseSemantics:
    """初版数了 proactive_event_logs 的 user_replied, 在生产上算出
    "我主动找了 4 次, 他回了 102 次"。那个事件是每条用户消息都写的 (挂在 ws/chat
    入口用来重置沉默衰减), 跟主动消息没有对应关系。

    数字荒谬所以一眼看穿。但若比例落在合理区间, 它会变成一条"用户对我的主动搭话
    很热情"的洞见写进记忆 —— 完全虚假, 且没有任何报错。
    """

    def test_does_not_count_the_decay_reset_event(self):
        body = _body_without_docstring(signals._proactive_response_fact)
        assert "user_replied" not in body, (
            "又在数 user_replied —— 那是每条用户消息都会写的衰减重置事件"
        )
        assert "proactive_event_logs" not in body

    def test_derives_replies_from_the_message_stream(self):
        body = _body_without_docstring(signals._proactive_response_fact)
        assert "metadata->>'proactive'" in body
        assert "role = 'user'" in body

    @pytest.mark.asyncio
    async def test_drops_the_fact_when_replies_exceed_sends(self):
        """自洽性兜底: 回应数不可能超过发送数。真出现了说明语义又错了。"""
        with patch.object(
            signals.db, "query_raw",
            AsyncMock(return_value=[{"sent": 4, "answered": 102}]),
        ):
            fact = await signals._proactive_response_fact("u", "a", date(2026, 1, 1), "ws-1")
        assert fact is None

    @pytest.mark.asyncio
    async def test_reports_a_consistent_pair(self):
        with patch.object(
            signals.db, "query_raw",
            AsyncMock(return_value=[{"sent": 6, "answered": 4}]),
        ):
            fact = await signals._proactive_response_fact("u", "a", date(2026, 1, 1), "ws-1")
        assert fact is not None
        assert fact.evidence == {"sent": 6, "answered": 4}


class TestDateParsing:
    """query_raw 取回 date 列给的是字符串, 直接相减抛 TypeError —— 而调用方吞异常,
    于是这条事实一直静默缺席。"""

    def test_parses_the_string_form_the_driver_returns(self):
        assert _as_date("2026-07-28") == date(2026, 7, 28)

    def test_parses_a_full_timestamp_string(self):
        assert _as_date("2026-07-28T13:00:00+00:00") == date(2026, 7, 28)

    def test_returns_none_instead_of_raising(self):
        assert _as_date("не дата") is None
        assert _as_date(None) is None

    @pytest.mark.asyncio
    async def test_rhythm_survives_string_dates(self):
        rows = [{"day": f"2026-07-{d:02d}", "n": 5} for d in range(20, 27)]
        with patch.object(signals.db, "query_raw", AsyncMock(return_value=rows)):
            fact = await signals._rhythm_fact("u", "a", date(2026, 1, 1), "ws-1")
        assert fact is not None
        assert fact.evidence["active_days"] == 7
        assert fact.evidence["span_days"] == 7


class TestSampleSizeFloors:
    """样本量不足时缺席, 而不是产出一个基于三条数据的"趋势"。"""

    @pytest.mark.asyncio
    async def test_timing_needs_enough_messages(self):
        rows = [{"hour": 21, "n": MIN_MESSAGES_FOR_TIMING - 1}]
        with patch.object(signals.db, "query_raw", AsyncMock(return_value=rows)):
            assert await signals._timing_fact("u", "a", date(2026, 1, 1), "ws-1") is None

    @pytest.mark.asyncio
    async def test_timing_emits_above_the_floor(self):
        rows = [{"hour": 21, "n": MIN_MESSAGES_FOR_TIMING}]
        with patch.object(signals.db, "query_raw", AsyncMock(return_value=rows)):
            fact = await signals._timing_fact("u", "a", date(2026, 1, 1), "ws-1")
        assert fact is not None
        assert "21点" in fact.statement

    @pytest.mark.asyncio
    async def test_rhythm_needs_enough_active_days(self):
        rows = [{"day": f"2026-07-2{d}", "n": 3} for d in range(0, 3)]
        with patch.object(signals.db, "query_raw", AsyncMock(return_value=rows)):
            assert await signals._rhythm_fact("u", "a", date(2026, 1, 1), "ws-1") is None


class TestScoping:
    """会话是按 workspace 分的, 而洞见只写进其中一个。不收口的话统计会混进别的
    workspace (比如重建 agent 前的旧会话), 得出的"他最近很活跃"说的是另一段关系。"""

    def test_every_query_filters_by_workspace(self):
        producers = [
            getattr(signals, name) for name in dir(signals)
            if name.endswith("_fact") and name.startswith("_")
        ]
        assert producers, "没找到事实生产函数"
        for produce in producers:
            body = _body_without_docstring(produce)
            assert "workspace_id IS NOT DISTINCT FROM" in body, (
                f"{produce.__name__} 没有按 workspace 收口"
            )

    def test_every_query_excludes_deleted_conversations(self):
        """用户删掉的对话不该继续影响 AI 对他的判断。

        接受两种写法: 消息类事实走 `is_deleted = false` (INNER JOIN, 消息必然属于
        某个会话); 游戏类走 `is_deleted IS NOT TRUE` (LEFT JOIN, 对局的
        conversation_id 可空 —— 从游戏入口直接开的局没有会话, 那种不算被删,
        用 INNER JOIN 会把它们整体丢掉)。
        """
        producers = [
            getattr(signals, name) for name in dir(signals)
            if name.endswith("_fact") and name.startswith("_")
        ]
        for produce in producers:
            body = _body_without_docstring(produce)
            assert (
                "is_deleted = false" in body or "is_deleted IS NOT TRUE" in body
            ), f"{produce.__name__} 统计了已删除会话"

    @pytest.mark.asyncio
    async def test_workspace_is_threaded_through_collection(self):
        seen: list = []

        async def _query(sql, *args):
            seen.append(args)
            return []

        with patch.object(signals.db, "query_raw", _query):
            await collect_behavioural_facts(
                user_id="u", agent_id="a", workspace_id="ws-42",
            )
        assert seen and all("ws-42" in args for args in seen)


class TestCollection:
    @pytest.mark.asyncio
    async def test_one_broken_aggregate_does_not_sink_the_rest(self):
        """少一条事实只是让归纳少点依据; 让整轮反思因为一个查询出错而失败不值得。"""
        calls = {"n": 0}

        async def _flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return [{"hour": 21, "n": 100}]

        with patch.object(signals.db, "query_raw", _flaky):
            facts = await collect_behavioural_facts(user_id="u", agent_id="a", workspace_id="ws-1")
        assert calls["n"] > 1, "第一个查询失败后就不往下走了"

    @pytest.mark.asyncio
    async def test_no_data_yields_no_facts(self):
        with patch.object(signals.db, "query_raw", AsyncMock(return_value=[])):
            assert await collect_behavioural_facts(user_id="u", agent_id="a", workspace_id="ws-1") == []

    def test_prompt_format_numbers_the_facts(self):
        """编号是给 LLM 引用用的 —— 引用不上的洞见会被丢弃。"""
        rendered = format_facts_for_prompt([
            BehaviouralFact(key="a", statement="事实甲", sample_size=10),
            BehaviouralFact(key="b", statement="事实乙", sample_size=20),
        ])
        assert rendered == "[1] 事实甲\n[2] 事实乙"


def test_facts_carry_evidence_for_review():
    """没有 evidence 就无法复核一条洞见是不是建立在错数据上。"""
    fact = BehaviouralFact(key="x", statement="s", sample_size=1)
    assert isinstance(fact.evidence, dict)
    for producer_name in dir(signals):
        if not producer_name.endswith("_fact"):
            continue
        source = inspect.getsource(getattr(signals, producer_name))
        assert "evidence=" in source, f"{producer_name} 没带 evidence"


def _game_rows(*specs):
    """specs: (game_key, total, finished, won, lost, title)"""
    return [
        {"game_key": k, "total": t, "finished": f, "won": w, "lost": l, "title": ti}
        for k, t, f, w, l, ti in specs
    ]


async def _game_fact(rows):
    with patch.object(signals.db, "query_raw", AsyncMock(return_value=rows)):
        return await signals._game_fact(
            "u", "a", datetime(2026, 7, 23, tzinfo=UTC), "ws",
        )


class TestGamePattern:
    """一起玩游戏的模式.

    这条事实取代**每局一条**的游戏记忆: 单局记忆试过不成立 —— 21 条里绝大多数是
    "我们下了一盘五子棋, 他赢了"这种同构句, 向量上高度相似互相挤占检索位, 而任何
    一条单独看都不值得想起。模式属于特质不是事实, 该进画像不进检索池。
    """

    @pytest.mark.asyncio
    async def test_reports_volume_and_favourite(self):
        f = await _game_fact(_game_rows(("gomoku", 40, 20, 12, 8, "五子棋")))
        assert f is not None
        assert "40 局" in f.statement
        assert "五子棋" in f.statement

    @pytest.mark.asyncio
    async def test_a_couple_of_games_is_not_a_pattern(self):
        """玩过两局跟"喜欢玩游戏"是两回事 —— 样本不足就缺席, 不编."""
        assert await _game_fact(_game_rows(("gomoku", 2, 2, 1, 1, "五子棋"))) is None

    @pytest.mark.asyncio
    async def test_no_games_at_all_yields_nothing(self):
        assert await _game_fact([]) is None

    @pytest.mark.asyncio
    async def test_win_rate_needs_enough_decided_games(self):
        """3 局 2 胜说明不了任何事, 写进画像反而误导."""
        f = await _game_fact(_game_rows(("gomoku", 9, 3, 2, 1, "五子棋")))
        assert f is not None
        for word in ("赢得多", "输得多", "胜负"):
            assert word not in f.statement

    @pytest.mark.asyncio
    async def test_lopsided_results_are_called_out(self):
        f = await _game_fact(_game_rows(("gomoku", 40, 30, 25, 5, "五子棋")))
        assert "赢得多" in f.statement
        f2 = await _game_fact(_game_rows(("gomoku", 40, 30, 5, 25, "五子棋")))
        assert "输得多" in f2.statement

    @pytest.mark.asyncio
    async def test_even_results_are_described_as_even(self):
        f = await _game_fact(_game_rows(("gomoku", 40, 20, 10, 10, "五子棋")))
        assert "胜负差不多" in f.statement

    @pytest.mark.asyncio
    async def test_single_game_says_only(self):
        f = await _game_fact(_game_rows(("gomoku", 20, 10, 5, 5, "五子棋")))
        assert "只玩" in f.statement

    @pytest.mark.asyncio
    async def test_variety_is_summarised_not_enumerated(self):
        """玩了十种不该把十个名字都念出来 —— 画像段落是有预算的."""
        f = await _game_fact(_game_rows(
            ("gomoku", 10, 6, 3, 3, "五子棋"), ("xiangqi", 8, 4, 2, 2, "中国象棋"),
            ("reversi", 6, 3, 1, 2, "黑白棋"), ("match3", 5, 2, 1, 1, "消消乐"),
        ))
        assert "另外还玩了 3 种" in f.statement
        assert "消消乐" not in f.statement

    @pytest.mark.asyncio
    async def test_heavy_abandonment_is_itself_a_signal(self):
        """中断局占 69%, 老是开局不下完是个真实的相处特点."""
        f = await _game_fact(_game_rows(("gomoku", 40, 8, 4, 4, "五子棋")))
        assert "没下完" in f.statement

    @pytest.mark.asyncio
    async def test_finishing_most_games_is_not_flagged(self):
        f = await _game_fact(_game_rows(("gomoku", 40, 36, 18, 18, "五子棋")))
        assert "没下完" not in f.statement

    @pytest.mark.asyncio
    async def test_evidence_is_traceable(self):
        """statement 是给模型看的, evidence 是给人复核的 —— 两者都不能少."""
        f = await _game_fact(_game_rows(("gomoku", 40, 30, 25, 5, "五子棋")))
        assert f.evidence["total"] == 40
        assert f.evidence["top_game"] == "gomoku"
        assert f.sample_size == 40

    @pytest.mark.asyncio
    async def test_missing_title_falls_back_to_the_key(self):
        """老数据可能没有 game_title, 不该渲染出"《None》"."""
        f = await _game_fact(_game_rows(("gomoku", 20, 10, 5, 5, None)))
        assert "None" not in f.statement

    @pytest.mark.asyncio
    async def test_is_registered_in_the_producer_list(self):
        """写了不挂上等于没写 —— 而且不会报错, 只是这条事实永远缺席."""
        src = inspect.getsource(collect_behavioural_facts)
        assert "_game_fact" in src

    @pytest.mark.asyncio
    async def test_a_broken_game_query_does_not_sink_the_other_facts(self):
        """单项失败不影响其余 —— 少一条事实只是让归纳少一点依据."""
        with patch.object(
            signals, "_game_fact", AsyncMock(side_effect=RuntimeError("boom")),
        ), patch.object(signals.db, "query_raw", AsyncMock(return_value=[])):
            facts = await collect_behavioural_facts(
                user_id="u", agent_id="a", workspace_id="ws",
            )
        assert facts == []
