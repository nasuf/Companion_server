"""当天游戏汇总并进每日自我总结.

放进 `review_daily_schedule` 而不是单开 job: 那里本来就要调 LLM 写自我回顾, 加一段
素材零成本; 而且"今天陪他下了三盘棋"和"今天下午在做皮具"出现在同一段回顾里, 才是
一个人回想一天的方式。

跟逐局记忆是两个维度 —— 单局不值得记 (native.py 已收紧到只留稀有的), 但"今天陪他
玩了三盘"值得, 那是陪伴的密度。
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from app.services.games import daily_digest as D
from app.services.games.daily_digest import GameDayDigest, render_digest


def _row(title="五子棋", status="settled", outcome="win", dur=120):
    return {"title": title, "status": status, "outcome": outcome, "dur": dur}


async def _collect(monkeypatch, rows):
    monkeypatch.setattr(D.db, "query_raw", AsyncMock(return_value=rows))
    return await D.collect_today_games(
        workspace_id="ws1", local_day_start=datetime(2026, 8, 6),
    )


@pytest.mark.asyncio
class TestCollect:
    async def test_counts_outcomes(self, monkeypatch):
        d = await _collect(monkeypatch, [
            _row(outcome="win"), _row(outcome="win"), _row(outcome="lose"),
            _row(outcome="draw"), _row(status="aborted", outcome=None),
        ])
        assert (d.total, d.finished, d.user_wins, d.ai_wins, d.draws, d.aborted) == \
            (5, 4, 2, 1, 1, 1)

    async def test_titles_sorted_by_frequency(self, monkeypatch):
        d = await _collect(monkeypatch, [
            _row(title="围棋"), _row(title="五子棋"), _row(title="五子棋"),
        ])
        assert d.titles[0] == "五子棋"

    async def test_minutes_from_total_seconds(self, monkeypatch):
        d = await _collect(monkeypatch, [_row(dur=90), _row(dur=90)])
        assert d.minutes == 3

    async def test_no_games_is_empty(self, monkeypatch):
        assert (await _collect(monkeypatch, [])).is_empty is True

    async def test_db_failure_degrades_quietly(self, monkeypatch):
        """游戏查询挂了不该让整份每日总结失败."""
        monkeypatch.setattr(D.db, "query_raw", AsyncMock(side_effect=RuntimeError("down")))
        d = await D.collect_today_games(
            workspace_id="ws1", local_day_start=datetime(2026, 8, 6),
        )
        assert d.is_empty is True

    async def test_missing_workspace_is_safe(self):
        d = await D.collect_today_games(
            workspace_id=None, local_day_start=datetime(2026, 8, 6),
        )
        assert d.is_empty is True


@pytest.mark.asyncio
class TestWindow:
    """窗口边界 —— 这里错了会静默算错账, 没人会发现."""

    async def test_query_has_both_bounds(self, monkeypatch):
        """没有上界的话, 凌晨 4:00 跑的"昨日回顾"会把今天凌晨也算进昨天."""
        captured: list = []

        async def fake(sql, *args):
            captured.append((sql, args))
            return []

        monkeypatch.setattr(D.db, "query_raw", fake)
        await D.collect_today_games(
            workspace_id="ws1", local_day_start=datetime(2026, 8, 5),
        )
        sql, args = captured[0]
        assert ">=" in sql and "<" in sql
        assert args[1].startswith("2026-08-05")
        assert args[2].startswith("2026-08-06"), "默认上界应为 start + 24h"

    async def test_explicit_end_is_honoured(self, monkeypatch):
        captured: list = []

        async def fake(sql, *args):
            captured.append(args)
            return []

        monkeypatch.setattr(D.db, "query_raw", fake)
        await D.collect_today_games(
            workspace_id="ws1",
            local_day_start=datetime(2026, 8, 5),
            local_day_end=datetime(2026, 8, 5, 12),
        )
        assert captured[0][2].startswith("2026-08-05T12")

    async def test_timestamps_are_cast_not_compared_as_text(self, monkeypatch):
        """Prisma 把 datetime 序列化成 text, 与 timestamp 列直接比较会报错 ——
        而第一版把异常吞了, 表现成"今天没玩游戏"。静默失败比报错危险得多。"""
        captured: list = []

        async def fake(sql, *args):
            captured.append(sql)
            return []

        monkeypatch.setattr(D.db, "query_raw", fake)
        await D.collect_today_games(
            workspace_id="ws1", local_day_start=datetime(2026, 8, 5),
        )
        assert "::timestamptz" in captured[0]

    async def test_failure_is_logged_not_swallowed(self, monkeypatch, caplog):
        monkeypatch.setattr(D.db, "query_raw", AsyncMock(side_effect=RuntimeError("boom")))
        with caplog.at_level("WARNING"):
            await D.collect_today_games(
                workspace_id="ws1", local_day_start=datetime(2026, 8, 5),
            )
        assert any("collect_today_games" in r.message for r in caplog.records)


class TestRender:
    def test_empty_renders_nothing(self):
        """空串让调用方能直接跳过整段, 不会往 prompt 里塞"今天玩了0局"."""
        assert render_digest(GameDayDigest()) == ""

    def test_includes_counts_and_outcome(self):
        text = render_digest(GameDayDigest(
            total=3, finished=3, user_wins=2, ai_wins=1,
            titles=["五子棋"], minutes=12,
        ))
        assert "3 局" in text and "五子棋" in text
        assert "用户赢 2 局" in text and "我赢 1 局" in text

    def test_two_titles_are_listed(self):
        text = render_digest(GameDayDigest(total=2, titles=["围棋", "象棋"]))
        assert "《围棋》和《象棋》" in text

    def test_many_titles_are_summarised(self):
        text = render_digest(GameDayDigest(
            total=6, titles=["围棋", "象棋", "跳棋", "五子棋"],
        ))
        assert "等 4 种" in text

    def test_occasional_abort_is_not_mentioned(self):
        """偶尔中途退出很正常, 提它只会让 agent 显得斤斤计较."""
        text = render_digest(GameDayDigest(total=5, finished=4, user_wins=4, aborted=1))
        assert "中途退出" not in text

    def test_mostly_aborted_is_worth_noting(self):
        """开五局跑四局是个信号, 值得让 agent 有所察觉."""
        text = render_digest(GameDayDigest(total=5, finished=1, user_wins=1, aborted=4))
        assert "4 局中途退出" in text

    def test_renders_facts_not_phrasing(self):
        """只给事实不给措辞 —— 这段会喂进 LLM 让它自己组织语言.

        在这里写好句子等于又造一个模板, 逐局记忆的模板化就是这么来的。
        """
        text = render_digest(GameDayDigest(total=2, titles=["围棋"], finished=2, user_wins=2))
        for canned in ("真开心", "很有意思", "陪伴", "我们一起"):
            assert canned not in text


class TestWiring:
    def test_prompt_has_the_placeholder(self):
        from app.services.prompting.defaults import SCHEDULE_DAILY_SUMMARY_PROMPT

        assert "{games_text}" in SCHEDULE_DAILY_SUMMARY_PROMPT

    def test_review_passes_games_text(self):
        import inspect

        from app.services.schedule_domain.schedule import review_daily_schedule

        src = inspect.getsource(review_daily_schedule)
        assert "games_text=" in src
        # 必须在 summary_prompt.format 之前算好
        assert src.index("collect_today_games") < src.index("summary_prompt")

    def test_free_names_in_review_resolve(self):
        """守 NameError.

        这段代码路径在单测里全被 mock 掉了 —— 少一个 import 测试照样绿, 但线上跑到
        就炸。实际漏过一次: timedelta 用了没导入。

        用 AST 而不是正则: 正则会把 `"".join(...)` 这类方法调用也当成裸名字, 白名单
        越滚越大, 维护成本超过它的价值。AST 能准确区分 Name 和 Attribute。
        """
        import ast
        import builtins
        import inspect
        import textwrap

        from app.services.schedule_domain import schedule as mod

        tree = ast.parse(textwrap.dedent(inspect.getsource(mod.review_daily_schedule)))
        fn = tree.body[0]

        bound: set[str] = {a.arg for a in fn.args.args}
        used: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name):
                (bound if isinstance(node.ctx, ast.Store) else used).add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.comprehension):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        bound.add(n.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)  # except ... as e
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)  # 嵌套 def
                bound.update(a.arg for a in node.args.args)

        unresolved = {
            n for n in used - bound
            if not hasattr(mod, n) and not hasattr(builtins, n)
        }
        assert not unresolved, f"这些名字既非局部也非模块可见, 线上会 NameError: {unresolved}"
