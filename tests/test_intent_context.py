"""Spec §3.3 step 1 严格实现:
每条用户消息都调 LLM 做意图识别, 不再因消息长度跳过;
prompt 带上最近对话作为上下文, 让 "好" 跟在 AI "要我再陪你吗?" 之后
能被判为 作息调整 意图 (而不是单看 "好" 被降级为 日常交流)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat.intent_dispatcher import (
    IntentResult,
    IntentType,
    detect_intent_llm,
    detect_intent,
    detect_intent_unified,
)


def _row(role: str, content: str, id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content, id=id)


class TestUnifiedPassesContextToPrompt:
    """unified_intent_recognize 必须把 context 传给 prompt render."""

    @pytest.mark.asyncio
    async def test_context_forwarded_to_prompt_render(self):
        """Context 字符串必须出现在 render_prompt 的参数里."""
        from app.services.chat.intent_replies import unified_intent_recognize

        captured: dict = {}

        async def _fake_render(key, vars, _caller, **_kw):
            captured.update(vars)
            return "作息调整"

        with patch(
            "app.services.chat.intent_replies.render_prompt",
            new=AsyncMock(side_effect=_fake_render),
        ):
            labels = await unified_intent_recognize(
                "好",
                context="AI: 要我再陪你一会儿吗?\n用户: 好",
            )

        assert labels == ["作息调整"]
        assert captured["user_message"] == "好"
        assert "要我再陪你一会儿吗" in captured["context"]

    @pytest.mark.asyncio
    async def test_empty_context_fills_placeholder(self):
        """空 context → prompt 里显示 "(无)" 而不是空字符串."""
        from app.services.chat.intent_replies import unified_intent_recognize

        captured: dict = {}

        async def _fake_render(key, vars, _caller, **_kw):
            captured.update(vars)
            return "日常交流"

        with patch(
            "app.services.chat.intent_replies.render_prompt",
            new=AsyncMock(side_effect=_fake_render),
        ):
            await unified_intent_recognize("你好")

        assert captured["context"] == "(无)"


class TestDetectIntentUnifiedForwardsContext:
    """detect_intent_unified → detect_intent_llm → unified_intent_recognize
    context 参数必须传到底."""

    @pytest.mark.asyncio
    async def test_context_flows_through(self):
        """上下文透传: detect_intent_unified 传给 detect_intent_llm,
        再传给 unified_intent_recognize."""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["作息调整"]),
        ) as mock_recognize:
            result = await detect_intent_unified(
                "好", context="AI: 要我再陪你吗?"
            )

        mock_recognize.assert_awaited_once_with("好", context="AI: 要我再陪你吗?")
        assert result.intent == IntentType.SCHEDULE_ADJUST

    @pytest.mark.asyncio
    async def test_short_message_still_calls_llm(self):
        """短消息 "好" (1 字符) 必须走 LLM 而不是被关键字快路径短路."""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["作息调整"]),
        ) as mock_recognize:
            await detect_intent_unified("好", context="AI: 晚点睡行吗?")

        # 关键: 即使 "好" 只有 1 字符, LLM 被实实在在调用了
        mock_recognize.assert_awaited_once()


class TestL3RecallIntentGuard:
    """L3 只用于明确久远记忆；普通记忆查询留在日常交流路径。"""

    @pytest.mark.asyncio
    async def test_plain_previous_recall_downgraded_to_daily_chat(self):
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["调用久远记忆"]),
        ):
            result = await detect_intent_llm(
                "你记得我上次和你说的那家书店吗 我五一准备去"
            )

        assert result.intent == IntentType.NONE

    @pytest.mark.asyncio
    async def test_explicit_old_recall_keeps_l3_intent(self):
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["调用久远记忆"]),
        ):
            result = await detect_intent_llm("你还记得半年前我说的那家书店吗")

        assert result.intent == IntentType.L3_RECALL


class TestIntentContextFetching:
    """orchestrator._fetch_intent_context 从 DB 拉最近消息组装成 prompt 段落."""

    @pytest.mark.asyncio
    async def test_fetch_formats_and_reverses(self):
        from app.services.chat.orchestrator import _fetch_intent_context

        rows = [  # DB desc 顺序 (最新在前)
            _row("user", "好", id="msg-current"),
            _row("assistant", "要我再陪你一会儿吗?", id="msg-ai2"),
            _row("user", "我今天有点累", id="msg-user1"),
            _row("assistant", "怎么了?", id="msg-ai1"),
        ]
        with patch("app.services.chat.orchestrator.db") as mock_db:
            mock_db.message = MagicMock()
            mock_db.message.find_many = AsyncMock(return_value=rows)

            context = await _fetch_intent_context("conv-1", exclude_id="msg-current")

        # 应该: 反转为时间顺序 + 排除当前 "好" + role 映射到 AI/用户
        assert "AI: 怎么了?" in context
        assert "用户: 我今天有点累" in context
        assert "AI: 要我再陪你一会儿吗?" in context
        # "好" 不出现 (exclude_id 精确排除)
        assert "用户: 好" not in context

    @pytest.mark.asyncio
    async def test_duplicate_short_reply_preserved_when_id_excludes_current(self):
        """快速连发 "好"/"好": 用 id 排除当前那条, 前一轮的 "好" 仍保留在 context 里."""
        from app.services.chat.orchestrator import _fetch_intent_context

        rows = [
            _row("user", "好", id="msg-current"),    # 当前用户消息 (排除)
            _row("assistant", "要我再陪你一会儿吗?", id="msg-ai2"),
            _row("user", "好", id="msg-prev"),        # 上一轮用户 "好" (保留)
            _row("assistant", "今天累吗?", id="msg-ai1"),
        ]
        with patch("app.services.chat.orchestrator.db") as mock_db:
            mock_db.message = MagicMock()
            mock_db.message.find_many = AsyncMock(return_value=rows)

            context = await _fetch_intent_context("conv-1", exclude_id="msg-current")

        # 前一轮的 "好" 必须保留 — 这是修 "exclude_content 误伤" bug 的关键
        assert context.count("用户: 好") == 1
        assert "AI: 今天累吗?" in context

    @pytest.mark.asyncio
    async def test_exclude_content_fallback_matches_first_only(self):
        """未传 exclude_id 时按 content 回退, 只过滤最后出现的那一条 (避免误伤历史重复)."""
        from app.services.chat.orchestrator import _fetch_intent_context

        rows = [
            _row("user", "好"),
            _row("assistant", "陪你一会儿?"),
            _row("user", "好"),  # 前一轮用户也说 "好" — 回退逻辑只过滤当前那条
        ]
        with patch("app.services.chat.orchestrator.db") as mock_db:
            mock_db.message = MagicMock()
            mock_db.message.find_many = AsyncMock(return_value=rows)

            context = await _fetch_intent_context("conv-1", exclude_content="好")

        # rows 按 desc 迭代, 回退过滤首个命中 (即最新那条当前消息);
        # 剩余应至少保留一条 "用户: 好" 和 AI 那句
        assert context.count("用户: 好") == 1
        assert "AI: 陪你一会儿?" in context

    @pytest.mark.asyncio
    async def test_fetch_handles_db_failure_gracefully(self):
        from app.services.chat.orchestrator import _fetch_intent_context

        with patch("app.services.chat.orchestrator.db") as mock_db:
            mock_db.message = MagicMock()
            mock_db.message.find_many = AsyncMock(side_effect=RuntimeError("db down"))

            context = await _fetch_intent_context("conv-1")

        assert context == ""  # 失败不抛, 返回空让 LLM 走无上下文路径

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_string(self):
        from app.services.chat.orchestrator import _fetch_intent_context

        with patch("app.services.chat.orchestrator.db") as mock_db:
            mock_db.message = MagicMock()
            mock_db.message.find_many = AsyncMock(return_value=[])

            context = await _fetch_intent_context("conv-1")

        assert context == ""


class TestIntentLlmFallback:
    """LLM 抛异常 → 落回关键字扫描. LLM 返 NONE 是合法结果, 不该 fallback."""

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_keyword(self):
        """注意: 关键字扫描用 message 单独判, 不看 context."""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(side_effect=RuntimeError("LLM timeout")),
        ):
            # "对不起" 命中关键字, 应返回 APOLOGY_PROMISE
            result = await detect_intent_unified("对不起", context="")

        assert result.intent == IntentType.APOLOGY_PROMISE

    @pytest.mark.asyncio
    async def test_llm_returns_none_does_not_fallback_to_keyword(self):
        """生产 bug 复现 (2026-05-03 trace 019dec46): LLM 返"日常交流" → NONE,
        之前会 fallback 到 keyword scan, 用户 "明天是我的生日" 被 _SCHEDULE_QUERY_KEYWORDS
        ['date'] 含 "明天" 错路由到 SCHEDULE_QUERY → AI 跑去播报自己日程.

        修复后: LLM 明确说 NONE 是合法结果, 必须 trust, 不 fallback."""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["日常交流"]),
        ):
            # "明天是我的生日" 含 "明天" → keyword scan 会命中 SCHEDULE_QUERY,
            # 但 LLM 已说 "日常交流" → 必须 trust LLM, 返 NONE
            result = await detect_intent_unified(
                "明天是我的生日", context="",
            )

        assert result.intent == IntentType.NONE, (
            f"LLM 返'日常交流' 必须 trust 不 fallback; "
            f"got {result.intent} (可能是 keyword scan 误命中 'SCHEDULE_QUERY')"
        )

    @pytest.mark.asyncio
    async def test_llm_returns_empty_labels_does_not_fallback(self):
        """LLM 返空 labels (空字符串 / 全是干扰词) → 也是合法 NONE, 不 fallback."""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=[]),
        ):
            result = await detect_intent_unified(
                "明天周末有空吗", context="",  # 故意含多个 schedule keyword
            )
        assert result.intent == IntentType.NONE


class TestScheduleQueryType:
    def test_keyword_fallback_prefers_date_over_current(self):
        """你明天忙吗 同时含 date/current 词，date 必须优先。"""
        result = detect_intent("你明天忙吗？")

        assert result.intent == IntentType.SCHEDULE_QUERY
        assert result.metadata["query_type"] == "date"

    def test_keyword_fallback_does_not_treat_bare_date_as_schedule_query(self):
        """显式时间不是计划查询意图本身：明天是事实陈述时不能短路日程。"""
        result = detect_intent("明天是我的生日")

        assert result.intent == IntentType.NONE

    def test_keyword_fallback_does_not_treat_first_person_plan_as_schedule_query(self):
        """fallback 要保守：用户陈述自己的计划，不应被安排/计划词误路由。"""
        result = detect_intent("我明天计划去看牙")

        assert result.intent == IntentType.NONE

    def test_keyword_fallback_uses_parser_for_next_weekday(self):
        """非固定关键词日期也应由时间解析器统一归为 date 查询。"""
        result = detect_intent("你下周三忙吗？")

        assert result.intent == IntentType.SCHEDULE_QUERY
        assert result.metadata["query_type"] == "date"

    def test_keyword_fallback_uses_parser_for_week_range(self):
        result = detect_intent("你下周忙吗？")

        assert result.intent == IntentType.SCHEDULE_QUERY
        assert result.metadata["query_type"] == "date"

    def test_keyword_fallback_keeps_today_how_are_you_as_current(self):
        """今天怎么样 是当前状态问法，不应被 今天 解析成 date 查询。"""
        result = detect_intent("你今天怎么样？")

        assert result.intent == IntentType.SCHEDULE_QUERY
        assert result.metadata["query_type"] == "current"

    @pytest.mark.asyncio
    async def test_llm_schedule_query_gets_structured_query_type(self):
        """LLM 只返回计划查询标签时，仍要从原文补 query_type。"""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["计划查询"]),
        ):
            result = await detect_intent_unified("你下周三忙吗？", context="")

        assert result.intent == IntentType.SCHEDULE_QUERY
        assert result.metadata["query_type"] == "date"


class TestSpecExampleEndToEnd:
    """spec §3.3 用例: "好" 跟在 AI 问题之后应识别 作息调整."""

    @pytest.mark.asyncio
    async def test_short_reply_with_schedule_context(self):
        """模拟完整链路: 用户 "好" + AI 上一句是作息提议, LLM 判 作息调整."""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["作息调整"]),
        ):
            result = await detect_intent_unified(
                "好",
                context="AI: 要我再陪你一会儿吗?\n用户: 我还不想睡",
            )

        assert result.intent == IntentType.SCHEDULE_ADJUST

    @pytest.mark.asyncio
    async def test_short_reply_without_context_daily(self):
        """同样的 "好" 无上下文 → LLM 判 日常交流 → 不触发任何 handler."""
        with patch(
            "app.services.chat.intent_replies.unified_intent_recognize",
            new=AsyncMock(return_value=["日常交流"]),
        ):
            result = await detect_intent_unified("好", context="")

        assert result.intent == IntentType.NONE
