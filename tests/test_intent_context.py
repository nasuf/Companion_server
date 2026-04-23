"""D4 — spec §3.3 step 1 严格实现:
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

        # reversed → 时间顺序 [好, 陪你一会儿?, 好]; 回退过滤最早出现的一条
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
    """LLM 抛异常 → 落回关键字扫描."""

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
