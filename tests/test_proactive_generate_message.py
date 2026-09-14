"""_generate_message 细粒度 skip_reason + anti-repetition 修复 (2026-09-14).

背景: 用户 admin QA 反复触发主动消息, 持续拿到 empty_or_skip.
根因: anti-repetition 守卫在相同 topic + LLM 复读时把消息一路兜死到 return None,
上游一律记 empty_or_skip, 用户看到"LLM 未生成有效回复"假象.

修复三条:
  1. admin_test 路径完全绕过 anti-repetition
  2. 生产路径 retry 后即使仍相似, 也 ship (重复消息 >> 静默失败)
  3. return None 前把细分 reason 塞 ctx["_skip_reason_detail"], caller propagate 出去
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.proactive import sender as sender_mod


def _build_ctx(**overrides):
    agent = MagicMock()
    agent.personality = {}
    ctx = {
        "agent": agent,
        "trigger_type": "silence_wakeup",
        "source": "greeting",
        "workspace_id": "ws-test",
        "trending_context": "",
    }
    ctx.update(overrides)
    return ctx


class TestSkipReasonDetail:
    """LLM 各种"没消息"路径把具体 reason 写进 ctx["_skip_reason_detail"]."""

    @pytest.mark.asyncio
    async def test_llm_returns_literal_skip(self):
        ctx = _build_ctx()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            with patch.object(sender_mod, "invoke_text",
                              new=AsyncMock(return_value="SKIP")):
                result = await sender_mod._generate_message(ctx)
        assert result is None
        assert ctx["_skip_reason_detail"] == "llm_skip_literal"

    @pytest.mark.asyncio
    async def test_llm_returns_too_short(self):
        ctx = _build_ctx()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            with patch.object(sender_mod, "invoke_text",
                              new=AsyncMock(return_value="嗯")):  # 1 char < 4
                result = await sender_mod._generate_message(ctx)
        assert result is None
        assert ctx["_skip_reason_detail"] == "llm_response_too_short:len=1"

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self):
        ctx = _build_ctx()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            with patch.object(sender_mod, "invoke_text",
                              new=AsyncMock(side_effect=RuntimeError("timeout"))):
                result = await sender_mod._generate_message(ctx)
        assert result is None
        assert ctx["_skip_reason_detail"] == "llm_error:RuntimeError"

    @pytest.mark.asyncio
    async def test_prompt_disabled_detail(self):
        from app.services.prompting.store import PromptDisabledError
        ctx = _build_ctx()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(side_effect=PromptDisabledError("proactive.silence_plain"))):
            result = await sender_mod._generate_message(ctx)
        assert result is None
        assert ctx["_skip_reason_detail"].startswith("prompt_disabled:")


class TestAntiRepetitionAdminBypass:
    """admin_test=True 时完全绕过 anti-repetition."""

    @pytest.mark.asyncio
    async def test_admin_test_bypasses_repeat_check(self):
        """相同 workspace + admin_test → 即使 recent 有一模一样的, 也不走守卫."""
        ctx = _build_ctx(_admin_test=True)
        # 若守卫真被调, 会看到 mock (但期望根本不调 is_repeat_of_recent)
        repeat_called = MagicMock()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            with patch.object(sender_mod, "invoke_text",
                              new=AsyncMock(return_value="最近在忙什么呀")):
                # patch 到 recent_messages 模块本身, 万一走了会命中失败断言
                with patch("app.services.proactive.recent_messages.is_repeat_of_recent",
                           new=AsyncMock(side_effect=repeat_called)):
                    result = await sender_mod._generate_message(ctx)
        assert result == "最近在忙什么呀"
        repeat_called.assert_not_called()


class TestAntiRepetitionShipsRetry:
    """生产路径: retry 后即使仍相似也 ship, 不再 return None."""

    @pytest.mark.asyncio
    async def test_retry_similar_ships_anyway(self):
        """recent 命中 → retry with diversity → retry 仍命中 → 仍 ship retry."""
        ctx = _build_ctx()  # 无 _admin_test
        # is_repeat_of_recent 两次都命中
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            # 两次 invoke_text 返回不同文本 (模拟 diversity retry 效果)
            invoke = AsyncMock(side_effect=["第一次消息啊啊啊", "第二次消息啊啊啊"])
            with patch.object(sender_mod, "invoke_text", new=invoke):
                with patch("app.services.proactive.recent_messages.is_repeat_of_recent",
                           new=AsyncMock(return_value=True)):
                    result = await sender_mod._generate_message(ctx)
        # 关键: 不该 return None, 应 ship retry (第二次消息)
        assert result == "第二次消息啊啊啊"
        assert invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_llm_short_ships_first_attempt(self):
        """retry LLM 出短文本时, ship 第一次 (虽然可能重复, 也比空好)."""
        ctx = _build_ctx()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            # 第一次正常, 第二次太短
            invoke = AsyncMock(side_effect=["第一次正常消息", "嗯"])
            with patch.object(sender_mod, "invoke_text", new=invoke):
                # 第一次判 repeat, 触发 retry
                is_repeat = AsyncMock(side_effect=[True, False])
                with patch("app.services.proactive.recent_messages.is_repeat_of_recent",
                           new=is_repeat):
                    result = await sender_mod._generate_message(ctx)
        # retry 短 → 不覆盖 response, 但**不 return None**
        assert result == "第一次正常消息"

    @pytest.mark.asyncio
    async def test_retry_llm_exception_ships_first_attempt(self):
        """retry LLM 抛异常 → ship 第一次 (不能因为二次调用挂就完全无消息)."""
        ctx = _build_ctx()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            invoke = AsyncMock(side_effect=["第一次正常消息", RuntimeError("network")])
            with patch.object(sender_mod, "invoke_text", new=invoke):
                with patch("app.services.proactive.recent_messages.is_repeat_of_recent",
                           new=AsyncMock(return_value=True)):
                    result = await sender_mod._generate_message(ctx)
        assert result == "第一次正常消息"

    @pytest.mark.asyncio
    async def test_no_repeat_no_retry(self):
        """recent 未命中 → 一次 LLM 调用即返, 不 retry (regression 保护)."""
        ctx = _build_ctx()
        with patch.object(sender_mod, "get_prompt_text",
                          new=AsyncMock(return_value="fake tpl {personality_brief}")):
            invoke = AsyncMock(return_value="正常主动消息内容")
            with patch.object(sender_mod, "invoke_text", new=invoke):
                with patch("app.services.proactive.recent_messages.is_repeat_of_recent",
                           new=AsyncMock(return_value=False)):
                    result = await sender_mod._generate_message(ctx)
        assert result == "正常主动消息内容"
        assert invoke.await_count == 1
