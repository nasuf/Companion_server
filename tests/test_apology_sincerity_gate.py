"""Spec §2.6.2.1: 道歉恢复耐心必须过 sincerity >= 0.5 门禁.

两条路径都受门禁约束:
  Path A — intent.unified 识别 apology_promise 后调 handle_apology_promise()
  Path B — 拉黑态下 boundary_phase._handle_blocked() 内置 detect_apology+gate

本测试只覆盖 Path A (D14 修复点). Path B 由 boundary 测试覆盖.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat.intent_handlers import (
    ShortCircuitCtx,
    handle_apology_promise,
)
from app.services.interaction.boundary import PATIENCE_MAX


def _make_ctx(
    *,
    cached_patience: int = 50,
    agent_id: str = "agent-1",
) -> ShortCircuitCtx:
    """构造最小可用的 ShortCircuitCtx stub."""
    agent = MagicMock()
    agent.id = agent_id
    agent.name = "Lumia"
    agent.personality = {}
    return ShortCircuitCtx(
        conversation_id="conv-1",
        agent_id=agent_id,
        user_id="user-1",
        agent=agent,
        reply_context=None,
        tracer=MagicMock(),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=cached_patience,
    )


class TestHandleApologyPromise:
    @pytest.mark.asyncio
    async def test_high_sincerity_restores_patience(self):
        """诚意 >= 0.5 → 过门禁, 调用 handle_apology + 返回 True."""
        ctx = _make_ctx(cached_patience=30)
        with patch(
            "app.services.chat.intent_handlers.detect_apology",
            new=AsyncMock(return_value={"is_apology": True, "sincerity": 0.8}),
        ), patch(
            "app.services.chat.intent_handlers.handle_apology",
            new=AsyncMock(return_value=70),
        ) as mock_handle, patch(
            "app.services.chat.intent_handlers.apology_reply",
            new=AsyncMock(return_value="没事了~"),
        ):
            handled, _events = await handle_apology_promise("对不起", ctx)

        assert handled is True
        assert mock_handle.await_count == 1

    @pytest.mark.asyncio
    async def test_low_sincerity_blocks_patience_restore(self):
        """诚意 < 0.5 → 门禁拦截, 不调 handle_apology, 返回 False 走正常 reply."""
        ctx = _make_ctx(cached_patience=30)
        with patch(
            "app.services.chat.intent_handlers.detect_apology",
            new=AsyncMock(return_value={"is_apology": True, "sincerity": 0.3}),
        ), patch(
            "app.services.chat.intent_handlers.handle_apology",
            new=AsyncMock(return_value=50),
        ) as mock_handle:
            handled, events = await handle_apology_promise(
                "对不起啦但我就是讨厌你", ctx,
            )

        assert handled is False
        assert events is None
        assert mock_handle.await_count == 0

    @pytest.mark.asyncio
    async def test_not_apology_blocks_path(self):
        """detect_apology 判定不是道歉 → 不恢复."""
        ctx = _make_ctx(cached_patience=30)
        with patch(
            "app.services.chat.intent_handlers.detect_apology",
            new=AsyncMock(return_value={"is_apology": False, "sincerity": 0.9}),
        ), patch(
            "app.services.chat.intent_handlers.handle_apology",
            new=AsyncMock(),
        ) as mock_handle:
            handled, _ = await handle_apology_promise("你好", ctx)

        assert handled is False
        assert mock_handle.await_count == 0

    @pytest.mark.asyncio
    async def test_patience_already_max_skips_entirely(self):
        """耐心已满 → 连 detect_apology 都不调, 早返 False."""
        ctx = _make_ctx(cached_patience=PATIENCE_MAX)
        with patch(
            "app.services.chat.intent_handlers.detect_apology",
            new=AsyncMock(),
        ) as mock_detect:
            handled, _ = await handle_apology_promise("对不起", ctx)

        assert handled is False
        assert mock_detect.await_count == 0  # 未触发 LLM

    @pytest.mark.asyncio
    async def test_detect_apology_exception_falls_through(self):
        """detect_apology 抛异常 → except 兜底返回 False, 不恢复耐心."""
        ctx = _make_ctx(cached_patience=30)
        with patch(
            "app.services.chat.intent_handlers.detect_apology",
            new=AsyncMock(side_effect=RuntimeError("LLM timeout")),
        ), patch(
            "app.services.chat.intent_handlers.handle_apology",
            new=AsyncMock(),
        ) as mock_handle:
            handled, _ = await handle_apology_promise("对不起", ctx)

        assert handled is False
        assert mock_handle.await_count == 0

    @pytest.mark.asyncio
    async def test_sincerity_boundary_exactly_05(self):
        """sincerity 恰好 == 0.5 也应过门禁 (spec 阈值含等)."""
        ctx = _make_ctx(cached_patience=30)
        with patch(
            "app.services.chat.intent_handlers.detect_apology",
            new=AsyncMock(return_value={"is_apology": True, "sincerity": 0.5}),
        ), patch(
            "app.services.chat.intent_handlers.handle_apology",
            new=AsyncMock(return_value=70),
        ) as mock_handle, patch(
            "app.services.chat.intent_handlers.apology_reply",
            new=AsyncMock(return_value="没事"),
        ):
            handled, _ = await handle_apology_promise("对不起", ctx)

        assert handled is True
        assert mock_handle.await_count == 1
