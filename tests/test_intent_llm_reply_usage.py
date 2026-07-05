"""_intent_llm_reply must route through invoke_text (resilience + usage tracking),
not a raw model.ainvoke that bypasses both."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat import orchestrator as orch


@pytest.mark.asyncio
async def test_intent_llm_reply_uses_invoke_text():
    agent = SimpleNamespace(id="a1", name="小樱")
    with (
        patch.object(orch, "build_system_prompt", new_callable=AsyncMock, return_value="SYS"),
        patch.object(orch, "get_prompt_text_or_default", new_callable=AsyncMock, return_value="{instruction}"),
        patch.object(orch, "snapshot_prompt_render_traces", return_value=[]),
        patch.object(orch, "record_prompt_render"),
        patch.object(orch, "get_chat_model", return_value=MagicMock()),
        patch.object(orch, "invoke_text", new_callable=AsyncMock, return_value="再见啦，好好休息~||多余的第二句") as mock_text,
    ):
        out = await orch._intent_llm_reply(agent, "我要睡了", "生成道别")

    mock_text.assert_awaited_once()
    # only first || segment, truncated to 60 chars
    assert out == "再见啦，好好休息~"


@pytest.mark.asyncio
async def test_intent_llm_reply_handles_none_content():
    agent = SimpleNamespace(id="a1", name="小樱")
    with (
        patch.object(orch, "build_system_prompt", new_callable=AsyncMock, return_value="SYS"),
        patch.object(orch, "get_prompt_text_or_default", new_callable=AsyncMock, return_value="{instruction}"),
        patch.object(orch, "snapshot_prompt_render_traces", return_value=[]),
        patch.object(orch, "record_prompt_render"),
        patch.object(orch, "get_chat_model", return_value=MagicMock()),
        patch.object(orch, "invoke_text", new_callable=AsyncMock, return_value=None),
    ):
        out = await orch._intent_llm_reply(agent, "我要睡了", "生成道别")

    assert out == ""
