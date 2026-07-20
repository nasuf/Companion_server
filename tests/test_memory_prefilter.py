"""Tests for memory pre-filter deterministic guards."""

from unittest.mock import AsyncMock

import pytest

from app.services.memory.recording import pre_filter


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "assistant: 这部电影陪伴了我整个青春。",
    "assistant: 我以前经常听这种老歌。",
    "assistant: 我一直觉得真诚比热闹重要。",
])
async def test_ai_stable_self_memory_bypasses_llm_prefilter(monkeypatch, message):
    """AI 表达长期经历/观点时直接进入抽取, 不依赖小模型二分类。"""
    get_prompt = AsyncMock()
    invoke = AsyncMock()
    monkeypatch.setattr(pre_filter, "get_prompt_text", get_prompt)
    monkeypatch.setattr(pre_filter, "invoke_text", invoke)

    assert await pre_filter.should_memorize(message, side="ai") is True

    get_prompt.assert_not_awaited()
    invoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "assistant: 我很喜欢这种带爵士味的旋律。",
    "assistant: 现在听还是超有感觉。",
])
async def test_ai_pure_preference_no_longer_fast_tracked(monkeypatch, message):
    """纯偏好自述不再快速放行 — pipeline 会拦截 AI 侧新增偏好 (人设漂移防护),
    fast-track 只会白烧一次大模型抽取; 交给小模型二分类正常判定。"""
    get_prompt = AsyncMock(return_value="【我说的话】{message}")
    invoke = AsyncMock(return_value="不记")
    monkeypatch.setattr(pre_filter, "get_prompt_text", get_prompt)
    monkeypatch.setattr(pre_filter, "invoke_text", invoke)

    assert await pre_filter.should_memorize(message, side="ai") is False

    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_ai_non_stable_message_still_uses_llm_prefilter(monkeypatch):
    get_prompt = AsyncMock(return_value="【我说的话】{message}")
    invoke = AsyncMock(return_value="不记")
    monkeypatch.setattr(pre_filter, "get_prompt_text", get_prompt)
    monkeypatch.setattr(pre_filter, "invoke_text", invoke)

    assert await pre_filter.should_memorize("assistant: 你吃饭了吗", side="ai") is False

    get_prompt.assert_awaited_once()
    invoke.assert_awaited_once()
