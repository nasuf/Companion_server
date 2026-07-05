"""Emotion analysis skips the utility LLM on a high-precision keyword hit."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.relationship import emotion as emo


@pytest.mark.asyncio
async def test_keyword_hit_skips_llm():
    with (
        patch("app.services.relationship.emotion.invoke_json", new_callable=AsyncMock) as mock_llm,
        patch("app.config.settings.emotion_keyword_fast_path", True),
    ):
        result = await emo.analyze_user_emotion("哈哈哈太好了")

    assert result["emotion"] == "高兴"
    assert result["source"] == "quick"
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_no_keyword_uses_llm():
    with (
        patch("app.services.relationship.emotion.get_prompt_text",
              new_callable=AsyncMock, return_value="判断情绪 {message}"),
        patch("app.services.relationship.emotion.invoke_json",
              new_callable=AsyncMock, return_value={"emotion": "焦虑", "intensity": 60, "confidence": 0.8}),
        patch("app.config.settings.emotion_keyword_fast_path", True),
    ):
        result = await emo.analyze_user_emotion("下周那个项目评审我还没准备好")

    assert result["emotion"] == "焦虑"
    assert result["source"] == "llm"


@pytest.mark.asyncio
async def test_fast_path_disabled_always_uses_llm():
    with (
        patch("app.services.relationship.emotion.get_prompt_text",
              new_callable=AsyncMock, return_value="判断情绪 {message}"),
        patch("app.services.relationship.emotion.invoke_json",
              new_callable=AsyncMock, return_value={"emotion": "高兴", "intensity": 70, "confidence": 0.9}) as mock_llm,
        patch("app.config.settings.emotion_keyword_fast_path", False),
    ):
        result = await emo.analyze_user_emotion("哈哈哈太好了")

    # disabled → keyword not short-circuited, LLM consulted
    mock_llm.assert_awaited_once()
    assert result["source"] == "llm"


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_keyword():
    with (
        patch("app.services.relationship.emotion.get_prompt_text",
              new_callable=AsyncMock, return_value="判断情绪 {message}"),
        patch("app.services.relationship.emotion.invoke_json",
              new_callable=AsyncMock, side_effect=Exception("llm down")),
        patch("app.config.settings.emotion_keyword_fast_path", False),
    ):
        # keyword present so fallback yields it; source=quick
        result = await emo.analyze_user_emotion("谢谢你")

    assert result["emotion"] == "感激"
