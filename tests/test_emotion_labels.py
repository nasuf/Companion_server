"""Emotion label service tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def test_emotion_to_tone_uses_label_signal():
    from app.services.relationship.emotion import emotion_to_tone

    assert emotion_to_tone({"emotion": "焦虑", "intensity": 80}) == "焦虑而紧绷"
    assert emotion_to_tone({"emotion": "高兴", "intensity": 70}) == "轻快而亲近"
    assert emotion_to_tone(None) == "平稳而克制"
    assert emotion_to_tone({}) == "平稳而克制"


def test_quick_emotion_estimate_returns_label_signal():
    from app.services.relationship.emotion import quick_emotion_estimate

    result = quick_emotion_estimate("我现在很焦虑，真的担心明天")

    assert result is not None
    assert result["emotion"] == "焦虑"
    assert result["intensity"] > 0
    assert result["source"] == "quick"


def test_emotion_intensity_helpers():
    from app.services.relationship.emotion import is_high_emotion, is_negative_emotion

    assert is_negative_emotion({"emotion": "失望", "intensity": 50})
    assert not is_negative_emotion({"emotion": "失望", "intensity": 20})
    assert is_high_emotion({"emotion": "焦虑", "intensity": 60})
    assert is_high_emotion({"emotion": "中性", "intensity": 75})
    assert not is_high_emotion({"emotion": "欣慰", "intensity": 45})


@pytest.mark.asyncio
async def test_analyze_user_emotion_invokes_label_prompt():
    from app.services.relationship.emotion import analyze_user_emotion

    # Disable the keyword fast-path so this exercises the LLM label path
    # explicitly (keyword-hit messages intentionally skip the LLM now).
    with patch(
        "app.services.relationship.emotion.get_prompt_text",
        AsyncMock(return_value="M={message}"),
    ), patch(
        "app.services.relationship.emotion.invoke_json",
        AsyncMock(return_value={"emotion": "悲伤", "intensity": 83, "confidence": 0.9}),
    ), patch("app.config.settings.emotion_keyword_fast_path", False):
        result = await analyze_user_emotion("我有点难过")

    assert result == {
        "emotion": "悲伤",
        "intensity": 83,
        "confidence": 0.9,
        "source": "llm",
    }


@pytest.mark.asyncio
async def test_analyze_user_emotion_falls_back_to_keywords():
    from app.services.relationship.emotion import analyze_user_emotion

    with patch(
        "app.services.relationship.emotion.get_prompt_text",
        AsyncMock(return_value="M={message}"),
    ), patch(
        "app.services.relationship.emotion.invoke_json",
        AsyncMock(side_effect=RuntimeError("llm down")),
    ):
        result = await analyze_user_emotion("谢谢你，辛苦了")

    assert result["emotion"] == "感激"
    assert result["source"] == "quick"
