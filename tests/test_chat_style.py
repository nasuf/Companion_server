"""Tests for chat naturalness helpers."""

from app.services.chat.orchestrator import detect_relational_context
from app.services.relationship.emotion import quick_emotion_estimate


def test_detect_relational_context_for_complaint():
    context = detect_relational_context("你怎么不理我", None)
    assert context is not None
    assert "被忽略感" in context


def test_detect_relational_context_for_distress():
    context = detect_relational_context("我现在很不好", {"emotion": "悲伤", "intensity": 60})
    assert context is not None
    assert "低落" in context or "烦闷" in context


def test_quick_emotion_estimate_catches_short_negative_message():
    result = quick_emotion_estimate("我现在很烦，不好")
    assert result is not None
    assert result["emotion"] in {"悲伤", "愤怒", "焦虑"}
    assert result["intensity"] > 0
