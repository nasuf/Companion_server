"""W4 回归: AI 情绪连续性 (上一轮情绪衰减后驱动本轮语气)."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.services.relationship.ai_mood import (
    _HALF_LIFE_S,
    format_ai_mood_text,
    load_ai_mood,
    save_ai_mood,
)

P = "app.services.relationship.ai_mood"


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def get(self, key):
        return self.store.get(key)


@pytest.mark.asyncio
class TestMoodPersistence:
    async def test_save_then_load_fresh(self):
        fake = _FakeRedis()
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            await save_ai_mood("c1", "悲伤", 80)
            mood = await load_ai_mood("c1")
        assert mood["emotion"] == "悲伤"
        assert 70 <= mood["intensity"] <= 80  # 刚存, 几乎无衰减

    async def test_neutral_emotion_not_saved(self):
        fake = _FakeRedis()
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            await save_ai_mood("c1", "中性", 90)
        assert not fake.store

    async def test_decay_half_life(self):
        """半衰期语义: 30min 后强度减半; 衰减到 <25 → 视为平复返回 None."""
        fake = _FakeRedis()
        fake.store["ai_mood:c1"] = json.dumps({
            "emotion": "高兴", "intensity": 80,
            "ts": time.time() - _HALF_LIFE_S,  # 一个半衰期前
        })
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            mood = await load_ai_mood("c1")
        assert mood is not None and 35 <= mood["intensity"] <= 45  # ~40

        fake.store["ai_mood:c1"] = json.dumps({
            "emotion": "高兴", "intensity": 80,
            "ts": time.time() - 2 * _HALF_LIFE_S,  # 两个半衰期 → ~20 < 25
        })
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            assert await load_ai_mood("c1") is None

    async def test_redis_failure_silent(self):
        with patch(f"{P}.get_redis", AsyncMock(side_effect=RuntimeError("down"))):
            await save_ai_mood("c1", "高兴", 80)  # 不抛
            assert await load_ai_mood("c1") is None


class TestMoodText:
    def test_strong_mood_wording(self):
        text = format_ai_mood_text({"emotion": "高兴", "intensity": 70})
        assert "高兴" in text and "还挺明显" in text and "活泼" in text

    def test_faint_mood_wording(self):
        text = format_ai_mood_text({"emotion": "悲伤", "intensity": 30})
        assert "淡淡的" in text and "少一点" in text

    def test_none_mood_empty(self):
        assert format_ai_mood_text(None) == ""


@pytest.mark.asyncio
async def test_prompt_builder_renders_mood_section():
    from app.services.chat.prompt_builder import _build_ai_mood_section
    from app.services.prompting import defaults as d

    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        AsyncMock(return_value=d.CHAT_AI_MOOD_SECTION_PROMPT),
    ):
        section = await _build_ai_mood_section("高兴（还挺明显），可以活泼话多一点")
    assert section is not None
    assert section.prompt_key == "chat.ai_mood_section"
    assert "对方的情绪永远优先" in section.body

    assert await _build_ai_mood_section("") is None
