from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_memory_section_adds_lightweight_importance_tags():
    from app.services.chat.prompt_builder import _build_memory_section
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    section = await _build_memory_section([
        ClassifiedMemory(
            id="m1",
            text="用户表达过强烈负面情绪",
            relevance="strong",
            score=0.9,
            source="user",
            importance=0.95,
            mention_count=4,
            last_accessed_at=datetime.now(timezone.utc),
        ),
        ClassifiedMemory(
            id="m2",
            text="AI 喜欢手作",
            relevance="medium",
            score=0.5,
            source="ai",
            importance=0.5,
        ),
    ])

    assert section is not None
    assert "(重要 · 多次提及 · 近期提到 · 和当前话题高度相关) 用户表达过强烈负面情绪" in section
    assert "回复时不要复述这些标记" in section
    assert "AI 喜欢手作" in section
