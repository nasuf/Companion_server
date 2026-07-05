"""Watermark must not advance on a transient extraction LLM failure.

Advancing on failure would permanently drop those messages from the memory
pipeline. A legitimately empty (but successful) extraction still advances.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.post_process import _pipeline_with_watermark
from app.services.memory.recording.pipeline import MemoryExtractionError


def _msgs():
    return [
        {"id": "m1", "role": "user", "content": "我喜欢喝红茶",
         "createdAt": "2026-07-05T10:00:00+00:00"},
    ]


@pytest.mark.asyncio
async def test_watermark_held_when_extraction_raises():
    with (
        patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=None)),
        patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set,
        patch(
            "app.services.chat.post_process.process_memory_pipeline",
            AsyncMock(side_effect=MemoryExtractionError("llm down")),
        ),
    ):
        n = await _pipeline_with_watermark(
            "u1", _msgs(), "c1", side="user", workspace_id="w1",
        )

    assert n == 0
    mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_watermark_advances_on_empty_but_successful_extraction():
    """空但成功的抽取 (预筛不记/噪声) 仍推进水位线, 不重复处理."""
    with (
        patch("app.services.chat.post_process.get_watermark", AsyncMock(return_value=None)),
        patch("app.services.chat.post_process.set_watermark", AsyncMock()) as mock_set,
        patch(
            "app.services.chat.post_process.process_memory_pipeline",
            AsyncMock(return_value=[]),
        ),
    ):
        n = await _pipeline_with_watermark(
            "u1", _msgs(), "c1", side="user", workspace_id="w1",
        )

    assert n == 0
    mock_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_extraction_error_flag_raises_in_pipeline():
    """extract_memories 返 _extraction_error → process_memory_pipeline 抛 MemoryExtractionError."""
    from app.services.memory.recording import pipeline as pipeline_mod

    with (
        patch.object(pipeline_mod, "should_extract_memory", return_value=True),
        patch("app.config.settings.enable_memory_prefilter", False),
        patch.object(
            pipeline_mod, "extract_memories",
            AsyncMock(return_value={
                "memories": [], "entities": [], "preferences": [], "topics": [],
                "_extraction_error": True,
            }),
        ),
        patch.object(pipeline_mod, "resolve_workspace_id", AsyncMock(return_value="w1")),
    ):
        with pytest.raises(MemoryExtractionError):
            await pipeline_mod.process_memory_pipeline(
                "u1", "user: 我喜欢喝红茶", side="user",
                statement_time=datetime(2026, 7, 5, tzinfo=timezone.utc),
            )
