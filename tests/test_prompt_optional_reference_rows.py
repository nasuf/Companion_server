from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.prompting.utils import compact_optional_reference_rows, render_prompt


def test_compact_optional_reference_rows_drops_empty_reference_header():
    template = """【任务】自然回复。

【参考信息】
- 用户画像：{user_portrait}

【要求】
- 简短。"""

    compacted = compact_optional_reference_rows(
        template,
        {"user_portrait": "(未知)"},
        optional_keys={"user_portrait"},
    )

    assert "用户画像" not in compacted
    assert "【参考信息】" not in compacted
    assert "【要求】" in compacted


def test_compact_optional_reference_rows_keeps_required_placeholder_rows():
    template = """【参考信息】
- 事项：{summary}
- 用户画像：{user_portrait}"""

    compacted = compact_optional_reference_rows(
        template,
        {"summary": "(未知)", "user_portrait": "(未知)"},
        optional_keys={"user_portrait"},
    )

    assert "- 事项：{summary}" in compacted
    assert "用户画像" not in compacted


@pytest.mark.asyncio
async def test_render_prompt_filters_optional_empty_rows_before_invoke():
    captured: dict[str, str] = {}

    async def _fake_invoke(prompt: str) -> str:
        captured["prompt"] = prompt
        return "ok"

    template = """【任务】自然回复。
【参考信息】
- 用户刚才说：{message}
- 最近对话：{context}
- 用户画像：{user_portrait}
- 你的性格：{personality_brief}
【输出】只输出回复。"""

    with patch(
        "app.services.prompting.utils.get_prompt_text",
        new=AsyncMock(return_value=template),
    ):
        result = await render_prompt(
            "dummy",
            {
                "message": "好",
                "context": "(无)",
                "user_portrait": "(未知)",
                "personality_brief": "温和",
            },
            _fake_invoke,
            optional_keys={"context", "user_portrait"},
        )

    assert result == "ok"
    assert "- 用户刚才说：好" in captured["prompt"]
    assert "- 你的性格：温和" in captured["prompt"]
    assert "最近对话" not in captured["prompt"]
    assert "用户画像" not in captured["prompt"]


def test_proactive_prompt_filters_empty_optional_reference_rows():
    from app.services.proactive.sender import _format_prompt
    from app.services.prompting.defaults import PROACTIVE_SILENCE_AI_MEMORY_PROMPT

    prompt = _format_prompt(
        "proactive.silence_ai_memory",
        {
            "topic_theme": "日常",
            "proactive_memories": [],
            "schedule_status": {"activity": "散步", "status": "idle"},
            "user_portrait": "(未知)",
            "recent_context": "(无)",
            "emotion": {"pleasure": 0.0, "arousal": 0.3, "dominance": 0.5},
            "__tpl": PROACTIVE_SILENCE_AI_MEMORY_PROMPT,
        },
        "温和",
    )

    assert prompt is not None
    assert "你当前心境" in prompt
    assert "你想起的自身记忆" not in prompt
    assert "用户画像" not in prompt
    assert "近期对话" not in prompt
