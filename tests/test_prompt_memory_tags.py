from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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


@pytest.mark.asyncio
async def test_system_prompt_skips_empty_placeholder_sections_on_weak_memory():
    from app.services.chat.prompt_builder import build_system_prompt

    async def _prompt_text(key: str) -> str:
        return {
            "chat.system_base": "像朋友一样回复。",
            "chat.consistency_rules": "。",
            "chat.response_instruction": "分{n}条消息回复，总共不超过{total}字。",
            "chat.anti_hallucination_hard_rule": "。",
        }[key]

    with patch(
        "app.services.chat.prompt_builder.get_prompt_text",
        AsyncMock(side_effect=_prompt_text),
    ):
        diagnostics = {}
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="Hillow", values={"gender": "female"}),
            memories=None,
            memory_relevance="weak",
            reply_count=1,
            reply_total=150,
            diagnostics=diagnostics,
        )

    assert "## 核心规则" in prompt
    assert "## 反幻觉硬约束" not in prompt
    assert "## 对话一致性" not in prompt
    assert "## 你记得的事情" not in prompt
    assert diagnostics["system_prompt_section_count"] >= 3
    assert diagnostics["empty_prompt_sections_removed_count"] == len(
        diagnostics["empty_prompt_sections_removed"]
    )
    assert "反幻觉硬约束" in diagnostics["empty_prompt_sections_removed"]
    assert "对话一致性" in diagnostics["empty_prompt_sections_removed"]
    assert "你记得的事情" in diagnostics["empty_prompt_sections_removed"]


@pytest.mark.asyncio
async def test_system_prompt_keeps_empty_memory_anchor_when_hard_rule_active():
    from app.services.chat.prompt_builder import build_system_prompt

    async def _prompt_text(key: str) -> str:
        return {
            "chat.system_base": "像朋友一样回复。",
            "chat.consistency_rules": "",
            "chat.response_instruction": "分{n}条消息回复，总共不超过{total}字。",
            "chat.anti_hallucination_hard_rule": "用户问记忆时必须检查记忆段。",
        }[key]

    with patch(
        "app.services.chat.prompt_builder.get_prompt_text",
        AsyncMock(side_effect=_prompt_text),
    ):
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="Hillow", values={"gender": "female"}),
            memories=None,
            memory_relevance="medium",
            reply_count=1,
            reply_total=150,
        )

    assert "## 反幻觉硬约束" in prompt
    assert "## 对话一致性" not in prompt
    assert "## 你记得的事情" in prompt
    assert "(本次没有联想到任何与当前话题相关的记忆)" in prompt
