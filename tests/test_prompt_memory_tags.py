from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


async def _prompt_text(key: str, **_kwargs) -> str:
    from app.services.prompting import defaults
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    overrides = {
        "chat.system_base": "像朋友一样回复。",
        "chat.consistency_rules": "不要说出与记忆矛盾的话。",
        "chat.response_instruction": "分{n}条消息回复，总共不超过{total}字。",
        "chat.anti_hallucination_hard_rule": "用户问记忆时必须检查记忆段。",
        "chat.memory_empty_anchor": defaults.CHAT_MEMORY_EMPTY_ANCHOR_PROMPT,
        "chat.memory_section_body": defaults.CHAT_MEMORY_SECTION_BODY_PROMPT,
    }
    if key in overrides:
        return overrides[key]
    definition = PROMPT_DEFINITION_MAP.get(key)
    return definition.default_text if definition else ""


def _body(section) -> str:
    return section.body if hasattr(section, "body") else section


@pytest.mark.asyncio
async def test_memory_section_adds_lightweight_importance_tags():
    from app.services.chat.prompt_builder import _build_memory_section
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    with patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)), \
         patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)):
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
    section_text = _body(section)
    assert "(重要 · 多次提及 · 近期提到 · 和当前话题高度相关) 用户表达过强烈负面情绪" in section_text
    assert "回复时不要复述这些标记" in section_text
    assert "AI 喜欢手作" in section_text


@pytest.mark.asyncio
async def test_memory_section_groups_task_and_safety_memories():
    from app.services.chat.prompt_builder import _build_memory_section
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    with patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)), \
         patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)):
        section = await _build_memory_section([
            ClassifiedMemory(
                id="task",
                text="用户的直属领导叫陈姐",
                relevance="strong",
                score=0.82,
                source="user",
                rank_reasons=["保护槽:关系命名"],
            ),
            ClassifiedMemory(
                id="safety",
                text="用户表达过强烈负面情绪",
                relevance="strong",
                score=0.9,
                source="user",
                rank_reasons=["保护槽:安全情绪"],
            ),
            ClassifiedMemory(
                id="literal",
                text="用户叫林小满",
                relevance="strong",
                score=0.83,
                source="user",
                rank_reasons=["保护槽:字面命中"],
            ),
            ClassifiedMemory(
                id="current-profile-fact",
                text="用户28岁",
                relevance="strong",
                score=0.83,
                source="user",
                rank_reasons=["保护槽:当前问题事实"],
            ),
            ClassifiedMemory(
                id="other",
                text="用户喜欢日料",
                relevance="medium",
                score=0.5,
                source="user",
            ),
        ])

    assert section is not None
    section_text = _body(section)
    assert "【回答当前关系 / 名字问题优先参考】" in section_text
    assert "用户的直属领导叫陈姐" in section_text
    assert "【回答当前问题可参考】" in section_text
    assert "用户叫林小满" in section_text
    assert "用户28岁" in section_text
    assert "【安全 / 情绪背景】" in section_text
    assert "用户表达过强烈负面情绪" in section_text
    assert "【用户告诉过你的其他事情】" in section_text
    assert "回答关系、称呼、名字类事实追问时优先使用该组" in section_text
    assert "【安全 / 情绪背景】只用于把握语气和风险" in section_text
    assert section_text.index("【回答当前关系 / 名字问题优先参考】") < section_text.index("【回答当前问题可参考】")


@pytest.mark.asyncio
async def test_memory_section_separates_user_profile_context_from_answer_facts():
    from app.services.chat.prompt_builder import _build_memory_section
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    with patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)), \
         patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)):
        section = await _build_memory_section([
            ClassifiedMemory(
                id="user-age",
                text="用户28岁",
                relevance="medium",
                score=0.6,
                source="user",
                rank_reasons=["AI资料查询:用户同类资料"],
            ),
            ClassifiedMemory(
                id="ai-age",
                text="我今年24岁",
                relevance="strong",
                score=0.9,
                source="ai",
            ),
        ])

    assert section is not None
    section_text = _body(section)
    assert "【用户同类资料（仅用于避免重复追问）】" in section_text
    assert "不要把它当成你的资料或答案依据" in section_text
    assert "【你自己的相关经历 / 人设】" in section_text
    assert section_text.index("用户28岁") < section_text.index("我今年24岁")


@pytest.mark.asyncio
async def test_memory_section_declares_fact_precedence_over_history_and_l3():
    from app.services.chat.prompt_builder import _build_memory_section
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    with patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)), \
         patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)):
        section = await _build_memory_section([
            ClassifiedMemory(
                id="age",
                text="用户今年 28 岁",
                relevance="strong",
                score=0.92,
                source="user",
                rank_reasons=["保护槽:字面命中"],
            ),
        ])

    assert section is not None
    section_text = _body(section)
    assert "事实优先级" in section_text
    assert "当前用户消息明确说出新事实或纠正旧事实时, 以当前用户消息为准" in section_text
    assert "以下方当前问题相关记忆为准" in section_text
    assert "不要用历史对话或 L3 模糊记忆覆盖它" in section_text
    assert "若历史对话或 L3 与这些记忆冲突" in section_text
    assert "不要直接采用冲突值" in section_text


@pytest.mark.asyncio
async def test_system_prompt_skips_empty_placeholder_sections_on_weak_memory():
    from app.services.chat.prompt_builder import build_system_prompt

    async def _prompt_text(key: str, **_kwargs) -> str:
        from app.services.prompting.registry import PROMPT_DEFINITION_MAP

        overrides = {
            "chat.system_base": "像朋友一样回复。",
            "chat.consistency_rules": "。",
            "chat.response_instruction": "分{n}条消息回复，总共不超过{total}字。",
            "chat.anti_hallucination_hard_rule": "。",
        }
        if key in overrides:
            return overrides[key]
        definition = PROMPT_DEFINITION_MAP.get(key)
        return definition.default_text if definition else ""

    with (
        patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
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
async def test_system_prompt_includes_agent_age_from_identity():
    from app.services.chat.prompt_builder import build_system_prompt

    async def _prompt_text(key: str, **_kwargs) -> str:
        from app.services.prompting.registry import PROMPT_DEFINITION_MAP

        overrides = {
            "chat.system_base": "像朋友一样回复。",
            "chat.consistency_rules": "。",
            "chat.response_instruction": "分{n}条消息回复，总共不超过{total}字。",
            "chat.anti_hallucination_hard_rule": "。",
        }
        if key in overrides:
            return overrides[key]
        definition = PROMPT_DEFINITION_MAP.get(key)
        return definition.default_text if definition else ""

    with (
        patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
    ):
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="Hia", age=24, values={"gender": "female"}),
            memories=None,
            memory_relevance="weak",
            reply_count=1,
            reply_total=150,
        )

    assert "你的名字叫Hia" in prompt
    assert "你的年龄是24岁" in prompt


@pytest.mark.asyncio
async def test_system_prompt_keeps_empty_memory_anchor_when_hard_rule_active():
    from app.services.chat.prompt_builder import build_system_prompt

    async def _prompt_text(key: str, **_kwargs) -> str:
        return {
            "chat.system_base": "像朋友一样回复。",
            "chat.consistency_rules": "",
            "chat.response_instruction": "分{n}条消息回复，总共不超过{total}字。",
            "chat.anti_hallucination_hard_rule": "用户问记忆时必须检查记忆段。",
            "chat.memory_empty_anchor": "(本次没有联想到任何与当前话题相关的记忆)",
        }.get(key, "")

    with (
        patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
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


@pytest.mark.asyncio
async def test_l3_section_cannot_override_current_memory_facts():
    from app.services.chat.prompt_builder import build_system_prompt
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    async def _prompt_text(key: str, **_kwargs) -> str:
        from app.services.prompting import defaults

        from app.services.prompting.registry import PROMPT_DEFINITION_MAP

        overrides = {
            "chat.system_base": "像朋友一样回复。",
            "chat.consistency_rules": "不要说出与记忆矛盾的话。",
            "chat.response_instruction": "分{n}条消息回复，总共不超过{total}字。",
            "chat.anti_hallucination_hard_rule": "用户问记忆时必须检查记忆段。",
            "chat.memory_section_body": defaults.CHAT_MEMORY_SECTION_BODY_PROMPT,
            "chat.l3_memory_section": defaults.CHAT_L3_MEMORY_SECTION_PROMPT,
        }
        if key in overrides:
            return overrides[key]
        definition = PROMPT_DEFINITION_MAP.get(key)
        return definition.default_text if definition else ""

    with (
        patch("app.services.chat.prompt_builder.get_prompt_text_for_context", AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default", AsyncMock(side_effect=_prompt_text)),
    ):
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="Hillow", values={"gender": "female"}),
            memories=[
                ClassifiedMemory(
                    id="age",
                    text="用户今年 28 岁",
                    relevance="strong",
                    score=0.92,
                    source="user",
                    rank_reasons=["保护槽:字面命中"],
                ),
            ],
            l3_memories=["用户之前说自己 28 岁是说错了。"],
            memory_relevance="strong",
            reply_count=1,
            reply_total=150,
        )

    assert "用户今年 28 岁" in prompt
    assert "用户之前说自己 28 岁是说错了" in prompt
    assert "L3 是低置信历史线索" in prompt
    assert "不能覆盖「你记得的事情」里的当前事实" in prompt
    assert "若两者冲突，以当前记忆为准" in prompt


@pytest.mark.asyncio
async def test_disabled_prompt_section_completely_removed_from_system_prompt():
    """admin 停用某 section 模板 → 该段 (含标题) 从最终 system prompt 中彻底消失."""
    from app.services.chat.prompt_builder import build_system_prompt
    from app.services.prompting.store import PromptDisabledError

    async def _prompt_text_with_disabled(key: str, **_kwargs) -> str:
        if key in ("chat.anti_hallucination_hard_rule", "chat.time_context_section"):
            raise PromptDisabledError(key)
        return await _prompt_text(key)

    with patch(
        "app.services.chat.prompt_builder.get_prompt_text_for_context",
        AsyncMock(side_effect=_prompt_text_with_disabled),
    ):
        diagnostics = {}
        prompt = await build_system_prompt(
            agent=SimpleNamespace(name="Hillow", values={"gender": "female"}),
            memories=None,
            memory_relevance="weak",
            time_context="当前时间：2026年07月02日 10:00 周四",
            reply_count=1,
            reply_total=150,
            diagnostics=diagnostics,
        )

    assert "## 核心规则" in prompt
    assert "## 你的身份" in prompt
    # 停用的两段完整移除 (标题 + 内容 + 动态注入都不在)
    assert "反幻觉硬约束" not in prompt
    assert "## 时间" not in prompt
    assert "2026年07月02日" not in prompt
    assert "反幻觉硬约束" in diagnostics["empty_prompt_sections_removed"]
    assert "时间" in diagnostics["empty_prompt_sections_removed"]
