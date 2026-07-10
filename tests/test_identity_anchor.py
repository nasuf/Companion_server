"""Regression: persistent identity anchor in the chat prompt + greeting.

Root cause (2026-07 production audit, agent 小伴): after core_memory permanent
injection was removed (spec §3 retrieval-only), the chat hot path carried no
职业/身份 grounding, so the LLM invented a persona ("在便利店打工/待业") when asked
its job. The fix re-anchors the most spoofable identity facts (职业/现居地) as a
stable, per-agent block inside the "你的身份" section, and makes the first
greeting reliably state the profession.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services.prompting import defaults as d


def _fake_get_factory():
    tpl_by_key = {
        "chat.personality_section": d.CHAT_PERSONALITY_SECTION_PROMPT,
        "chat.style_base_rule": d.CHAT_STYLE_BASE_RULE_PROMPT,
        "chat.style_closing_rule": d.CHAT_STYLE_CLOSING_RULE_PROMPT,
    }

    async def fake_get(key, **kwargs):
        return tpl_by_key[key]

    return fake_get


class _Agent:
    def __init__(self, *, occupation=None, city=None):
        self.name = "小伴"
        self.age = 22
        self.values = {"gender": "female", "personality": {}}
        self.occupation = occupation
        self.city = city


@pytest.mark.asyncio
async def test_identity_anchor_injects_occupation_and_city():
    from app.services.chat.prompt_builder import _build_personality_section

    agent = _Agent(occupation="伴生公司客服员", city="云南省普洱市思茅区南屏镇凤凰路社区")
    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        side_effect=_fake_get_factory(),
    ):
        section = await _build_personality_section(agent)

    assert section is not None
    body = section.body
    # The occupation must be a hard, stated fact — this is what stops the LLM
    # from answering "便利店/待业" when asked.
    assert "你的职业是伴生公司客服员" in body
    assert "现居云南省普洱市思茅区南屏镇凤凰路社区" in body
    # And it must be framed as a non-negotiable identity constraint.
    assert "绝不能凭空编造成与此不符" in body


@pytest.mark.asyncio
async def test_identity_anchor_absent_when_no_occupation_or_city():
    """Agents without occupation/city render cleanly (no dangling anchor text)."""
    from app.services.chat.prompt_builder import _build_personality_section

    agent = _Agent(occupation=None, city=None)
    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        side_effect=_fake_get_factory(),
    ):
        section = await _build_personality_section(agent)

    assert section is not None
    body = section.body
    assert "你的职业是" not in body
    assert "这是你真实的身份设定" not in body
    # Name still present — section is otherwise intact.
    assert "小伴" in body


@pytest.mark.asyncio
async def test_identity_anchor_is_cache_stable_per_agent():
    """Same agent → byte-identical section (must stay in the stable prefix)."""
    from app.services.chat.prompt_builder import _build_personality_section

    agent = _Agent(occupation="伴生公司客服员", city="普洱")
    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        side_effect=_fake_get_factory(),
    ):
        first = await _build_personality_section(agent)
        second = await _build_personality_section(agent)
    assert first.body == second.body


def test_first_greeting_prompt_requires_stating_profession():
    """The first greeting must actively surface the profession, not treat it as
    reference-only (previously '（只参考不用刻意提及）' suppressed it)."""
    tpl = d.PROACTIVE_FIRST_GREETING_PROMPT
    assert "{occupation}" in tpl
    # Requirement now explicitly asks to state what the persona does for work.
    assert "顺带说出自己是做什么的" in tpl
    # The blanket "don't mention" note over the whole 参考信息 block is gone.
    assert "（只参考不用刻意提及）" not in tpl
