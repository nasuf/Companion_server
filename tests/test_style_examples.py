"""Phase C2 回归: MBTI 四象限说话示例 (few-shot).

性格靠示例定型比靠规则列表有效 (MaiBot 对比结论). 示例按 (E/I, F/T) 四象限
静态映射, per-agent 稳定 (不破坏 prompt cache 前缀).
"""

from __future__ import annotations

import pytest

from app.services.style import _STYLE_EXAMPLES, generate_style_examples


def _mbti(e: int, f: int) -> dict:
    # signal() 经 _LETTER_MAP 读 4 轴百分比字段: EI/NS/TF/JP (E/N/T/J 为正向).
    # F 强度 = 1 - TF/100, 故 TF = 100 - f.
    return {"EI": e, "NS": 50, "TF": 100 - f, "JP": 50}


class TestQuadrantSelection:
    def test_ef_quadrant(self):
        out = generate_style_examples(_mbti(e=80, f=80))
        assert _STYLE_EXAMPLES["EF"][0][1] in out

    def test_et_quadrant(self):
        out = generate_style_examples(_mbti(e=80, f=20))
        assert _STYLE_EXAMPLES["ET"][0][1] in out

    def test_if_quadrant(self):
        out = generate_style_examples(_mbti(e=20, f=80))
        assert _STYLE_EXAMPLES["IF"][0][1] in out

    def test_it_quadrant(self):
        out = generate_style_examples(_mbti(e=20, f=20))
        assert _STYLE_EXAMPLES["IT"][0][1] in out

    def test_none_mbti_does_not_crash(self):
        out = generate_style_examples(None)
        assert "对方说" in out


class TestExampleContract:
    def test_header_warns_against_copying(self):
        """few-shot 必须声明"不要照抄" — 否则 LLM 会原句复读示例."""
        out = generate_style_examples(_mbti(e=80, f=80))
        assert "不要照抄" in out

    def test_each_quadrant_has_three_scenarios(self):
        for quadrant, examples in _STYLE_EXAMPLES.items():
            assert len(examples) == 3, f"{quadrant} 应覆盖 3 个场景"

    def test_examples_are_wechat_length(self):
        """示例回复本身必须符合"微信短消息"体感 (否则示范了长篇大论)."""
        for examples in _STYLE_EXAMPLES.values():
            for _user, reply in examples:
                assert len(reply) <= 40, f"示例过长: {reply}"

    def test_deterministic_per_agent(self):
        """cache 契约: 同一 MBTI 两次生成完全一致 (per-agent 稳定段)."""
        m = _mbti(e=80, f=30)
        assert generate_style_examples(m) == generate_style_examples(m)


@pytest.mark.asyncio
async def test_personality_section_includes_examples():
    """端到端: personality section 渲染包含示例块."""
    from unittest.mock import patch

    from app.services.chat.prompt_builder import _build_personality_section
    from app.services.prompting import defaults as d

    tpl_by_key = {
        "chat.personality_section": d.CHAT_PERSONALITY_SECTION_PROMPT,
        "chat.style_base_rule": d.CHAT_STYLE_BASE_RULE_PROMPT,
        "chat.style_closing_rule": d.CHAT_STYLE_CLOSING_RULE_PROMPT,
    }

    async def fake_get(key, **kwargs):
        return tpl_by_key[key]

    class FakeAgent:
        name = "小满"
        age = 22
        values = {"gender": "female", "personality": {}}

    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        side_effect=fake_get,
    ):
        section = await _build_personality_section(FakeAgent())

    assert section is not None
    assert "你说话大概是这种感觉" in section.body
    assert "对方说" in section.body
