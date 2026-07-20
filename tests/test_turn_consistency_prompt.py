"""Prompt guard for aggregated user turns."""

from __future__ import annotations


def test_consistency_prompt_handles_same_turn_rewrites_generically():
    """同回合重复问题不能只依赖确定性 coalescing 关键词。"""
    from app.services.prompting.defaults import CONSISTENCY_RULES_PROMPT

    prompt = CONSISTENCY_RULES_PROMPT

    assert "同一个用户回合" in prompt
    assert "语义相近" in prompt
    assert "只回答一次" in prompt
    assert "逐行重复" in prompt


def test_consistency_prompt_enforces_preference_memory_adherence():
    """偏好类问题必须只用已注入记忆作答, 不许临时编 (记忆无视/编造 bug 治本)."""
    from app.services.prompting.defaults import CONSISTENCY_RULES_PROMPT

    prompt = CONSISTENCY_RULES_PROMPT
    assert "个人喜好" in prompt
    assert "只能用下方「你自己的相关经历 / 人设」里" in prompt
    assert "绝不临时编一个没列出的喜好" in prompt
