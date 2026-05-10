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
