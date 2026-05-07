"""Regression tests for memory relevance classifier.

历史 bug: relevance.py 内联 prompt 不读 registry → admin 改 memory.relevance
不生效, 且单条 user_message 输入对省略式追问 ("颜色呢？") 误判为「弱」漏召回.
现在: 走 render_prompt(memory.relevance) + 透传最近几轮上下文 (仅在 spec 简
洁模板基础上加 {context} 占位符, 其他措辞 / 输出格式保持 spec 原样)。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.memory.retrieval import relevance as relevance_mod


@pytest.fixture
def mock_render(monkeypatch):
    """Patch render_prompt so we can capture (key, params) and stub the LLM result."""
    mock = AsyncMock()
    monkeypatch.setattr(relevance_mod, "render_prompt", mock)
    return mock


async def test_uses_registry_key_not_inline_prompt(mock_render):
    """走 registry 而非内联 prompt → admin 面板的 memory.relevance 编辑生效。"""
    mock_render.return_value = "中"

    await relevance_mod.classify_memory_relevance("颜色呢？")

    args, _ = mock_render.call_args
    assert args[0] == "memory.relevance"


async def test_passes_context_to_resolve_ellipsis(mock_render):
    """省略式追问需要上下文解指代——必须把 context 透传给 LLM。"""
    mock_render.return_value = '{"level": "强", "enhanced_query": "颜色"}'
    ctx = "用户: 你有喜欢的水果吗？\nAI: 草莓、阳光玫瑰葡萄"

    result = await relevance_mod.classify_memory_relevance("颜色呢？", context=ctx)

    args, _ = mock_render.call_args
    params = args[1]
    assert params["message"] == "颜色呢？"
    assert params["context"] == ctx
    assert result.level == "strong"


async def test_empty_context_substituted_with_placeholder(mock_render):
    """无上下文 (首轮) 时填 (无), LLM 退化到仅看当前消息——保留 spec 单消息行为。"""
    mock_render.return_value = '{"level": "弱", "enhanced_query": ""}'

    await relevance_mod.classify_memory_relevance("你好")

    args, _ = mock_render.call_args
    assert args[1]["context"] == "(无)"


@pytest.mark.parametrize("raw,expected", [
    # JSON 标准输出
    ('{"level": "强", "enhanced_query": ""}', "strong"),
    ('{"level": "中", "enhanced_query": ""}', "medium"),
    ('{"level": "弱", "enhanced_query": ""}', "weak"),
    # JSON 含前后冗余文字 (LLM 多嘴) — 提取 {} 部分
    ('好的, {"level": "强", "enhanced_query": "颜色"}', "strong"),
    # 旧格式单字符 fallback (LLM 忽略 schema)
    ("强", "strong"),
    ("中", "medium"),
    ("弱", "weak"),
    ("「强」", "strong"),
    ("强相关", "strong"),
    # 完全无效 → 默认 medium
    ("不知道", "medium"),
    ("", "medium"),
    (None, "medium"),
])
async def test_level_mapping(mock_render, raw, expected):
    mock_render.return_value = raw
    result = await relevance_mod.classify_memory_relevance("test", context="ctx")
    assert result.level == expected


async def test_enhanced_query_extracted_from_json(mock_render):
    """Phase 2.4: enhanced_query 从 JSON 解出, 用于 hybrid retrieval."""
    mock_render.return_value = (
        '{"level": "强", "enhanced_query": "用户的妈妈最近住院的情况"}'
    )

    result = await relevance_mod.classify_memory_relevance(
        "那他怎样了?", context="用户: 我妈妈住院了 / AI: 怎么了?",
    )

    assert result.level == "strong"
    assert result.enhanced_query == "用户的妈妈最近住院的情况"


async def test_enhanced_query_capped_at_50_chars(mock_render):
    """LLM 啰嗦 → cap 到 50 字, 避免长 query 稀释 embedding 信号."""
    long_query = "x" * 100
    mock_render.return_value = (
        f'{{"level": "强", "enhanced_query": "{long_query}"}}'
    )

    result = await relevance_mod.classify_memory_relevance("test")
    assert len(result.enhanced_query) <= 50


async def test_old_single_char_format_no_enhanced_query(mock_render):
    """fallback 路径 (旧单字符) → enhanced_query 必空."""
    mock_render.return_value = "强"

    result = await relevance_mod.classify_memory_relevance("test")
    assert result.level == "strong"
    assert result.enhanced_query == ""


async def test_llm_failure_defaults_to_medium(mock_render):
    """LLM 异常 → 不让上层崩, 退回 medium 走中等召回。"""
    mock_render.side_effect = RuntimeError("LLM down")

    result = await relevance_mod.classify_memory_relevance("test")

    assert result.level == "medium"
    assert result.enhanced_query == ""
