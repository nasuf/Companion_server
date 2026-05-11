"""Phase 6: prompt 大体检 验收 + token 实测.

改动:
- 删 graph_context section (~150-200 tokens 冗余 + 抽象列表幻觉源)
- 删 relational_context 注入 (跟 SYSTEM_BASE 重叠, 信号仍用于路由)
- SYSTEM_BASE 去重 (3 句"不是 AI" → 1 句, 防"粉色大象"效应)
- 删 PERSONALITY_RULES 拼接 (4 句全跟 SYSTEM_BASE/RESPONSE_INSTRUCTION 重叠)
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _fake_agent(name="Nina"):
    return SimpleNamespace(
        id="a1", name=name, values={"gender": "female"},
        personalityScores={
            "lively": 50, "rational": 50, "emotional": 50,
            "planned": 50, "spontaneous": 50, "creative": 50, "humorous": 50,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Section 删除验收
# ═══════════════════════════════════════════════════════════════════


def test_build_system_prompt_no_graph_relational_section_calls():
    """build_system_prompt 源码中不再调 _build_graph_context_section / _build_relational_context_section."""
    import inspect
    from app.services.chat.prompt_builder import build_system_prompt

    src = inspect.getsource(build_system_prompt)
    assert "_build_graph_context_section" not in src
    assert "_build_relational_context_section" not in src
    # section 名字也不该出现在 sections.append (除注释)
    # (注释行可能有"关系上下文" 字眼, 但必须不在调用 sections.append 的语句里)
    for line in src.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # 非注释行不该 append "关系上下文" 或 "关系回应重点"
        assert not (
            "sections.append" in line and ("关系上下文" in line or "关系回应重点" in line)
        ), f"prompt_builder 代码行仍 append 删除的 section: {line}"


def test_build_system_prompt_signature_drops_relational_graph():
    """build_system_prompt 签名不再含 relational_context / graph_context 必需参数,
    历史调用方走 **_deprecated_kwargs 兜底 (内部静默丢弃)."""
    import inspect
    from app.services.chat.prompt_builder import build_system_prompt

    sig = inspect.signature(build_system_prompt)
    params = sig.parameters
    # 显式参数已删
    assert "relational_context" not in params
    assert "graph_context" not in params
    # **kwargs 兜底必须存在 (向后兼容)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    assert has_var_kw, "build_system_prompt 应有 **kwargs 兜底, 防老 caller 崩"


def test_no_section_helpers_for_graph_relational():
    """_build_graph_context_section / _build_relational_context_section 已删除."""
    from app.services.chat import prompt_builder

    assert not hasattr(prompt_builder, "_build_graph_context_section")
    assert not hasattr(prompt_builder, "_build_relational_context_section")


def test_system_base_dedup():
    """SYSTEM_BASE 不再 3 次否定 'AI'."""
    from app.services.prompting.defaults import SYSTEM_BASE_PROMPT

    # "AI" 仍可能出现 1 次 (e.g. "不是 AI 助手"), 但不超过 1 次
    ai_count = SYSTEM_BASE_PROMPT.count("AI")
    assert ai_count <= 1, (
        f"SYSTEM_BASE 'AI' 字面应 ≤1 次 (Phase 6 去重); got {ai_count}"
    )
    # 不再有"绝对不是 Qwen / ChatGPT / 公司开发的产品" 的具体品牌罗列
    assert "Qwen" not in SYSTEM_BASE_PROMPT
    assert "ChatGPT" not in SYSTEM_BASE_PROMPT
    assert "人工智能" not in SYSTEM_BASE_PROMPT  # 同样减少否定 token 数


def test_personality_rules_text_not_in_section_template():
    """_build_personality_section 源码中, body 拼接 template 不再含
    {personality_rules} (跟另一 test 互补 — 这个查 string template, 那个查 await)."""
    import inspect
    from app.services.chat.prompt_builder import _build_personality_section

    src = inspect.getsource(_build_personality_section)
    # body 拼接行不该有 personality_rules 占位
    assert "{personality_rules}" not in src


# ═══════════════════════════════════════════════════════════════════
# Token 实测 — 验证 Phase 6 真的省 token
# ═══════════════════════════════════════════════════════════════════


def _estimate_tokens(text: str) -> int:
    """中文 1.5 token/char (跟 context_selector.estimate_tokens 一致)."""
    return int(len(text) * 1.5)


def test_static_prefix_token_savings():
    """实测 SYSTEM_BASE + PERSONALITY 静态段 token 数. Phase 6 应有可观节省."""
    from app.services.prompting.defaults import (
        SYSTEM_BASE_PROMPT, PERSONALITY_RULES_PROMPT,
    )

    system_base_tokens = _estimate_tokens(SYSTEM_BASE_PROMPT)
    rules_tokens = _estimate_tokens(PERSONALITY_RULES_PROMPT)

    # SYSTEM_BASE 去重后应 < 200 chars × 1.5 = 300 tokens
    # 历史 ~250 chars → 现在 ~140 chars (估)
    assert system_base_tokens < 250, (
        f"SYSTEM_BASE 应 < 250 tokens (Phase 6 去重); got {system_base_tokens}"
    )

    # PERSONALITY_RULES 仍存在但不被拼接 (DEPRECATED)
    # 仅断言它有内容 — 删除留作未来工作
    assert rules_tokens > 0


def test_personality_rules_text_dropped_from_static_prompts():
    """PERSONALITY_RULES prompt 即便 registry 还存在 (兼容期), prompt_builder 不再拼接.

    这跟 test_personality_rules_no_longer_appended 的 dynamic 测试互补:
    sync 直接验静态文本不在 default _build_personality_section 拼接逻辑里.
    """
    import inspect
    from app.services.chat import prompt_builder

    src = inspect.getsource(prompt_builder._build_personality_section)
    # 关键: 不再调 get_prompt_text + 不再用 personality_rules 变量拼接 body
    assert 'get_prompt_text("chat.personality_rules")' not in src, (
        "_build_personality_section 不该再读 chat.personality_rules"
    )
    # body 拼接行不该含 {personality_rules}
    assert "{personality_rules}" not in src


def _is_string_literal_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_string_literal_expr(node.left) and _is_string_literal_expr(node.right)
    return False


def _prompt_assignment_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in target.elts:
            names.extend(_prompt_assignment_names(item))
        return names
    return []


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_runtime_code_has_no_inline_llm_prompt_literals():
    """LLM 运行时 prompt 不允许散落在业务代码中。

    新 prompt 必须先放进 app/services/prompting/defaults.py 并注册到 registry/store,
    业务代码只能通过 get_prompt_text/render_prompt 取模板后填充。
    """
    app_root = Path(__file__).resolve().parents[1] / "app"
    defaults_path = app_root / "services" / "prompting" / "defaults.py"
    llm_call_names = {"invoke_text", "invoke_json", "invoke_json_with_usage"}
    violations: list[str] = []

    for path in app_root.rglob("*.py"):
        if path == defaults_path:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(app_root)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if value is None or not _is_string_literal_expr(value):
                    continue
                for target in targets:
                    for name in _prompt_assignment_names(target):
                        if name == "prompt" or name.endswith("_prompt"):
                            violations.append(f"{rel}:{node.lineno} inline prompt literal assigned to {name}")

            if isinstance(node, ast.Call) and _call_name(node.func) in llm_call_names:
                # invoke_text(model, prompt) / invoke_json(model, prompt): 第二个位置参数是 prompt.
                if len(node.args) >= 2 and _is_string_literal_expr(node.args[1]):
                    violations.append(f"{rel}:{node.lineno} inline prompt literal passed to {_call_name(node.func)}")

    assert violations == []
