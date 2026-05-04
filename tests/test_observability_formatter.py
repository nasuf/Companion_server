"""HumanContextFormatter — 验证 console 内联 context 行为.

确保:
- 全空 context → 退化原 format, 不打 ` |  | ` 空标记
- 有 conversation_id / agent_name / user_id → 内联到 message 前
- conversation_id / user_id 这类长 ID 截 8 字符
- agent_name 不截 (人类可读名)
- workspace_id / username / request_id 不在 console 内联 (只走 Axiom)
- 不被 message 内容里的子串误触发 (之前用 replace 实现的 fragility 修复验证)
"""

from __future__ import annotations

import logging

from app.observability.log_formatter import HumanContextFormatter


def _make_record(message: str, **extras) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=message, args=None, exc_info=None,
    )
    for k, v in extras.items():
        setattr(record, k, v)
    return record


def _format(record: logging.LogRecord) -> str:
    fmt = HumanContextFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="2026-01-01",
    )
    return fmt.format(record)


def test_no_context_falls_back_to_plain_format():
    out = _format(_make_record("hello world"))
    assert out.endswith("| hello world")
    # 无空 context 标记
    assert " |  | " not in out


def test_inline_truncates_long_uuid_ids():
    record = _make_record(
        "user message arrived",
        conversation_id="conv_aaaaaaaa_bbbbbbbb",
        user_id="user_xxxxxx_yyyyyy",
    )
    out = _format(record)
    assert "conv=conv_aaa" in out  # 8-char truncate
    assert "user=user_xxx" in out
    # 完整 ID 不应出现在 console (字段全量仍在 LogRecord 给 Axiom)
    assert "conv_aaaaaaaa_bbbbbbbb" not in out


def test_inline_does_not_truncate_agent_name():
    record = _make_record("hi", agent_name="Alice")
    out = _format(record)
    assert "agent=Alice" in out


def test_message_substring_not_replaced():
    """旧 replace 实现 bug: logger name 'app.foo' + message 'foo' 会误替换. 修复后用
    formatMessage hook 不依赖字符串匹配."""
    record = _make_record("conv", conversation_id="conv_1234")
    out = _format(record)
    # message 内容是字面 "conv", 不应被 ctx 段误吃; ctx 应在 message 前
    assert out.endswith("conv=conv_123 | conv")


def test_omits_unset_inline_keys():
    """只设 conversation_id, agent_name 不应出现."""
    record = _make_record("test", conversation_id="C-1")
    out = _format(record)
    assert "conv=C-1" in out
    assert "agent=" not in out
    assert "user=" not in out


def test_workspace_id_username_not_inlined():
    """这些字段只走 Axiom JSON, console 不打."""
    record = _make_record(
        "test",
        workspace_id="ws_secret",
        username="alice",
        request_id="req_xyz",
    )
    out = _format(record)
    assert "workspace=" not in out
    assert "username=" not in out
    assert "request=" not in out
    # 但 LogRecord 上仍有这些 attrs (供 AxiomHandler 序列化)
    assert record.workspace_id == "ws_secret"  # type: ignore[attr-defined]
