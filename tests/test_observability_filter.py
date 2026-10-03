"""ContextInjectionFilter — 验证 LogRecord 字段注入行为.

确保:
- ContextVar 值挂到 LogRecord attrs (供 AxiomHandler 序列化为 JSON 字段)
- None 字段不挂 attr (避免 JSON null 噪声 + Axiom schema 列爆炸)
- 调用方 extra={...} 优先 — 不被 ContextVar 覆盖
"""

from __future__ import annotations

import logging
import logging.handlers
import queue

import httpx
import pytest

from app.observability.context import bind_context
from app.observability.log_filter import ContextInjectionFilter


def _make_record(**extra) -> logging.LogRecord:
    """造一条 LogRecord, 模拟 logger.info('msg', extra={...}) 路径."""
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="msg", args=None, exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_filter_injects_set_context_vars():
    f = ContextInjectionFilter()
    with bind_context(conversation_id="C-1", agent_name="Bob"):
        record = _make_record()
        assert f.filter(record) is True
        assert record.conversation_id == "C-1"  # type: ignore[attr-defined]
        assert record.agent_name == "Bob"  # type: ignore[attr-defined]


def test_filter_skips_unset_vars():
    """None 字段不应在 LogRecord 上出现 — Axiom schema 才能保持窄列."""
    f = ContextInjectionFilter()
    with bind_context(conversation_id="C-2"):  # 只 bind 一个
        record = _make_record()
        f.filter(record)
        assert hasattr(record, "conversation_id")
        # 没 bind 的字段不应 setattr
        assert not hasattr(record, "agent_name")
        assert not hasattr(record, "user_id")


def test_filter_does_not_override_extra():
    """调用方 extra={'agent_id': 'explicit'} 优先于 ContextVar."""
    f = ContextInjectionFilter()
    with bind_context(agent_id="from-ctx"):
        record = _make_record(agent_id="from-extra")
        f.filter(record)
        # extra 传的应保留, 不被 ContextVar 覆盖
        assert record.agent_id == "from-extra"  # type: ignore[attr-defined]


def test_filter_no_context_no_attrs():
    """没绑过 ContextVar 时, LogRecord 上不应多任何 attr."""
    f = ContextInjectionFilter()
    record = _make_record()
    f.filter(record)
    for attr in ("conversation_id", "agent_id", "agent_name", "user_id",
                 "workspace_id", "username", "request_id", "trace_id"):
        assert not hasattr(record, attr), f"unexpected {attr} on record"


@pytest.mark.parametrize("origin", ["https://api.push.apple.com", "https://api.sandbox.push.apple.com:443"])
def test_apns_token_is_removed_from_formatted_and_cached_structured_record(origin):
    token = "abcd" * 17
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1,
        'HTTP Request: %s %s "%s %d %s"',
        ("POST", httpx.URL(f"{origin}/3/device/{token}"), "HTTP/2", 200, "OK"), None)
    record.message = record.getMessage()  # Simulate an earlier formatter.
    with bind_context(conversation_id="synthetic-conversation"):
        assert ContextInjectionFilter().filter(record)
    assert "/3/device/[redacted]" in record.getMessage()
    assert token not in repr(record.__dict__)
    assert record.conversation_id == "synthetic-conversation"
    # Axiom applies the filter before enqueue and again in its listener.
    assert ContextInjectionFilter().filter(record)
    assert token not in repr(record.__dict__)


def test_apns_token_never_enters_the_actual_axiom_queue_handler_record():
    token = "abcd" * 17
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1,
        f"HTTP Request: POST https://api.push.apple.com/3/device/{token}?legacy=private", (), None)
    items = queue.Queue()
    handler = logging.handlers.QueueHandler(items)
    handler.addFilter(ContextInjectionFilter())
    handler.handle(record)
    queued = items.get_nowait()
    assert "/3/device/[redacted]" in queued.getMessage()
    assert token not in repr(queued.__dict__) and "legacy=private" not in repr(queued.__dict__)


def test_non_apns_http_logging_retains_its_original_args_and_message():
    args = ("GET", httpx.URL("https://synthetic.invalid/ordinary-request"), "HTTP/1.1", 200, "OK")
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1,
        'HTTP Request: %s %s "%s %d %s"', args, None)
    message = record.getMessage()
    assert ContextInjectionFilter().filter(record)
    assert record.getMessage() == message and record.args is args


def test_bad_httpx_formatting_cannot_emit_private_fallback_args():
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1, "%d",
        ("https://api.push.apple.com/3/device/private",), None)
    assert ContextInjectionFilter().filter(record)
    assert record.getMessage() == "HTTPX log formatting unavailable"
    assert "device/private" not in repr(record.__dict__)
