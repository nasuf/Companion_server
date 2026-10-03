"""ContextInjectionFilter: 把 ContextVar 的值挂到 LogRecord 上.

挂载方式: 直接 `setattr(record, key, value)` — AxiomHandler / JSON Formatter
序列化 LogRecord 时会把所有非内置 attr 当业务字段输出. 不需要改 Formatter
模板, 不需要改调用站点.

冲突策略: extra={...} 已传同名字段时不覆盖 — 调用方意图优先.
"""

from __future__ import annotations

import logging
import re

from app.observability.context import _VARS


_APNS_DEVICE_URL = re.compile(
    r"(https?://api(?:\.sandbox)?\.push\.apple\.com(?::[0-9]+)?/3/device/)[^\s\"']+",
    re.IGNORECASE,
)


def _redact_apns_request(record: logging.LogRecord) -> None:
    # Both console and Axiom's queue already apply this handler filter. Protect
    # legacy HTTPX request logs before formatting/queueing, without changing
    # client networking, environment proxies, logger levels or global filters.
    if record.name != "httpx" and not record.name.startswith("httpx."):
        return
    try:
        message = record.getMessage()
        masked = _APNS_DEVICE_URL.sub(r"\1[redacted]", message)
    except Exception:
        # Logging's fallback otherwise prints raw args on a formatting error.
        message, masked = None, "HTTPX log formatting unavailable"
    if masked != message:
        record.msg, record.args = masked, ()
        # An earlier handler may have cached an unsanitized formatted message.
        record.message = masked


class ContextInjectionFilter(logging.Filter):
    """快照 ContextVar 并保护 APNs 请求 URL（调用方 extra 优先）."""

    def filter(self, record: logging.LogRecord) -> bool:
        _redact_apns_request(record)
        for key, var in _VARS.items():
            if hasattr(record, key):
                continue  # 调用方 extra={key:...} 已传, 不覆盖
            value = var.get()
            if value is not None:
                setattr(record, key, value)
        return True
