"""ContextInjectionFilter: 把 ContextVar 的值挂到 LogRecord 上.

挂载方式: 直接 `setattr(record, key, value)` — AxiomHandler / JSON Formatter
序列化 LogRecord 时会把所有非内置 attr 当业务字段输出. 不需要改 Formatter
模板, 不需要改调用站点.

冲突策略: extra={...} 已传同名字段时不覆盖 — 调用方意图优先.
"""

from __future__ import annotations

import logging

from app.observability.context import _VARS


class ContextInjectionFilter(logging.Filter):
    """从 ContextVar 读字段挂到 LogRecord 上 (调用方 extra 优先)."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, var in _VARS.items():
            if hasattr(record, key):
                continue  # 调用方 extra={key:...} 已传, 不覆盖
            value = var.get()
            if value is not None:
                setattr(record, key, value)
        return True
