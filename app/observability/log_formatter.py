"""HumanContextFormatter: console 人类可读格式 + 内联 ID.

格式: `time | LEVEL | logger | conv=abc12345 agent=Alice user=u_001 | message`

只内联 3 个最高频字段 (conversation_id / agent_name / user_id) 防 console
变冗长; 完整字段集 (workspace_id / username / request_id / trace_id) 仍走
LogRecord attrs, AxiomHandler 拿全集 JSON 化 — 远程查询不丢字段.

uuid 类 ID 截 8 字符防长 ID 把 console 撑爆 (Axiom 字段还是全量).
全部空时退化成原 format — admin HTTP 路径常无 ID, 不打 ` |  | ` 空标记.

实现: 重写 `formatMessage` 把 context 拼到 message 前. 不用 `replace()`
是因为 message 偶尔包含 logger name 子串会误替换 (e.g. logger 名 `app.foo.bar`
+ message 含 `bar` → super().format 后 replace 错位).
"""

from __future__ import annotations

import logging

# 只内联 console 显示的高频字段. 其他 ContextVar 字段仍在 LogRecord attrs 里
# 给 AxiomHandler 序列化, console 不展示.
_INLINE_KEYS: tuple[tuple[str, str, bool], ...] = (
    # (record_attr, console_label, is_uuid_truncate)
    ("conversation_id", "conv", True),
    ("agent_name", "agent", False),
    ("user_id", "user", True),
)


class HumanContextFormatter(logging.Formatter):
    """重写 formatMessage 给 record.message 前缀加 context 标签."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        # 注入 context 段 — 在调用 super 的 % 替换前先挂到 record 上
        ctx_parts: list[str] = []
        for attr, label, truncate in _INLINE_KEYS:
            val = getattr(record, attr, None)
            if not val:
                continue
            shown = val[:8] if truncate and isinstance(val, str) else val
            ctx_parts.append(f"{label}={shown}")
        # 把 context 段拼到 message 前缀, 避免改 format string 模板
        if ctx_parts:
            record.message = f"{' '.join(ctx_parts)} | {record.message}"
        return super().formatMessage(record)
