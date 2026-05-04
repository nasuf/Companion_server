"""结构化可观测性 — ContextVar-based log enrichment + Axiom 远程日志.

公共 API:
- `bind_context(**kwargs)`: contextmanager, 绑定一段调用栈的 log 上下文
- `current_context()`: 读取当前所有非 None 字段, 测试/调试用
- `setup_axiom()`: 按 env 装 AxiomHandler (configure_logging 内调用)

支持的字段 (全部默认 None, Filter 见 None 自动 omit):
  conversation_id / workspace_id / agent_id / agent_name / user_id /
  username / request_id / trace_id

设计原则: 0 调用站点改动拿到 ID 字段 — 既有 `logger.info(...)` 自动被 Filter
注入字段; 高价值热点路径再加 `extra={"event": ..., ...}` 业务字段做查询维度.
"""

from app.observability.context import bind_context, current_context
from app.observability.axiom_setup import setup_axiom

__all__ = ["bind_context", "current_context", "setup_axiom"]
