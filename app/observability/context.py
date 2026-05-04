"""ContextVar 定义 + bind_context() context manager.

用法 (典型 — WebSocket 入口):

    with bind_context(
        conversation_id=conversation_id,
        agent_id=agent.id,
        agent_name=agent.name,
        user_id=user_id,
    ):
        # 这里以及派生的所有 asyncio.create_task / fire_background
        # 自动继承上下文; 该作用域内的 logger.info(...) 会被 Filter 注入字段
        await _handle_message(...)

设计取舍:
- 默认 None — Filter 见 None 自动跳过, JSON 输出干净, console 不打印空字段
- Token 栈 + LIFO reset — 嵌套绑定安全 (内层 override 外层, exit 后恢复)
- 未知 key 静默忽略 — 调用方传错字段名不炸, 防小拼写错误中断主流程
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

# 默认 None — Filter 见 None 自动 omit, console 不打印, JSON 干净
conversation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "conversation_id", default=None
)
workspace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "workspace_id", default=None
)
agent_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_id", default=None
)
agent_name_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_name", default=None
)
user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "user_id", default=None
)
username_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "username", default=None
)
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)

# 注册表: key (Axiom 字段名) → ContextVar.
# Filter 和 Formatter 都遍历这个 dict, 加新字段只改这一处.
_VARS: dict[str, contextvars.ContextVar[str | None]] = {
    "conversation_id": conversation_id_var,
    "workspace_id": workspace_id_var,
    "agent_id": agent_id_var,
    "agent_name": agent_name_var,
    "user_id": user_id_var,
    "username": username_var,
    "request_id": request_id_var,
    "trace_id": trace_id_var,
}


@contextmanager
def bind_context(**kwargs: str | None) -> Iterator[None]:
    """绑定一段调用栈的 log 上下文. 自动随 asyncio task 传播.

    传 None = 明确清空 (覆盖外层非 None 值).
    不传 = 保留外层值 (典型: 子作用域只想换 1-2 个字段).
    嵌套调用按 LIFO 恢复 — 内层 with 退出后 ContextVar 还原到外层值.
    未知 key 静默忽略 — 防小拼写错误中断主流程.
    """
    tokens: list[tuple[contextvars.ContextVar[str | None], contextvars.Token]] = []
    for key, value in kwargs.items():
        var = _VARS.get(key)
        if var is None:
            continue
        tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        # LIFO reset — 跟 set 顺序逆序还原
        for var, token in reversed(tokens):
            var.reset(token)


def current_context() -> dict[str, str]:
    """快照所有非 None 的字段. Filter / 测试 / 调试用."""
    snapshot: dict[str, str] = {}
    for k, v in _VARS.items():
        value = v.get()
        if value is not None:
            snapshot[k] = value
    return snapshot
