"""聊天请求级别的 LLM token 用量累加器 (基于 ContextVar).

工作流:
  orchestrator 入口  → start_session()  起累加 dict
  各 phase 调用 LLM → record(model, input, output)  wrapper 内部调
  orchestrator 出口 → flush_session()  拿汇总写一行 llm_usage 表

ContextVar 跟 langsmith trace_id 传播一样, asyncio task 内 share, 跨 task
自动隔离 (FastAPI 每个请求一个 task → 各请求互不干扰).

sub_intent_mode 直接复用父 session 不开新的 (orchestrator 自己控制 start
只在 sub_intent_mode=False 跑).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from typing import Callable, Literal, TypedDict

UsageScope = Literal[
    "chat", "post_process", "proactive", "agent_creation", "schedule_cron",
]

logger = logging.getLogger(__name__)


class _ModelUsage(TypedDict):
    input: int
    output: int


class UsageSummary(TypedDict):
    tokens_by_model: dict[str, _ModelUsage]
    input_tokens: int
    output_tokens: int
    call_count: int


_current: ContextVar[UsageSummary | None] = ContextVar("llm_usage_session", default=None)


def start_session() -> Token:
    """开新累加 session. 返回的 token 用于 flush_session 还原 ContextVar."""
    summary: UsageSummary = {
        "tokens_by_model": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "call_count": 0,
    }
    return _current.set(summary)


def has_session() -> bool:
    return _current.get() is not None


def record(model: str, input_tokens: int, output_tokens: int) -> None:
    """LLM wrapper 调用此函数累加. 没活跃 session 时 silently drop (单元测试 /
    后台异步任务 不在 chat session 里, 不该计入)."""
    summary = _current.get()
    if summary is None:
        return
    if not model:
        model = "unknown"
    bucket = summary["tokens_by_model"].setdefault(model, {"input": 0, "output": 0})
    bucket["input"] += int(input_tokens or 0)
    bucket["output"] += int(output_tokens or 0)
    summary["input_tokens"] += int(input_tokens or 0)
    summary["output_tokens"] += int(output_tokens or 0)
    summary["call_count"] += 1


def flush_session(token: Token) -> UsageSummary | None:
    """关闭 session, 返回累加结果. 只有 call_count > 0 才返回, 否则 None
    (调用方拿 None 不写 DB, 避免空行污染统计表)."""
    summary = _current.get()
    _current.reset(token)
    if summary is None or summary["call_count"] == 0:
        return None
    return summary


@asynccontextmanager
async def usage_session(
    *,
    scope: UsageScope,
    conversation_id: str | None,
    agent_id: str | None,
    user_id: str | None,
    trace_id_provider: Callable[[], str | None] | None = None,
):
    """统一封装 start_session → 业务 → flush_session → write_usage_row.

    `trace_id_provider` 是 callable 而非 str: tracer.trace_id 在 enter() 后
    才有值, 而调用方常常 `enter()` 在外面 / 业务在里面, callable 形式让
    flush 时再取最新值.
    """
    from app.services.llm.usage_repo import write_usage_row
    token = start_session()
    try:
        yield
    finally:
        summary = flush_session(token)
        if summary:
            await write_usage_row(
                summary=summary,
                conversation_id=conversation_id,
                agent_id=agent_id,
                user_id=user_id,
                trace_id=trace_id_provider() if trace_id_provider else None,
                scope=scope,
            )


@asynccontextmanager
async def traced_usage_session(
    *,
    name: str,
    scope: UsageScope,
    conversation_id: str | None,
    agent_id: str | None,
    user_id: str | None,
):
    """LangSmith trace + usage_session 组合, 给 yield 出来的 tracer 让调用方读 safe_trace_id."""
    from app.services.chat.tracing import LangSmithTracer
    tracer = LangSmithTracer(name, conversation_id or "").enter()
    try:
        async with usage_session(
            scope=scope, conversation_id=conversation_id,
            agent_id=agent_id, user_id=user_id,
            trace_id_provider=lambda: tracer.safe_trace_id,
        ):
            yield tracer
    finally:
        tracer.close()
