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
    "offline", "music",
]

logger = logging.getLogger(__name__)


class _ModelUsage(TypedDict):
    input: int
    output: int
    # 命中 provider prefix cache 的 input tokens (计费 ~0.1x-0.4x 原价).
    # input 字段是总量, cached_input 是其中命中的部分 (cached_input <= input).
    cached_input: int


class UsageSummary(TypedDict):
    tokens_by_model: dict[str, _ModelUsage]
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    call_count: int
    latency_ms_total: int
    latency_count: int
    failure_count: int
    fallback_count: int
    circuit_open_count: int


_current: ContextVar[UsageSummary | None] = ContextVar("llm_usage_session", default=None)


def start_session() -> Token:
    """开新累加 session. 返回的 token 用于 flush_session 还原 ContextVar."""
    summary: UsageSummary = {
        "tokens_by_model": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "call_count": 0,
        "latency_ms_total": 0,
        "latency_count": 0,
        "failure_count": 0,
        "fallback_count": 0,
        "circuit_open_count": 0,
    }
    return _current.set(summary)


def has_session() -> bool:
    return _current.get() is not None


def record(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> None:
    """LLM wrapper 调用此函数累加. 没活跃 session 时 silently drop (单元测试 /
    后台异步任务 不在 chat session 里, 不该计入).

    cached_input_tokens: input 中命中 provider prefix cache 的部分 (DeepSeek
    prompt_cache_hit_tokens / qwen cached_tokens, LangChain 归一为
    input_token_details.cache_read). 驱动后台缓存命中率监控.
    """
    summary = _current.get()
    if summary is None:
        return
    if not model:
        model = "unknown"
    bucket = summary["tokens_by_model"].setdefault(
        model, {"input": 0, "output": 0, "cached_input": 0},
    )
    bucket["input"] += int(input_tokens or 0)
    bucket["output"] += int(output_tokens or 0)
    # 旧 session dict 兼容 (理论上不存在, setdefault 兜底)
    bucket["cached_input"] = bucket.get("cached_input", 0) + int(cached_input_tokens or 0)
    summary["input_tokens"] += int(input_tokens or 0)
    summary["output_tokens"] += int(output_tokens or 0)
    summary["cached_input_tokens"] += int(cached_input_tokens or 0)
    summary["call_count"] += 1


def record_runtime_event(
    *,
    result: str,
    latency_ms: int | None = None,
) -> None:
    """Record session-level LLM runtime signals.

    Token usage and runtime health are both written to `llm_usage`. This keeps
    dashboard queries cheap and avoids a separate hot-path insert per LLM call.
    """
    summary = _current.get()
    if summary is None:
        return
    if latency_ms is not None:
        summary["latency_ms_total"] += int(latency_ms)
        summary["latency_count"] += 1
    if result == "fallback":
        summary["fallback_count"] += 1
    elif result == "circuit_open":
        summary["circuit_open_count"] += 1
        summary["failure_count"] += 1
    elif result != "ok":
        summary["failure_count"] += 1


def flush_session(token: Token) -> UsageSummary | None:
    """关闭 session, 返回累加结果.

    有 token usage 或 runtime health signal 时才写 DB; 这样全失败但被熔断/
    fallback 的请求也能进入运营统计。
    """
    summary = _current.get()
    _current.reset(token)
    if summary is None:
        return None
    has_runtime_signal = any(
        summary[key] > 0
        for key in ("latency_count", "failure_count", "fallback_count", "circuit_open_count")
    )
    if summary["call_count"] == 0 and not has_runtime_signal:
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
    from app.services.prompting.trace_components import (
        reset_prompt_render_trace,
        start_prompt_render_trace,
    )
    tracer = LangSmithTracer(name, conversation_id or "").enter()
    prompt_trace_token = start_prompt_render_trace()
    try:
        async with usage_session(
            scope=scope, conversation_id=conversation_id,
            agent_id=agent_id, user_id=user_id,
            trace_id_provider=lambda: tracer.safe_trace_id,
        ):
            yield tracer
    finally:
        reset_prompt_render_trace(prompt_trace_token)
        tracer.close()
