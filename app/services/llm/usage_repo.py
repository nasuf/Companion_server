"""LlmUsage 表的写入封装. 失败仅 log, 不抛异常 — 计费统计不应阻断聊天主流程."""

from __future__ import annotations

import asyncio
import logging

from prisma import Json

from app.db import db
from app.services.llm.pricing import estimate_cost_cny, is_known_model
from app.services.llm.usage_tracker import UsageScope, UsageSummary

logger = logging.getLogger(__name__)
_pending_writes: set[asyncio.Task] = set()
WRITE_TIMEOUT_SECONDS = 5.0


async def persist_usage(**kwargs) -> None:
    """Bounded, cancellation-independent write shared by all session owners.

    Shield preserves the caller's cancellation while retaining the actual task
    until its five-second deadline. No retries: an ambiguous commit must not
    double-count a charged call. Neither provider nor SQL error text is logged.
    """
    async def write() -> None:
        try:
            async with asyncio.timeout(WRITE_TIMEOUT_SECONDS):
                await write_usage_row(**kwargs)
        except Exception as exc:
            logger.warning("[llm-usage] write failed scope=%s trace=%s type=%s",
                           kwargs.get("scope", "chat"), (kwargs.get("trace_id") or "")[:8],
                           type(exc).__name__)

    task = asyncio.create_task(write(), name="persist-llm-usage")
    _pending_writes.add(task)
    task.add_done_callback(_pending_writes.discard)
    await asyncio.shield(task)


async def drain_usage_writes() -> None:
    """Finish already-owned metering writes before the DB pool closes."""
    pending = set(_pending_writes)
    if not pending:
        return
    done, remaining = await asyncio.wait(pending, timeout=WRITE_TIMEOUT_SECONDS + .1)
    for task in remaining:
        task.cancel()
    if remaining:
        await asyncio.gather(*remaining, return_exceptions=True)
        logger.warning("[llm-usage] shutdown writes timed out count=%d", len(remaining))


def _total_cost_cny(tokens_by_model: dict) -> float:
    total = 0.0
    for model, t in tokens_by_model.items():
        total += estimate_cost_cny(
            model,
            int(t.get("input", 0)),
            int(t.get("output", 0)),
            cached_input_tokens=int(t.get("cached_input", 0)),
        )
    return round(total, 6)


async def aggregate_usage_by_trace_ids(trace_ids: list[str]) -> dict[str, dict]:
    """按 trace_id 汇总 llm_usage 行 → {trace_id: usage_dict}.

    同一 trace 可能有多行 (chat 热路径 + post_process 后台), 求和后作为
    "这一轮回复的完整成本". cached_input 存在 tokensByModel JSON 里, 逐行
    展开累加. 查询失败返回空 dict — 用量展示是装饰性信息, 不该影响调用方.
    """
    ids = [t for t in trace_ids if t]
    if not ids:
        return {}
    try:
        rows = await db.llmusage.find_many(where={"traceId": {"in": ids}})
    except Exception as e:
        logger.debug(f"[llm-usage] aggregate query failed: {e}")
        return {}
    agg: dict[str, dict] = {}
    for row in rows:
        entry = agg.setdefault(row.traceId, {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cost_cny": 0.0,
            "call_count": 0,
            "unmetered_calls": 0,
        })
        entry["input_tokens"] += int(row.inputTokens or 0)
        entry["output_tokens"] += int(row.outputTokens or 0)
        entry["cost_cny"] += float(row.costCny or 0.0)
        entry["call_count"] += int(row.callCount or 0)
        tokens_by_model = row.tokensByModel if isinstance(row.tokensByModel, dict) else {}
        entry["unmetered_calls"] += sum(
            int(bucket.get("unmetered_calls", 0) or 0) for bucket in tokens_by_model.values()
            if isinstance(bucket, dict))
        entry["cached_input_tokens"] += sum(
            int(bucket.get("cached_input", 0) or 0)
            for bucket in tokens_by_model.values()
            if isinstance(bucket, dict)
        )
    for entry in agg.values():
        entry["cost_cny"] = round(entry["cost_cny"], 6)
    return agg


async def write_usage_row(
    *,
    summary: UsageSummary,
    conversation_id: str | None,
    agent_id: str | None,
    user_id: str | None,
    trace_id: str | None,
    scope: UsageScope = "chat",
) -> None:
    cost = _total_cost_cny(summary["tokens_by_model"])
    # Record whether each model had a price when its fee was computed. A model
    # later removed from the registry must not turn this recorded fee unknown,
    # and one unpriced model in a multi-model row must not hide behind the rest.
    tokens_by_model = {
        model: {**bucket, "priced": is_known_model(model)}
        for model, bucket in summary["tokens_by_model"].items()
    }
    try:
        await db.llmusage.create(
            data={
                "scope": scope,
                "conversationId": conversation_id,
                "agentId": agent_id,
                "userId": user_id,
                "traceId": trace_id,
                "inputTokens": summary["input_tokens"],
                "outputTokens": summary["output_tokens"],
                "tokensByModel": Json(tokens_by_model),
                "costCny": cost,
                "callCount": summary["call_count"],
                "webSearchCalls": summary.get("web_search_calls", 0),
                "latencyMsTotal": summary["latency_ms_total"],
                "latencyCount": summary["latency_count"],
                "failureCount": summary["failure_count"],
                "fallbackCount": summary["fallback_count"],
                "circuitOpenCount": summary["circuit_open_count"],
            }
        )
    except Exception as e:
        logger.warning(
            "[llm-usage] write failed scope=%s trace=%s type=%s",
            scope, (trace_id or "")[:8], type(e).__name__,
        )
