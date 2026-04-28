"""LlmUsage 表的写入封装. 失败仅 log, 不抛异常 — 计费统计不应阻断聊天主流程."""

from __future__ import annotations

import logging

from prisma import Json

from app.db import db
from app.services.llm.pricing import estimate_cost_cny
from app.services.llm.usage_tracker import UsageSummary

logger = logging.getLogger(__name__)


def _total_cost_cny(tokens_by_model: dict) -> float:
    total = 0.0
    for model, t in tokens_by_model.items():
        total += estimate_cost_cny(model, int(t.get("input", 0)), int(t.get("output", 0)))
    return round(total, 6)


async def write_usage_row(
    *,
    summary: UsageSummary,
    conversation_id: str,
    agent_id: str | None,
    user_id: str | None,
    trace_id: str | None,
) -> None:
    cost = _total_cost_cny(summary["tokens_by_model"])
    try:
        await db.llmusage.create(
            data={
                "conversationId": conversation_id,
                "agentId": agent_id,
                "userId": user_id,
                "traceId": trace_id,
                "inputTokens": summary["input_tokens"],
                "outputTokens": summary["output_tokens"],
                "tokensByModel": Json(summary["tokens_by_model"]),
                "costCny": cost,
                "callCount": summary["call_count"],
            }
        )
    except Exception as e:
        logger.warning(
            f"[llm-usage] write failed conv={conversation_id[:8]} "
            f"agent={(agent_id or '?')[:8]} cost=¥{cost:.6f}: {e}"
        )
