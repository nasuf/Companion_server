"""trace 本地镜像写入 + 读取.

用户点 Trace 按钮触发 share + load_public_trace 后, 把 enriched detail 缓存
到 message_traces 表; 同消息再次打开省去 LangSmith API 往返 (网络抽风 / quota /
秒级打开). 失败兜底: 本地未命中 → 调用方 fallback 到 LangSmith 公开 API.
"""

from __future__ import annotations

import logging
from typing import Any

from prisma import Json

from app.db import db

logger = logging.getLogger(__name__)


async def write_trace_mirror(
    *, detail: dict[str, Any], message_id: str | None = None,
) -> bool:
    """把 enriched trace detail 写入 message_traces 表. 失败仅 log warning,
    不抛异常 (调用方不必再处理). detail 由 caller 通过 load_public_trace 取得;
    集中调一次 load 既给客户端响应也给 mirror, 避免双拉.
    """
    trace = detail.get("trace") or {}
    steps = detail.get("steps") or []
    trace_id = str(trace.get("trace_id") or trace.get("root_id") or "")
    if not trace_id:
        logger.warning("[trace_mirror] no trace_id in detail")
        return False
    conv_id = str(trace.get("conversation_id") or "")
    if not conv_id:
        logger.warning(f"[trace_mirror] no conversation_id for trace {trace_id}")
        return False

    payload = {
        "traceId": trace_id,
        "conversationId": conv_id,
        "messageId": message_id,
        "rootMessage": trace.get("message"),
        "totalDurationMs": trace.get("duration_ms"),
        "totalTokens": trace.get("total_tokens"),
        "llmStepCount": trace.get("llm_step_count"),
        "stepsJson": Json(steps),
        "summaryJson": Json(trace),
        "shareStatus": "shared",
        "shareUrl": trace.get("external_url"),
    }
    try:
        await db.messagetrace.upsert(
            where={"traceId": trace_id},
            data={
                "create": payload,
                "update": {k: v for k, v in payload.items() if k != "traceId"},
            },
        )
        logger.info(
            f"[trace_mirror] wrote trace={trace_id[:8]} conv={conv_id[:8]} "
            f"steps={len(steps)} duration={trace.get('duration_ms')}ms"
        )
        return True
    except Exception as e:
        logger.warning(f"[trace_mirror] db upsert failed for {trace_id}: {e}")
        return False


async def get_trace_mirror(trace_id: str) -> dict[str, Any] | None:
    """按 trace_id 直查. 返回跟 load_public_trace 同 shape 的 dict, 没找到返 None."""
    if not trace_id:
        return None
    try:
        row = await db.messagetrace.find_unique(where={"traceId": trace_id})
    except Exception as e:
        logger.warning(f"[trace_mirror] db read failed for trace {trace_id}: {e}")
        return None
    if not row:
        return None
    return _row_to_detail(row)


async def get_trace_mirror_by_message(message_id: str) -> dict[str, Any] | None:
    """按 message_id 反查最近一条 mirror. 一条消息可能对应一个 trace
    (主调用 trace_id 写到 message.metadata.trace_id), 取 createdAt 最新."""
    if not message_id:
        return None
    try:
        rows = await db.messagetrace.find_many(
            where={"messageId": message_id},
            order={"createdAt": "desc"},
            take=1,
        )
    except Exception as e:
        logger.warning(f"[trace_mirror] db read by msg failed {message_id}: {e}")
        return None
    if not rows:
        return None
    return _row_to_detail(rows[0])


def _row_to_detail(row: Any) -> dict[str, Any]:
    """MessageTrace row → load_public_trace 同 shape dict.

    Json 列在 prisma client 里返回 dict/list (不是 JSON string), 直接传.
    """
    summary = row.summaryJson or {}
    if not isinstance(summary, dict):
        summary = {}
    steps = row.stepsJson or []
    if not isinstance(steps, list):
        steps = []
    # 给前端补全顶层字段, 确保跟 load_public_trace 输出一致
    summary.setdefault("trace_id", row.traceId)
    summary.setdefault("conversation_id", row.conversationId)
    summary.setdefault("message", row.rootMessage)
    summary.setdefault("duration_ms", row.totalDurationMs)
    summary.setdefault("total_tokens", row.totalTokens)
    summary.setdefault("llm_step_count", row.llmStepCount)
    if row.shareUrl:
        summary.setdefault("external_url", row.shareUrl)
    return {"trace": summary, "steps": steps}
