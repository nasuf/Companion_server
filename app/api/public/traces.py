from __future__ import annotations

import logging

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.services.chat.tracing import resolve_trace_for_message
from app.services.public_trace import load_public_trace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])


class TraceResolveRequest(BaseModel):
    trace_url: str


@router.post("/public-detail")
async def get_public_trace_detail(
    payload: TraceResolveRequest,
    _: dict = Depends(require_user),
):
    return await load_public_trace(payload.trace_url)


async def _attach_trace_usage(result: dict, message_id: str) -> None:
    """把该轮的 llm_usage 汇总 (tokens/缓存命中/费用) 挂到 detail.usage.

    trace_id 从消息 metadata 取; 任何一步失败静默 — 用量是展示性信息,
    不该影响 trace 详情打开.
    """
    try:
        detail = result.get("detail") if isinstance(result, dict) else None
        if not isinstance(detail, dict):
            return
        from app.db import db
        from app.services.llm.usage_repo import aggregate_usage_by_trace_ids

        message = await db.message.find_unique(where={"id": message_id})
        metadata = message.metadata if message and isinstance(message.metadata, dict) else {}
        trace_id = str(metadata.get("trace_id") or "")
        if not trace_id:
            return
        usage = (await aggregate_usage_by_trace_ids([trace_id])).get(trace_id)
        if usage:
            detail["usage"] = usage
    except Exception as e:
        logger.debug(f"[TRACE] usage attach failed for msg {message_id[:8]}: {e}")


@router.post("/resolve/{message_id}")
async def resolve_trace(
    message_id: str,
    user: dict = Depends(require_user),
):
    """懒触发: 用户首次点 Trace 按钮调用. 本地 trace_runs (或 legacy LangSmith)
    → enrich → 写 mirror → 返回 detail 一气呵成. 命中本地 mirror 直接返回.

    返回:
      200 {"trace_url": "..."|null, "detail": {...}} — 成功 (本地 trace 无外链)
      400 {"detail": "no_trace_id"}             — 老消息没有底层 run
      403 {"detail": "not_your_message"}        — 跨用户访问 (admin 跳过校验)
      404 {"detail": "message_not_found"}
      404 {"detail": "trace_expired"}           — trace_runs 已过保留期清理
      503 {"detail": "share_failed: ..."}       — LangSmith API 三次重试都失败
    """
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="auth_required")

    try:
        result = await resolve_trace_for_message(
            message_id, user_id=user_id,
            is_admin=user.get("role") == "admin",
        )
        await _attach_trace_usage(result, message_id)
        return result
    except HTTPException:
        # load_public_trace 抛 HTTPException (e.g. 404/502) — 直接透传, 别裹成 503
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail="not_your_message")
    except ValueError as e:
        msg = str(e)
        status = 404 if msg in ("message_not_found", "trace_expired") else 400
        raise HTTPException(status_code=status, detail=msg)
    except Exception as e:
        logger.exception(f"[TRACE] resolve failed for msg {message_id[:8]}")
        raise HTTPException(status_code=503, detail=f"share_failed: {type(e).__name__}")
