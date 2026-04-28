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


@router.post("/resolve/{message_id}")
async def resolve_trace(
    message_id: str,
    user: dict = Depends(require_user),
):
    """懒触发: 用户首次点 Trace 按钮调用. share_run + load_public_trace +
    写 mirror + 返回 detail 一气呵成. 命中本地 mirror 直接返回, 走完一次
    share 后续打开都是 ~50ms.

    返回:
      200 {"trace_url": "...", "detail": {...}} — 成功 (含完整 trace + steps)
      400 {"detail": "no_trace_id"}             — 老消息没有底层 LangSmith run
      403 {"detail": "not_your_message"}        — 跨用户访问 (admin 跳过校验)
      404 {"detail": "message_not_found"}
      503 {"detail": "share_failed: ..."}       — LangSmith API 三次重试都失败
    """
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="auth_required")

    try:
        return await resolve_trace_for_message(
            message_id, user_id=user_id,
            is_admin=user.get("role") == "admin",
        )
    except HTTPException:
        # load_public_trace 抛 HTTPException (e.g. 404/502) — 直接透传, 别裹成 503
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail="not_your_message")
    except ValueError as e:
        msg = str(e)
        status = 404 if msg == "message_not_found" else 400
        raise HTTPException(status_code=status, detail=msg)
    except Exception as e:
        logger.exception(f"[TRACE] resolve failed for msg {message_id[:8]}")
        raise HTTPException(status_code=503, detail=f"share_failed: {type(e).__name__}")
