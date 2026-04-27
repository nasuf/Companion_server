from __future__ import annotations

import logging

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.redis_client import get_redis
from app.services.chat.tracing import retry_share_for_message
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


@router.post("/retry/{message_id}")
async def retry_trace(
    message_id: str,
    user: dict = Depends(require_user),
):
    """重试单条消息的 LangSmith trace 分享.

    用于历史消息: trace_pending=true 永久卡住 / trace_failed=true 重试触发.
    内部 share_run_with_retry 已含 3 次指数退避; 加 Redis SETNX 限单消息 30s
    内只能点 1 次, 防止用户连点耗 LangSmith API quota.

    返回:
      200 {"trace_url": "..."} — 成功 (或之前已成功)
      400 {"detail": "no_trace_id"} — 老消息没有底层 LangSmith run, 不能重试
      403 {"detail": "not_your_message"} — 跨用户访问
      404 {"detail": "message_not_found"}
      429 {"detail": "rate_limited"} — 30s 内已点过
      503 {"detail": "share_failed: ..."} — LangSmith API 三次重试都失败
    """
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="auth_required")

    # Per-message rate limit: SETNX TTL 30s. 共享 share_run_with_retry 内 3 次
    # 重试已含 7s 指数退避, 30s 锁覆盖整次重试 + 1x 余量, 不会卡住正常用户.
    redis = await get_redis()
    lock_key = f"trace:retry:{message_id}"
    if not await redis.set(lock_key, "1", nx=True, ex=30):
        raise HTTPException(status_code=429, detail="rate_limited")

    try:
        return await retry_share_for_message(message_id, user_id=user_id)
    except PermissionError:
        raise HTTPException(status_code=403, detail="not_your_message")
    except ValueError as e:
        # message_not_found / no_trace_id / metadata_invalid
        msg = str(e)
        status = 404 if msg == "message_not_found" else 400
        raise HTTPException(status_code=status, detail=msg)
    except Exception as e:
        # share_run 三次重试失败 — 清锁让用户能再点 (体感更好), 但 logger.exception
        # 留 stack 便于查 LangSmith 侧问题.
        try:
            await redis.delete(lock_key)
        except Exception:
            pass
        logger.exception(f"[TRACE] retry endpoint failed for msg {message_id[:8]}")
        raise HTTPException(status_code=503, detail=f"share_failed: {type(e).__name__}")
