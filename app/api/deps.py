"""共享 FastAPI 依赖."""

from __future__ import annotations

from fastapi import HTTPException, status

from app.redis_client import is_redis_healthy


async def require_redis() -> None:
    """应用到写路径 endpoint (POST /api/chat / proactive 触发等).

    Redis 不健康时直接 503, 不让请求穿透到后续 Redis 依赖 (聚合 / 计数 /
    延迟队列) 导致的乱象. 只读 GET 端点不加此依赖, 在 readonly mode 下
    继续可用.
    """
    if not is_redis_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis unavailable; write operations temporarily disabled",
        )
