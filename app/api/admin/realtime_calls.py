"""实时语音通话诊断 (VOICE01.01): 逐通计数与基线分位数, 只读, 不含通话原文。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_admin_jwt

router = APIRouter(prefix="/admin-api/realtime-calls", tags=["admin-realtime-calls"])


@router.get("/baseline")
async def call_baseline(
    user_id: str | None = Query(None, max_length=200),
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(50, ge=1, le=200),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.realtime_voice.call_stats import baseline
    return await baseline(user_id=user_id, days=days, limit=limit)
