"""Admin runtime job inspection and DLQ actions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from app.api.jwt_auth import require_admin_jwt
from app.services.runtime.job_queue import (
    RuntimeJobConflict,
    diagnose_runtime_job_queue,
    inspect_runtime_job,
    list_runtime_jobs,
    retry_runtime_jobs,
    resolve_runtime_job,
    retry_runtime_job,
)

router = APIRouter(prefix="/admin-api/runtime-jobs", tags=["admin-runtime-jobs"])


class RuntimeJobBatchRequest(BaseModel):
    job_ids: list[str] = Field(default_factory=list, min_length=1, max_length=200)


@router.get("")
async def list_jobs(
    status: str | None = Query(None, pattern="^(queued|delayed|running|dead_letter|dlq|failed|succeeded)$"),
    job_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        return await list_runtime_jobs(status=status, job_type=job_type, limit=limit)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="任务队列暂时不可用，请刷新状态后重试。") from error


@router.get("/diagnostics")
async def get_diagnostics(
    cursor: int = Query(0, ge=0),
    idempotency_cursor: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        return await diagnose_runtime_job_queue(cursor=cursor, idempotency_cursor=idempotency_cursor, limit=limit)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="任务队列暂时不可用，请刷新状态后重试。") from error


@router.post("/retry")
async def retry_jobs(
    payload: RuntimeJobBatchRequest,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        return await retry_runtime_jobs(payload.job_ids)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="任务队列暂时不可用，请刷新状态后重试。") from error


@router.get("/location-events")
async def get_location_events(
    limit: int = Query(20, ge=1, le=50),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.location_memory import inspect_location_events
    try:
        return await inspect_location_events(limit=limit)
    except Exception as error:
        raise HTTPException(status_code=503, detail="位置分享处理记录暂时不可用，请重试。") from error


@router.get("/message-followups")
async def get_message_followups(
    limit: int = Query(20, ge=1, le=50),
    status: str = Query("all", pattern="^(all|failed|pending_dispatch|queued|completed|cancelled)$"),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.runtime.message_followups import inspect_message_followups
    try:
        return await inspect_message_followups(limit=limit, status=status)
    except Exception as error:
        raise HTTPException(status_code=503, detail="消息统计处理记录暂时不可用，请重试。") from error


@router.post("/message-followups/{message_id}/retry")
async def retry_message_followup(
    message_id: str = Path(min_length=1, max_length=200),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.runtime.message_followups import FollowupRetryConflict, retry_message_followup as retry
    try:
        found = await retry(message_id)
    except FollowupRetryConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="消息统计恢复暂时不可用，请刷新后重试。") from error
    if not found:
        raise HTTPException(status_code=404, detail="message_followup_not_found")
    return {"status": "pending_dispatch", "message_id": message_id}


@router.get("/turn-followups")
async def get_turn_followups(
    limit: int = Query(20, ge=1, le=50),
    status: str = Query("all", pattern="^(all|failed|pending_dispatch|queued|completed|cancelled)$"),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.runtime.turn_followups import inspect_turn_followups
    try:
        return await inspect_turn_followups(limit=limit, status=status)
    except Exception as error:
        raise HTTPException(status_code=503, detail="整轮统计处理记录暂时不可用，请重试。") from error


@router.post("/turn-followups/{turn_id}/retry")
async def retry_turn_followup(
    turn_id: str = Path(min_length=1, max_length=200),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.runtime.turn_followups import FollowupRetryConflict, retry_turn_followup as retry
    try:
        found = await retry(turn_id)
    except FollowupRetryConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="整轮统计恢复暂时不可用，请刷新后重试。") from error
    if not found:
        raise HTTPException(status_code=404, detail="turn_followup_not_found")
    return {"status": "pending_dispatch", "turn_id": turn_id}


@router.get("/arrival-followups")
async def get_arrival_followups(
    limit: int = Query(20, ge=1, le=50),
    status: str = Query("all", pattern="^(all|failed|pending_dispatch|queued|completed|cancelled)$"),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.offline.arrival_followups import inspect_arrival_followups
    try:
        return await inspect_arrival_followups(limit=limit, status=status)
    except Exception as error:
        raise HTTPException(status_code=503, detail="到达陪伴处理记录暂时不可用，请重试。") from error


@router.post("/arrival-followups/{activity_id}/retry")
async def retry_arrival_followup(
    activity_id: str = Path(min_length=1, max_length=200),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.offline.arrival_followups import FollowupRetryConflict, retry_arrival_followup as retry
    try:
        found = await retry(activity_id)
    except FollowupRetryConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="到达陪伴恢复暂时不可用，请刷新后重试。") from error
    if not found:
        raise HTTPException(status_code=404, detail="arrival_followup_not_found")
    return {"status": "pending_dispatch", "activity_id": activity_id}


@router.get("/memory-processing")
async def get_memory_processing(
    limit: int = Query(20, ge=1, le=50),
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    from app.services.memory.recording.diagnostics import inspect_memory_processing
    try:
        return await inspect_memory_processing(limit=limit)
    except Exception as error:
        raise HTTPException(status_code=503, detail="记忆录入处理记录暂时不可用，请稍后重试。") from error


@router.get("/memory-write-receipts")
async def memory_write_receipts(_: dict = Depends(require_admin_jwt)) -> dict:
    from app.services.memory.recording.receipt_read import recent_receipts
    return await recent_receipts()


@router.get("/memory-message-processing")
async def memory_message_processing(
    limit: int = Query(20, ge=1, le=50),
    hours: int = Query(24, ge=1, le=168),
    _: dict = Depends(require_admin_jwt),
) -> dict:
    from app.services.memory.recording.receipt_read import recent_message_processing
    try:
        return await recent_message_processing(limit=limit, hours=hours)
    except Exception as error:
        raise HTTPException(status_code=503, detail="逐消息记忆处理记录暂时不可用。") from error


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        job = await inspect_runtime_job(job_id)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="任务队列暂时不可用，请刷新状态后重试。") from error
    if job is None:
        raise HTTPException(status_code=404, detail="runtime_job_not_found")
    return job


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        job = await retry_runtime_job(job_id)
    except RuntimeJobConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RedisError as error:
        raise HTTPException(status_code=503, detail="任务队列暂时不可用，请刷新状态后重试。") from error
    if job is None:
        raise HTTPException(status_code=404, detail="runtime_job_not_found")
    return job


@router.post("/{job_id}/resolve")
async def resolve_job(
    job_id: str,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    try:
        job = await resolve_runtime_job(job_id)
    except RuntimeJobConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RedisError as error:
        raise HTTPException(status_code=503, detail="任务队列暂时不可用，请刷新状态后重试。") from error
    if job is None:
        raise HTTPException(status_code=404, detail="runtime_job_not_found")
    return job
