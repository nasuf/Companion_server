"""Admin runtime job inspection and DLQ actions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
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
