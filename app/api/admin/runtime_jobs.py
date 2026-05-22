"""Admin runtime job inspection and DLQ actions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.services.runtime.job_queue import (
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
    return await list_runtime_jobs(status=status, job_type=job_type, limit=limit)


@router.post("/retry")
async def retry_jobs(
    payload: RuntimeJobBatchRequest,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    return await retry_runtime_jobs(payload.job_ids)


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    job = await inspect_runtime_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="runtime_job_not_found")
    return job


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    job = await retry_runtime_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="runtime_job_not_found")
    return job


@router.post("/{job_id}/resolve")
async def resolve_job(
    job_id: str,
    _: dict = Depends(require_admin_jwt),
) -> dict[str, Any]:
    job = await resolve_runtime_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="runtime_job_not_found")
    return job
