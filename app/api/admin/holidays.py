"""Admin API: 日历 · 节假日管理.

Endpoints:
  POST  /admin-api/holidays/preview       — 查询外部源, 不落库
  POST  /admin-api/holidays/bulk_save     — 保存挑选结果
  GET   /admin-api/holidays               — 已存列表 (可按年/国家过滤)
  DELETE /admin-api/holidays/{id}         — 单条删除
  POST  /admin-api/holidays/refresh       — 手动触发 refresh (同步执行)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.services.schedule_domain import holiday_cache, holiday_repo
from app.services.schedule_domain.holiday_repo import (
    REFRESHABLE_SOURCES,
    HolidayEntry,
)
from app.services.schedule_domain.holiday_sources import collect_candidates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-api/holidays", tags=["admin-holidays"])


# ─── Schemas ────────────────────────────────────────────────────────────

SourceKey = Literal["chinesecalendar", "nager", "local"]


class HolidayCandidatePayload(BaseModel):
    date: date
    name: str
    type: Literal["legal", "traditional", "international", "custom"]
    country_code: str = "CN"
    is_workday_swap: bool = False
    source: str
    metadata: dict[str, Any] | None = None


class HolidayResponse(HolidayCandidatePayload):
    id: str
    created_at: str | None = None
    updated_at: str | None = None


class PreviewRequest(BaseModel):
    year: int = Field(..., ge=2020, le=2100)
    # 要查询的数据源列表; 默认全查. 前端按需精细控制.
    sources: list[SourceKey] = Field(
        default_factory=lambda: ["chinesecalendar", "nager", "local"]
    )


class PreviewResponse(BaseModel):
    candidates: list[HolidayCandidatePayload]
    sources_status: dict[str, dict[str, Any]]


class BulkSaveRequest(BaseModel):
    entries: list[HolidayCandidatePayload]


class BulkSaveResponse(BaseModel):
    inserted: int
    updated: int
    skipped: int


class RefreshRequest(BaseModel):
    year: int = Field(..., ge=2020, le=2100)


# ─── Helpers ───────────────────────────────────────────────────────────

def _entry_to_response(e: HolidayEntry) -> HolidayResponse:
    assert e.id is not None  # always set when coming from DB
    return HolidayResponse(
        id=e.id,
        date=e.date,
        name=e.name,
        type=e.type,
        country_code=e.country_code,
        is_workday_swap=e.is_workday_swap,
        source=e.source,
        metadata=e.metadata,
        created_at=e.created_at.isoformat() if e.created_at else None,
        updated_at=e.updated_at.isoformat() if e.updated_at else None,
    )


def _payload_to_entry(p: HolidayCandidatePayload) -> HolidayEntry:
    return HolidayEntry(
        date=p.date,
        name=p.name,
        type=p.type,
        country_code=p.country_code,
        is_workday_swap=p.is_workday_swap,
        source=p.source,
        metadata=p.metadata,
    )


# ─── Endpoints ─────────────────────────────────────────────────────────

@router.post("/preview", response_model=PreviewResponse)
async def preview_holidays(
    payload: PreviewRequest,
    _: str = Depends(require_admin_jwt),
) -> PreviewResponse:
    entries, status = await collect_candidates(
        payload.year,
        sources=set(payload.sources),
    )
    return PreviewResponse(
        candidates=[
            HolidayCandidatePayload(
                date=e.date,
                name=e.name,
                type=e.type,
                country_code=e.country_code,
                is_workday_swap=e.is_workday_swap,
                source=e.source,
                metadata=e.metadata,
            )
            for e in entries
        ],
        sources_status=status.to_response_dict(),
    )


@router.post("/bulk_save", response_model=BulkSaveResponse)
async def bulk_save_holidays(
    payload: BulkSaveRequest,
    _: str = Depends(require_admin_jwt),
) -> BulkSaveResponse:
    if not payload.entries:
        raise HTTPException(status_code=400, detail="entries must not be empty")
    entries = [_payload_to_entry(p) for p in payload.entries]
    try:
        # Admin-driven save is authoritative — allowed to overwrite manual rows
        stats = await holiday_repo.upsert_many(entries, allow_overwrite_manual=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await holiday_cache.reload()
    return BulkSaveResponse(**stats)


@router.get("", response_model=list[HolidayResponse])
async def list_admin_holidays(
    year: int | None = Query(None, ge=2020, le=2100),
    country: str | None = None,
    _: str = Depends(require_admin_jwt),
) -> list[HolidayResponse]:
    entries = await holiday_repo.list_holidays(year=year, country_code=country)
    return [_entry_to_response(e) for e in entries]


@router.delete("/{holiday_id}")
async def delete_holiday(
    holiday_id: str,
    _: str = Depends(require_admin_jwt),
) -> dict[str, bool]:
    ok = await holiday_repo.delete_by_id(holiday_id)
    if not ok:
        raise HTTPException(status_code=404, detail="holiday not found")
    await holiday_cache.reload()
    return {"ok": True}


@router.post("/refresh", response_model=BulkSaveResponse)
async def refresh_holidays(
    payload: RefreshRequest,
    _: str = Depends(require_admin_jwt),
) -> BulkSaveResponse:
    """Manual refresh of one year. Preserves manual rows (source='manual').

    Cron-equivalent path; gives admin a button to pull latest without
    waiting for the weekly job.
    """
    entries, _status = await collect_candidates(payload.year)
    refreshable = [e for e in entries if e.source in REFRESHABLE_SOURCES]
    stats = await holiday_repo.upsert_many(refreshable, allow_overwrite_manual=False)
    logger.info(f"Admin-triggered refresh for {payload.year}: {stats}")
    await holiday_cache.reload()
    return BulkSaveResponse(**stats)
