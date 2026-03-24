from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter, Depends

from app.api.jwt_auth import require_user
from app.services.public_trace import load_public_trace

router = APIRouter(prefix="/traces", tags=["traces"])


class TraceResolveRequest(BaseModel):
    trace_url: str


@router.post("/public-detail")
async def get_public_trace_detail(
    payload: TraceResolveRequest,
    _: dict = Depends(require_user),
):
    return await load_public_trace(payload.trace_url)
