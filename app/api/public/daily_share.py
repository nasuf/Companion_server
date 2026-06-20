from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_user
from app.models.chat_links import DailyShareLinksResponse
from app.models.daily_share import DailySharePhotosResponse
from app.services.chat_links import list_user_link_groups
from app.services.daily_share.photos import list_user_photo_groups

router = APIRouter(prefix="/daily-share", tags=["daily-share"])


@router.get("/photos", response_model=DailySharePhotosResponse)
async def list_daily_share_photos(
    limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    user: dict = Depends(require_user),
) -> DailySharePhotosResponse:
    groups = await list_user_photo_groups(user["sub"], limit=limit)
    return DailySharePhotosResponse(
        total=sum(group.count for group in groups),
        groups=groups,
    )


@router.get("/links", response_model=DailyShareLinksResponse)
async def list_daily_share_links(
    limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    user: dict = Depends(require_user),
) -> DailyShareLinksResponse:
    groups = await list_user_link_groups(user["sub"], limit=limit)
    return DailyShareLinksResponse(
        total=sum(group.count for group in groups),
        groups=groups,
    )
