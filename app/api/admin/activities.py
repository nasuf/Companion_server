"""Search a user, inspect stored mobile activity copy and its bounded evidence."""

from fastapi import APIRouter, Depends, Query, Response

from app.api.jwt_auth import require_admin_jwt
from app.models.admin_activities import (
    ActivityFilter,
    ActivityPage,
    AdminActivityDetail,
    AdminActivityItem,
    AdminActivityRelatedPage,
    RelatedSection,
)
from app.models.admin_wallet import AdminWalletUserSearchItem
from app.services import game_points
from app.services.offline import activity_admin as service


def _no_store(response: Response):
    response.headers["Cache-Control"] = "private, no-store"


router = APIRouter(
    prefix="/admin-api/activities",
    tags=["admin", "offline"],
    dependencies=[Depends(require_admin_jwt), Depends(_no_store)],
)


@router.get("/users", response_model=list[AdminWalletUserSearchItem])
async def search_users(
    q: str = Query(default="", max_length=80),
    limit: int = Query(default=20, ge=1, le=50),
):
    return await game_points.search_users(q, limit=limit)


@router.get("/users/{user_id}")
async def user_context(user_id: str, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await service.user_context(user_id)


@router.get("", response_model=ActivityPage[AdminActivityItem])
async def list_activities(
    response: Response,
    user_id: str = Query(min_length=1, max_length=64),
    workspace_id: str | None = Query(default=None, max_length=64),
    status: ActivityFilter = "all",
    q: str = Query(default="", max_length=80),
    limit: int = Query(default=30, ge=1, le=50),
    offset: int = Query(default=0, ge=0, le=100_000),
):
    response.headers["Cache-Control"] = "no-store"
    return await service.list_activities(
        user_id, workspace_id, status, q, limit, offset
    )


@router.get("/{activity_id}", response_model=AdminActivityDetail)
async def detail(
    activity_id: str,
    response: Response,
    user_id: str = Query(min_length=1, max_length=64),
    workspace_id: str | None = Query(default=None, max_length=64),
):
    response.headers["Cache-Control"] = "no-store"
    return await service.detail(activity_id, user_id, workspace_id)


@router.get("/{activity_id}/images/{index}")
async def image(
    activity_id: str, index: int, user_id: str = Query(min_length=1, max_length=64)
):
    return await service.recommendation_image(activity_id, user_id, index)


@router.get("/{activity_id}/media/{media_id}/content")
async def media(
    activity_id: str, media_id: str, user_id: str = Query(min_length=1, max_length=64)
):
    return await service.media_content(activity_id, user_id, media_id)


@router.get("/{activity_id}/messages/{message_id}/images/{index}")
async def message_image(
    activity_id: str,
    message_id: str,
    index: int,
    user_id: str = Query(min_length=1, max_length=64),
):
    return await service.message_image(activity_id, user_id, message_id, index)


@router.get("/{activity_id}/messages/{message_id}/attachments/{attachment_id}")
async def attachment(
    activity_id: str,
    message_id: str,
    attachment_id: str,
    user_id: str = Query(min_length=1, max_length=64),
):
    return await service.attachment_content(
        activity_id, user_id, message_id, attachment_id
    )


@router.get("/{activity_id}/{section}", response_model=AdminActivityRelatedPage)
async def related(
    activity_id: str,
    section: RelatedSection,
    response: Response,
    user_id: str = Query(min_length=1, max_length=64),
    workspace_id: str | None = Query(default=None, max_length=64),
    mode: str = Query(default="linked", pattern="^(linked|journey)$"),
    limit: int = Query(default=30, ge=1, le=50),
    offset: int = Query(default=0, ge=0, le=100_000),
):
    response.headers["Cache-Control"] = "no-store"
    return await service.related(
        activity_id, user_id, section, mode, limit, offset, workspace_id
    )
