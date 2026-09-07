from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from prisma.errors import RecordNotFoundError

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.models.feedback import (
    UserFeedbackItem,
    UserFeedbackListResponse,
    UserFeedbackStatusUpdate,
)
from app.services.feedback import storage

router = APIRouter(prefix="/admin-api/user-feedback", tags=["admin-user-feedback"])

_ALLOWED_STATUSES = {"open", "read", "resolved"}


def _serialize(row, *, admin_media: bool = True) -> UserFeedbackItem:
    user = getattr(row, "user", None)
    username = getattr(user, "username", None) if user else None
    display_name = getattr(user, "displayName", None) if user else None
    if isinstance(display_name, str):
        display_name = display_name.strip() or None
    image_keys = list(getattr(row, "imageKeys", None) or [])
    return UserFeedbackItem(
        id=row.id,
        user_id=row.userId,
        username=username,
        display_name=display_name,
        content=row.content,
        contact=row.contact,
        occurred_at=row.occurredAt,
        image_urls=[
            storage.build_media_url(key, admin=admin_media) for key in image_keys
        ],
        status=row.status,
        app_version=row.appVersion,
        platform=row.platform,
        created_at=row.createdAt.isoformat(),
        updated_at=row.updatedAt.isoformat(),
    )


@router.get("", response_model=UserFeedbackListResponse)
async def list_user_feedback(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: dict = Depends(require_admin_jwt),
):
    where: dict = {}
    if status:
        normalized = status.strip().lower()
        if normalized not in _ALLOWED_STATUSES:
            raise HTTPException(status_code=400, detail="invalid_status")
        where["status"] = normalized
    total = await db.userfeedback.count(where=where or None)
    rows = await db.userfeedback.find_many(
        where=where or None,
        include={"user": True},
        order={"createdAt": "desc"},
        skip=offset,
        take=limit,
    )
    return UserFeedbackListResponse(
        items=[_serialize(row) for row in rows],
        total=total,
    )


@router.get("/media/{key}")
async def get_feedback_media_admin(
    key: str,
    _: dict = Depends(require_admin_jwt),
):
    row = await db.userfeedback.find_first(where={"imageKeys": {"has": key}})
    if row is None:
        raise HTTPException(status_code=404, detail="Feedback media not found")
    return storage.serve_media(key)


@router.get("/{feedback_id}", response_model=UserFeedbackItem)
async def get_user_feedback(
    feedback_id: str,
    _: dict = Depends(require_admin_jwt),
):
    row = await db.userfeedback.find_unique(
        where={"id": feedback_id},
        include={"user": True},
    )
    if row is None:
        raise HTTPException(status_code=404, detail="feedback_not_found")
    return _serialize(row)


@router.patch("/{feedback_id}", response_model=UserFeedbackItem)
async def update_user_feedback_status(
    feedback_id: str,
    payload: UserFeedbackStatusUpdate,
    _: dict = Depends(require_admin_jwt),
):
    normalized = payload.status.strip().lower()
    if normalized not in _ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail="invalid_status")
    try:
        row = await db.userfeedback.update(
            where={"id": feedback_id},
            data={"status": normalized},
            include={"user": True},
        )
    except RecordNotFoundError:
        raise HTTPException(status_code=404, detail="feedback_not_found")
    return _serialize(row)
