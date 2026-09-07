from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.jwt_auth import require_user
from app.db import db
from app.models.feedback import UserFeedbackCreateResponse
from app.services.feedback import storage

router = APIRouter(prefix="/users/me/feedback", tags=["user-feedback"])


@router.post("", response_model=UserFeedbackCreateResponse)
async def submit_user_feedback(
    content: str = Form(...),
    contact: str = Form(...),
    occurred_at: str | None = Form(default=None),
    app_version: str | None = Form(default=None),
    platform: str | None = Form(default=None),
    images: list[UploadFile] = File(default=[]),
    user: dict = Depends(require_user),
):
    body = (content or "").strip()
    contact_value = (contact or "").strip()
    if len(body) < 5:
        raise HTTPException(status_code=400, detail="content_too_short")
    if not contact_value:
        raise HTTPException(status_code=400, detail="contact_required")
    if len(images) > storage.max_images_per_feedback():
        raise HTTPException(status_code=400, detail="too_many_images")

    user_id = str(user["sub"])
    image_keys: list[str] = []
    try:
        for upload in images:
            blob = await upload.read()
            key = storage.save_feedback_image(
                user_id=user_id,
                blob=blob,
                mime=upload.content_type,
            )
            image_keys.append(key)
    except HTTPException:
        for key in image_keys:
            storage.delete_feedback_image(key)
        raise
    except Exception as exc:
        for key in image_keys:
            storage.delete_feedback_image(key)
        raise HTTPException(status_code=400, detail="invalid_image") from exc

    row = await db.userfeedback.create(
        data={
            "userId": user_id,
            "content": body,
            "contact": contact_value,
            "occurredAt": (occurred_at or "").strip() or None,
            "imageKeys": image_keys,
            "appVersion": (app_version or "").strip() or None,
            "platform": (platform or "").strip() or None,
        }
    )
    return UserFeedbackCreateResponse(
        id=row.id,
        created_at=row.createdAt.isoformat(),
    )


@router.get("/media/{key}")
async def get_feedback_media(key: str, user: dict = Depends(require_user)):
    row = await db.userfeedback.find_first(
        where={
            "userId": str(user["sub"]),
            "imageKeys": {"has": key},
        }
    )
    if row is None and user.get("role") != "admin":
        raise HTTPException(status_code=404, detail="Feedback media not found")
    if row is None:
        # Admin can read any feedback attachment via the admin route; this path
        # is scoped to the uploader's own submissions.
        raise HTTPException(status_code=404, detail="Feedback media not found")
    return storage.serve_media(key)
