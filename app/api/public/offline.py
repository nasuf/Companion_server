from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_admin_jwt, require_user
from app.models.chat_media import ChatAttachmentResponse
from app.models.offline import (
    GiftAddressRequest,
    GiftAddressResponse,
    GiftThanksRequest,
    GiftThanksResponse,
    GiftTrackingResponse,
    GiftsHomeResponse,
    OfflineActivitiesResponse,
    OfflineActivityClearResponse,
    OfflineActivityCompleteRequest,
    OfflineActivityImageUpload,
    OfflineActivityItem,
    OfflineHomeResponse,
    RealWorldGiftItem,
)
from app.services.offline import (
    activity_media_repo,
    activity_media_storage,
    activity_service,
    gift_service,
)

router = APIRouter(prefix="/offline", tags=["offline"])


@router.get("/home", response_model=OfflineHomeResponse)
async def get_offline_home(
    workspace_id: str | None = Query(default=None),
    user: dict = Depends(require_user),
):
    return await activity_service.get_home(str(user["sub"]), workspace_id)


@router.get("/activities", response_model=OfflineActivitiesResponse)
async def get_offline_activities(
    workspace_id: str | None = Query(default=None),
    user: dict = Depends(require_user),
):
    return await activity_service.list_activities(str(user["sub"]), workspace_id)


@router.post("/activities/recommend", response_model=OfflineActivityItem | None)
async def create_offline_activity_recommendation(
    workspace_id: str | None = Query(default=None),
    user: dict = Depends(require_user),
):
    activity = await activity_service.create_recommendation_for_user(
        user_id=str(user["sub"]),
        workspace_id=workspace_id,
        source="manual",
    )
    return OfflineActivityItem(**activity) if activity else None


@router.delete("/admin/activities", response_model=OfflineActivityClearResponse)
async def clear_current_user_offline_activities(
    user: dict = Depends(require_admin_jwt),
):
    result = await activity_service.clear_all_activities(str(user["sub"]))
    return OfflineActivityClearResponse(**result)


@router.post("/admin/gifts/mock", response_model=RealWorldGiftItem)
async def create_mock_gift(
    workspace_id: str | None = Query(default=None),
    delivered: bool = Query(default=False),
    user: dict = Depends(require_admin_jwt),
):
    gift = await gift_service.create_mock_gift_for_user(
        user_id=str(user["sub"]),
        workspace_id=workspace_id,
        delivered=delivered,
    )
    return RealWorldGiftItem(**gift)


@router.get("/activities/{activity_id}", response_model=OfflineActivityItem)
async def get_offline_activity(
    activity_id: str,
    user: dict = Depends(require_user),
):
    return await activity_service.get_activity(str(user["sub"]), activity_id)


@router.post("/activities/{activity_id}/accept", response_model=OfflineActivityItem)
async def accept_offline_activity(
    activity_id: str,
    user: dict = Depends(require_user),
):
    return await activity_service.accept_activity(str(user["sub"]), activity_id)


@router.post("/activities/{activity_id}/ignore", response_model=OfflineActivityItem)
async def ignore_offline_activity(
    activity_id: str,
    user: dict = Depends(require_user),
):
    return await activity_service.ignore_activity(str(user["sub"]), activity_id)


@router.post("/activities/{activity_id}/complete", response_model=OfflineActivityItem)
async def complete_offline_activity(
    activity_id: str,
    data: OfflineActivityCompleteRequest,
    user: dict = Depends(require_user),
):
    return await activity_service.complete_activity(
        str(user["sub"]),
        activity_id,
        text=data.text,
        photo_attachment_ids=data.photo_attachment_ids,
        audio_attachment_id=data.audio_attachment_id,
    )


@router.post(
    "/activities/{activity_id}/media",
    response_model=ChatAttachmentResponse,
)
async def upload_offline_activity_image(
    activity_id: str,
    data: OfflineActivityImageUpload,
    user: dict = Depends(require_user),
):
    user_id = str(user["sub"])
    if not await activity_media_repo.activity_belongs_to_user(activity_id, user_id):
        raise HTTPException(status_code=404, detail="Activity not found")
    if data.kind == "audio":
        mime = activity_media_storage.normalize_audio_mime(data.mime)
        blob = activity_media_storage.decode_audio_base64(data.base64)
        storage_key = activity_media_storage.save_audio_blob(
            user_id=user_id,
            blob=blob,
            mime=mime,
        )
        width = None
        height = None
    else:
        mime = activity_media_storage.normalize_image_mime(data.mime)
        blob = activity_media_storage.decode_image_base64(data.base64)
        storage_key = activity_media_storage.save_image_blob(
            user_id=user_id,
            blob=blob,
            mime=mime,
        )
        width = data.width
        height = data.height
    try:
        media = await activity_media_repo.create_media(
            recommendation_id=activity_id,
            user_id=user_id,
            kind=data.kind,
            storage_key=storage_key,
            url=activity_media_storage.media_url(storage_key),
            mime=mime,
            size=len(blob),
            name=data.name,
            width=width,
            height=height,
            duration_seconds=data.duration_seconds if data.kind == "audio" else None,
        )
    except Exception:
        activity_media_storage.delete_media_file(storage_key)
        raise
    return _media_response(media)


@router.get("/media/{storage_key}")
async def get_offline_activity_media(
    storage_key: str,
    user: dict = Depends(require_user),
):
    return activity_media_storage.serve_media(
        storage_key,
        user_id=str(user["sub"]),
        is_admin=user.get("role") == "admin",
    )


@router.get("/gifts", response_model=GiftsHomeResponse)
async def get_offline_gifts(
    workspace_id: str | None = Query(default=None),
    user: dict = Depends(require_user),
):
    return await gift_service.get_gifts(str(user["sub"]), workspace_id)


@router.get("/gifts/address", response_model=GiftAddressResponse)
async def get_gift_address(user: dict = Depends(require_user)):
    return await gift_service.get_address(str(user["sub"]))


@router.put("/gifts/address", response_model=GiftAddressResponse)
async def update_gift_address(
    data: GiftAddressRequest,
    user: dict = Depends(require_user),
):
    return await gift_service.save_address(str(user["sub"]), data)


@router.get("/gifts/{gift_id}", response_model=RealWorldGiftItem)
async def get_offline_gift(
    gift_id: str,
    user: dict = Depends(require_user),
):
    return await gift_service.get_gift(str(user["sub"]), gift_id)


@router.get("/gifts/{gift_id}/tracking", response_model=GiftTrackingResponse)
async def get_gift_tracking(
    gift_id: str,
    user: dict = Depends(require_user),
):
    return await gift_service.get_tracking(str(user["sub"]), gift_id)


@router.post("/gifts/{gift_id}/thanks", response_model=GiftThanksResponse)
async def thank_gift(
    gift_id: str,
    data: GiftThanksRequest,
    user: dict = Depends(require_user),
):
    return await gift_service.send_thanks(
        str(user["sub"]),
        gift_id,
        data.message,
        client_id=data.client_id,
    )


def _media_response(
    media: activity_media_repo.OfflineActivityMedia,
) -> ChatAttachmentResponse:
    return ChatAttachmentResponse(
        id=media.id,
        kind=media.kind,
        name=media.name,
        mime=media.mime,
        size=media.size,
        width=media.width,
        height=media.height,
        duration_seconds=media.duration_seconds,
        url=media.url,
        vision_status="ready",
        vision_summary=None,
        created_at=str(media.created_at) if media.created_at else None,
    )
