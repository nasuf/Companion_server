from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_admin_jwt, require_user
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
    OfflineActivityItem,
    OfflineHomeResponse,
    RealWorldGiftItem,
)
from app.services.offline import activity_service, gift_service

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
    return await gift_service.send_thanks(str(user["sub"]), gift_id, data.message)
