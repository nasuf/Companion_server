from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.jwt_auth import require_user
from app.db import db
from app.models.user import (
    ProfileStatsResponse,
    UserLocationRequest,
    UserLocationResponse,
    UserResponse,
    UserUpdate,
)
from app.services.portrait import get_latest_portrait
from app.services.profile_stats import get_profile_stats_for_workspace
from app.services.workspace.workspaces import get_active_workspace, get_workspace_by_id

router = APIRouter(prefix="/users", tags=["users"])


# 注: POST /users 创建匿名 user 的旧端点已删除。新建用户必须走
# /auth/register 或受控的第三方登录入口，避免绕过身份初始化。


@router.get("/me/profile-stats", response_model=ProfileStatsResponse)
async def get_my_profile_stats(
    workspace_id: str | None = None,
    user: dict = Depends(require_user),
):
    user_id = user.get("sub")
    workspace = (
        await get_workspace_by_id(workspace_id)
        if workspace_id
        else await get_active_workspace(user_id=user_id)
    )
    if not workspace or getattr(workspace, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Workspace not found")
    if user.get("role") != "admin" and workspace.userId != user_id:
        raise HTTPException(status_code=403, detail="Not your workspace")
    return await get_profile_stats_for_workspace(user_id=user_id, workspace=workspace)


def _location_response(row) -> UserLocationResponse:
    latitude = getattr(row, "location_latitude", None)
    longitude = getattr(row, "location_longitude", None)
    permission_status = getattr(row, "location_permission_status", None)
    updated_at = getattr(row, "location_updated_at", None)
    return UserLocationResponse(
        has_location=latitude is not None
        and longitude is not None
        and permission_status in {"whileInUse", "always"},
        latitude=latitude,
        longitude=longitude,
        city=getattr(row, "location_city", None),
        region=getattr(row, "location_region", None),
        country=getattr(row, "location_country", None),
        permission_status=permission_status,
        updated_at=updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at,
    )


@router.put("/me/location", response_model=UserLocationResponse)
async def update_my_location(
    data: UserLocationRequest,
    user: dict = Depends(require_user),
):
    rows = await db.query_raw(
        """
        UPDATE users
        SET location_latitude = $2,
            location_longitude = $3,
            location_city = NULLIF($4, ''),
            location_region = NULLIF($5, ''),
            location_country = NULLIF($6, ''),
            location_source = NULLIF($7, ''),
            location_permission_status = NULLIF($8, ''),
            location_updated_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        RETURNING location_latitude, location_longitude, location_city,
                  location_region, location_country, location_permission_status,
                  location_updated_at
        """,
        str(user["sub"]),
        data.latitude,
        data.longitude,
        (data.city or "").strip(),
        (data.region or "").strip(),
        (data.country or "").strip(),
        data.source.strip(),
        data.permission_status.strip(),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    return _location_response(rows[0])


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=user.id, username=user.username, email=user.email, created_at=str(user.createdAt)
    )


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, data: UserUpdate):
    update_data = data.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    user = await db.user.update(where={"id": user_id}, data=update_data)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=user.id, username=user.username, email=user.email, created_at=str(user.createdAt)
    )


@router.get("/{user_id}/portrait")
async def get_user_portrait(user_id: str, agent_id: str):
    """Get the latest AI-generated user portrait."""
    portrait = await get_latest_portrait(user_id, agent_id)
    if not portrait:
        raise HTTPException(status_code=404, detail="Portrait not found")
    return {"portrait": portrait}


@router.get("", response_model=list[UserResponse])
async def list_users(limit: int = Query(default=50, le=200), offset: int = 0):
    users = await db.user.find_many(take=limit, skip=offset)
    return [
        UserResponse(
            id=u.id, username=u.username, email=u.email, created_at=str(u.createdAt)
        )
        for u in users
    ]
