from fastapi import APIRouter, HTTPException, Query

from app.db import db
from app.models.user import UserUpdate, UserResponse
from app.services.portrait import get_latest_portrait

router = APIRouter(prefix="/users", tags=["users"])


# 注: POST /users 创建匿名 user 的旧端点已删除. schema 要求 hashedPassword
# 必填, 旧端点没传 → 隐式废弃。所有新建用户必须走 /auth/register。


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
