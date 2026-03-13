from fastapi import APIRouter, HTTPException, Query

from app.db import db
from app.models.user import UserCreate, UserUpdate, UserResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserResponse)
async def create_user(data: UserCreate):
    user = await db.user.create(data={"name": data.name, "email": data.email})
    return UserResponse(
        id=user.id, name=user.name, email=user.email, created_at=str(user.createdAt)
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=user.id, name=user.name, email=user.email, created_at=str(user.createdAt)
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
        id=user.id, name=user.name, email=user.email, created_at=str(user.createdAt)
    )


@router.get("", response_model=list[UserResponse])
async def list_users(limit: int = Query(default=50, le=200), offset: int = 0):
    users = await db.user.find_many(take=limit, skip=offset)
    return [
        UserResponse(
            id=u.id, name=u.name, email=u.email, created_at=str(u.createdAt)
        )
        for u in users
    ]
