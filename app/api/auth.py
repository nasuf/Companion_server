from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.db import db
from app.models.auth import RegisterRequest, LoginRequest, AuthResponse
from app.services.auth import hash_password, verify_password, create_jwt
from app.api.jwt_auth import require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _build_auth_response(user, token: str) -> AuthResponse:
    agent = await db.aiagent.find_first(where={"userId": user.id})
    return AuthResponse(
        token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
        has_agent=agent is not None,
        agent_id=agent.id if agent else None,
        agent_name=agent.name if agent else None,
    )


@router.post("/register", response_model=AuthResponse)
async def register(data: RegisterRequest):
    existing = await db.user.find_unique(where={"username": data.username})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已被注册",
        )

    hashed = hash_password(data.password)
    user = await db.user.create(
        data={
            "name": data.username,
            "username": data.username,
            "hashedPassword": hashed,
            "role": "user",
        }
    )

    token = create_jwt(user.id, user.role)
    logger.info(f"User registered: {user.username} ({user.id})")
    return await _build_auth_response(user, token)


@router.post("/login", response_model=AuthResponse)
async def login(data: LoginRequest):
    user = await db.user.find_unique(where={"username": data.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    if not verify_password(data.password, user.hashedPassword):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    token = create_jwt(user.id, user.role)
    logger.info(f"User logged in: {user.username} ({user.id})")
    return await _build_auth_response(user, token)


@router.get("/me", response_model=AuthResponse)
async def get_me(payload: dict = Depends(require_user)):
    user = await db.user.find_unique(where={"id": payload["sub"]})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
        )

    token = create_jwt(user.id, user.role)
    return await _build_auth_response(user, token)
