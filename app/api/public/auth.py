from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.db import db
from app.models.auth import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    WeChatMobileLoginRequest,
)
from app.services.auth import hash_password, verify_password, create_jwt
from app.services.auth_security import (
    audit_auth_request_event,
    clear_login_failures,
    enforce_login_rate_limit,
    enforce_register_rate_limit,
    record_login_failure,
)
from app.api.jwt_auth import require_user
from app.services.workspace.workspaces import get_active_workspace
from app.services.wechat_auth import (
    WeChatLoginError,
    exchange_wechat_code,
    find_or_create_wechat_user,
)
from app.services.user_activity import record_user_activity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


async def _build_auth_response(user, token: str) -> AuthResponse:
    workspace = await get_active_workspace(user_id=user.id)
    agent = None
    conversation = None
    if workspace:
        agent = await db.aiagent.find_unique(where={"id": workspace.agentId})
        conversation = await db.conversation.find_first(
            where={
                "workspaceId": workspace.id,
                "isDeleted": False,
            },
            order={"updatedAt": "desc"},
        )
    return AuthResponse(
        token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
        has_agent=workspace is not None and agent is not None,
        agent_id=agent.id if agent else None,
        agent_name=agent.name if agent else None,
        agent_avatar_key=getattr(agent, "avatarKey", None) if agent else None,
        agent_avatar_url=getattr(agent, "avatarUrl", None) if agent else None,
        agent_city=getattr(agent, "city", None) if agent else None,
        workspace_id=workspace.id if workspace else None,
        conversation_id=conversation.id if conversation else None,
    )


@router.post("/register", response_model=AuthResponse)
async def register(data: RegisterRequest, request: Request):
    await enforce_register_rate_limit(request)
    existing = await db.user.find_unique(where={"username": data.username})
    if existing:
        audit_auth_request_event(
            "register_conflict",
            request,
            username=data.username,
            outcome="failed",
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已被注册",
        )

    hashed = hash_password(data.password)
    user = await db.user.create(
        data={
            "username": data.username,
            "hashedPassword": hashed,
            "role": "user",
        }
    )

    token = create_jwt(user.id, user.role)
    audit_auth_request_event(
        "register_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    logger.info("User registered", extra={"event": "auth_register", "user_id": user.id})
    await record_user_activity(user.id, source="register")
    return await _build_auth_response(user, token)


@router.post("/login", response_model=AuthResponse)
async def login(data: LoginRequest, request: Request):
    await enforce_login_rate_limit(request, data.username)
    user = await db.user.find_unique(where={"username": data.username})
    if not user:
        await record_login_failure(request, data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    if not user.hashedPassword or not verify_password(data.password, user.hashedPassword):
        await record_login_failure(request, data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    await clear_login_failures(request, data.username)
    token = create_jwt(user.id, user.role)
    audit_auth_request_event(
        "login_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    logger.info("User logged in", extra={"event": "auth_login", "user_id": user.id})
    await record_user_activity(user.id, source="password_login")
    return await _build_auth_response(user, token)


@router.post("/wechat/mobile", response_model=AuthResponse)
async def wechat_mobile_login(data: WeChatMobileLoginRequest, request: Request):
    rate_limit_key = f"wechat:{data.platform}"
    await enforce_login_rate_limit(request, rate_limit_key)
    try:
        token_payload = await exchange_wechat_code(data.code)
        user = await find_or_create_wechat_user(token_payload)
    except WeChatLoginError:
        await record_login_failure(request, rate_limit_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="微信登录失败，请稍后重试",
        )

    if not user:
        await record_login_failure(request, rate_limit_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="微信登录失败，请稍后重试",
        )

    await clear_login_failures(request, rate_limit_key)
    token = create_jwt(user.id, user.role)
    audit_auth_request_event(
        "wechat_mobile_login_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    logger.info(
        "User logged in with WeChat",
        extra={
            "event": "auth_wechat_login",
            "user_id": user.id,
            "platform": data.platform,
        },
    )
    await record_user_activity(user.id, source="wechat_login")
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
    await record_user_activity(user.id, source="auth_me")
    return await _build_auth_response(user, token)
