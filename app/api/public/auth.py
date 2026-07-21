from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.db import db
from app.models.auth import (
    AuthResponse,
    LoginRequest,
    PhoneBindRequest,
    RegisterRequest,
    SmsLoginRequest,
    SmsSendRequest,
    WeChatH5BindRequest,
    WeChatH5LoginRequest,
    WeChatMiniLoginRequest,
    WeChatMobileLoginRequest,
    WeChatProfileUpdate,
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
    SignupInfo,
    WeChatLoginError,
    _wechat_h5_configured,
    exchange_wechat_code,
    exchange_wechat_h5_code,
    exchange_wechat_miniprogram_code,
    find_or_create_wechat_user,
    update_wechat_profile,
)
from app.services.agent_avatars import build_avatar_url
from app.services.agent_template import ensure_default_agent_for_user
from app.services.notifications.presence import record_online
from app.services.user_activity import UserActivityWriteError, record_user_activity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])
_WECHAT_PROVIDER = "wechat"


async def _record_auth_activity(user_id: str, *, source: str) -> None:
    # 登录 / auth_me 即视为在线, 让刚登录 (含 H5) 立即计入实时在线; 后续 WS
    # ping / auth_me 刷新会持续续期. Best-effort, 失败不影响登录主流程.
    await record_online(user_id)
    try:
        await record_user_activity(
            user_id,
            source=source,
            raise_on_total_failure=True,
        )
    except UserActivityWriteError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="上线记录失败，请稍后重试",
        )


async def _wechat_profile_for_user(user_id: str) -> tuple[str | None, str | None]:
    identity = await db.authidentity.find_first(
        where={"userId": user_id, "provider": _WECHAT_PROVIDER},
        order={"updatedAt": "desc"},
    )
    profile = getattr(identity, "rawProfile", None) if identity else None
    if not isinstance(profile, dict):
        return None, None

    nickname = profile.get("nickname")
    avatar_url = profile.get("headimgurl")
    display_name = nickname.strip() if isinstance(nickname, str) else None
    avatar = avatar_url.strip() if isinstance(avatar_url, str) else None
    if avatar and avatar.startswith("http://"):
        # 微信头像 CDN (qlogo.cn) 常回 http://, 在 https 页面里会被浏览器按
        # mixed-content 拦截导致头像不显示; CDN 本身支持 https, 读取侧统一升级.
        avatar = "https://" + avatar[len("http://"):]
    return display_name or None, avatar or None


def _mask_phone(phone: str | None) -> str | None:
    if not phone or len(phone) != 11:
        return phone
    return f"{phone[:3]}****{phone[-4:]}"


async def _build_auth_response(user, token: str) -> AuthResponse:
    from app.services.phone_auth import get_identity_summary

    phone, wechat_bound = await get_identity_summary(user.id)
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
    user_display_name, user_avatar_url = await _wechat_profile_for_user(user.id)
    if not user_display_name and phone:
        # Phone-only accounts have no WeChat nickname; fall back to a friendly
        # "用户+尾号" instead of surfacing the opaque ph_xxxx username hash.
        user_display_name = f"用户{phone[-4:]}"
    agent_avatar_key = getattr(agent, "avatarKey", None) if agent else None
    return AuthResponse(
        token=token,
        user_id=user.id,
        username=user.username,
        user_display_name=user_display_name or user.username,
        user_avatar_url=user_avatar_url,
        role=user.role,
        has_agent=workspace is not None and agent is not None,
        agent_id=agent.id if agent else None,
        agent_name=agent.name if agent else None,
        agent_avatar_key=agent_avatar_key,
        agent_avatar_url=(
            build_avatar_url(agent_avatar_key)
            or (getattr(agent, "avatarUrl", None) if agent else None)
        ),
        agent_city=getattr(agent, "city", None) if agent else None,
        workspace_id=workspace.id if workspace else None,
        conversation_id=conversation.id if conversation else None,
        phone=_mask_phone(phone),
        wechat_bound=wechat_bound,
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
    # channel comes from the client (app / miniprogram / h5 / web); legacy
    # clients omit it and land on the plain "password" source.
    signup_source = f"password_{data.channel}" if data.channel else "password"
    user_create_data = {
        "username": data.username,
        "hashedPassword": hashed,
        "role": "user",
        "signupSource": signup_source,
    }
    if data.platform:
        user_create_data["signupPlatform"] = data.platform
    if data.os_version:
        user_create_data["signupOsVersion"] = data.os_version
    if data.app_version:
        user_create_data["signupAppVersion"] = data.app_version
    user = await db.user.create(data=user_create_data)

    token = create_jwt(user.id, user.role)
    audit_auth_request_event(
        "register_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    logger.info(
        "User registered",
        extra={
            "event": "auth_register",
            "user_id": user.id,
            "signup_source": signup_source,
            "signup_platform": data.platform,
        },
    )
    await ensure_default_agent_for_user(user.id)
    await _record_auth_activity(user.id, source="register")
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
    await ensure_default_agent_for_user(user.id)
    await _record_auth_activity(user.id, source="password_login")
    return await _build_auth_response(user, token)


@router.post("/wechat/mobile", response_model=AuthResponse)
async def wechat_mobile_login(data: WeChatMobileLoginRequest, request: Request):
    rate_limit_key = f"wechat:{data.platform}"
    await enforce_login_rate_limit(request, rate_limit_key)
    try:
        token_payload = await exchange_wechat_code(data.code)
        user = await find_or_create_wechat_user(
            token_payload,
            signup=SignupInfo(
                source="wechat_app",
                platform=data.platform,
                os_version=data.os_version,
                app_version=data.app_version,
            ),
        )
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
    await _record_auth_activity(user.id, source="wechat_login")
    return await _build_auth_response(user, token)


@router.post("/wechat/miniprogram", response_model=AuthResponse)
async def wechat_miniprogram_login(data: WeChatMiniLoginRequest, request: Request):
    rate_limit_key = "wechat:miniprogram"
    await enforce_login_rate_limit(request, rate_limit_key)
    try:
        token_payload = await exchange_wechat_miniprogram_code(data.code)
        user = await find_or_create_wechat_user(
            token_payload,
            signup=SignupInfo(
                source="wechat_miniprogram",
                platform=data.platform,
                os_version=data.os_version,
                app_version=data.app_version,
            ),
        )
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
    # Give brand-new users an instant cloned default agent (if configured) so the
    # Mini Program can go straight to chat. No-op for returning users.
    await ensure_default_agent_for_user(user.id)
    token = create_jwt(user.id, user.role)
    audit_auth_request_event(
        "wechat_miniprogram_login_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    logger.info(
        "User logged in with WeChat Mini Program",
        extra={"event": "auth_wechat_mini_login", "user_id": user.id},
    )
    await _record_auth_activity(user.id, source="wechat_miniprogram_login")
    return await _build_auth_response(user, token)


@router.post("/sms/send")
async def sms_send(data: SmsSendRequest, request: Request):
    """发送登录/绑定验证码. 响应不区分号码是否已注册 (防枚举)."""
    from app.services import sms

    if not sms.service.sms_login_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="短信服务暂未开放",
        )
    phone = sms.normalize_cn_phone(data.phone)
    if phone is None:
        raise HTTPException(status_code=400, detail="手机号格式不正确")

    # 发送侧限流全部在 sms.service 层: 60s 冷却 + 手机号日限 + IP 时/日限.
    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else ""
    )
    try:
        await sms.send_login_code(phone, client_ip=client_ip or None)
    except sms.SmsRateLimited as exc:
        detail = {
            "cooldown": "发送太频繁，请稍后再试",
            "daily_limit": "该手机号今日发送次数已达上限",
            "ip_limit": "发送次数过多，请稍后再试",
        }.get(exc.reason, "发送太频繁，请稍后再试")
        raise HTTPException(status_code=429, detail=detail)
    except sms.SmsSendError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="短信发送失败，请稍后重试",
        )
    return {"ok": True, "ttl_minutes": sms.service.CODE_TTL_MINUTES}


@router.post("/sms/login", response_model=AuthResponse)
async def sms_login(data: SmsLoginRequest, request: Request):
    """手机号验证码登录: 无账号自动注册 (与微信登录同一身份表)."""
    from app.services import sms
    from app.services.phone_auth import find_or_create_phone_user

    phone = sms.normalize_cn_phone(data.phone)
    if phone is None:
        raise HTTPException(status_code=400, detail="手机号格式不正确")

    await enforce_login_rate_limit(request, f"smslogin:{phone}")
    if not await sms.verify_code(phone, data.code):
        await record_login_failure(request, f"smslogin:{phone}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="验证码错误或已过期",
        )

    await clear_login_failures(request, f"smslogin:{phone}")
    # channel comes from the client (app / h5 / ...); legacy or unknown clients
    # land on the plain "sms" source — mirrors the register endpoint's rule.
    signup_source = f"sms_{data.channel}" if data.channel else "sms"
    user = await find_or_create_phone_user(
        phone,
        signup_fields=SignupInfo(
            source=signup_source,
            platform=data.platform,
            os_version=data.os_version,
            app_version=data.app_version,
        ).user_create_fields(),
    )
    await ensure_default_agent_for_user(user.id)
    token = create_jwt(user.id, user.role)
    audit_auth_request_event(
        "sms_login_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    logger.info(
        "User logged in with SMS",
        extra={"event": "auth_sms_login", "user_id": user.id},
    )
    await _record_auth_activity(user.id, source="sms_login")
    return await _build_auth_response(user, token)


@router.post("/phone/bind", response_model=AuthResponse)
async def phone_bind(data: PhoneBindRequest, request: Request, payload: dict = Depends(require_user)):
    """已登录用户绑定/换绑手机号 (短信验证码确认)."""
    from app.services import sms
    from app.services.phone_auth import IdentityConflict, bind_phone_to_user

    user = await db.user.find_unique(where={"id": payload["sub"]})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")

    phone = sms.normalize_cn_phone(data.phone)
    if phone is None:
        raise HTTPException(status_code=400, detail="手机号格式不正确")

    await enforce_login_rate_limit(request, f"smsbind:{phone}")
    if not await sms.verify_code(phone, data.code):
        await record_login_failure(request, f"smsbind:{phone}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="验证码错误或已过期",
        )

    try:
        await bind_phone_to_user(user.id, phone)
    except IdentityConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该手机号已绑定其他账号",
        )
    audit_auth_request_event(
        "phone_bind_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    token = create_jwt(user.id, user.role)
    return await _build_auth_response(user, token)


@router.post("/wechat/h5/bind", response_model=AuthResponse)
async def wechat_h5_bind(data: WeChatH5BindRequest, request: Request, payload: dict = Depends(require_user)):
    """已登录用户 (如手机号账号) 绑定微信身份."""
    from app.services.phone_auth import IdentityConflict
    from app.services.wechat_auth import bind_wechat_to_user

    user = await db.user.find_unique(where={"id": payload["sub"]})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")

    rate_limit_key = "wechat:h5bind"
    await enforce_login_rate_limit(request, rate_limit_key)
    try:
        token_payload = await exchange_wechat_h5_code(data.code)
        await bind_wechat_to_user(user.id, token_payload)
    except WeChatLoginError:
        await record_login_failure(request, rate_limit_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="微信授权失败，请稍后重试",
        )
    except IdentityConflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该微信已绑定其他账号",
        )
    await clear_login_failures(request, rate_limit_key)
    audit_auth_request_event(
        "wechat_bind_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    token = create_jwt(user.id, user.role)
    return await _build_auth_response(user, token)


@router.get("/wechat/h5/config")
async def wechat_h5_config():
    """H5 页面启动时探测: 是否展示微信一键登录 + OAuth 跳转所需的公众号 appid.

    appid 本身是公开信息 (会出现在 OAuth 跳转 URL 里), 无需鉴权.
    """
    from app.config import settings
    from app.services.sms.service import sms_login_available

    enabled = _wechat_h5_configured()
    return {
        "enabled": enabled,
        "app_id": settings.wechat_h5_app_id.strip() or None if enabled else None,
        "sms_enabled": sms_login_available(),
    }


@router.post("/wechat/h5", response_model=AuthResponse)
async def wechat_h5_login(data: WeChatH5LoginRequest, request: Request):
    """公众号网页授权登录 (H5). 与小程序/移动端写同一套身份表, unionid 归一."""
    rate_limit_key = "wechat:h5"
    await enforce_login_rate_limit(request, rate_limit_key)
    try:
        token_payload = await exchange_wechat_h5_code(data.code)
        user = await find_or_create_wechat_user(
            token_payload,
            signup=SignupInfo(
                source="wechat_h5",
                platform=data.platform,
                os_version=data.os_version,
                app_version=data.app_version,
            ),
        )
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
    # New users get the default cloned agent so H5 can go straight to chat.
    await ensure_default_agent_for_user(user.id)
    token = create_jwt(user.id, user.role)
    audit_auth_request_event(
        "wechat_h5_login_success",
        request,
        username=user.username,
        user_id=user.id,
        outcome="success",
    )
    logger.info(
        "User logged in with WeChat H5",
        extra={"event": "auth_wechat_h5_login", "user_id": user.id},
    )
    await _record_auth_activity(user.id, source="wechat_h5_login")
    return await _build_auth_response(user, token)


@router.post("/wechat/profile", response_model=AuthResponse)
async def update_wechat_profile_endpoint(
    data: WeChatProfileUpdate,
    payload: dict = Depends(require_user),
):
    """Persist the Mini Program 头像昵称填写能力 result (nickname + avatar)."""
    user_id = payload["sub"]
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
        )

    avatar_url: str | None = None
    if data.avatar_base64:
        from app.services.chat_media import storage

        mime = storage.normalize_image_mime(data.avatar_mime)
        blob = storage.decode_image_base64(data.avatar_base64)
        storage.validate_image_size(blob)
        storage_key = storage.save_image_blob(user_id=user_id, blob=blob, mime=mime)
        avatar_url = storage.media_url(storage_key)

    await update_wechat_profile(
        user_id,
        nickname=data.nickname,
        avatar_url=avatar_url,
    )
    token = create_jwt(user.id, user.role)
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
    await _record_auth_activity(user.id, source="auth_me")
    return await _build_auth_response(user, token)


@router.post("/heartbeat")
async def heartbeat(payload: dict = Depends(require_user)):
    """轻量在线心跳 — 只刷新统一在线计数, 不查库.

    让 H5/web 前台 (即使不在聊天页) 与原生 App 的前台 presence 心跳同等被计入
    实时在线. 前端在会话有效且页面可见时每 ~40s 调一次 (TTL 90s).
    """
    await record_online(payload["sub"])
    return {"ok": True}
