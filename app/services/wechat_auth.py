from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import HTTPException, status
from prisma import Json
from prisma.errors import UniqueViolationError

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)

_WECHAT_PROVIDER = "wechat"
_TOKEN_URL = "https://api.weixin.qq.com/sns/oauth2/access_token"
_USERINFO_URL = "https://api.weixin.qq.com/sns/userinfo"
_JSCODE2SESSION_URL = "https://api.weixin.qq.com/sns/jscode2session"


@dataclass(frozen=True)
class WeChatTokenPayload:
    openid: str
    unionid: str | None
    scope: str | None
    raw: dict[str, Any]

    @property
    def provider_account_id(self) -> str:
        return self.unionid or self.openid


@dataclass(frozen=True)
class SignupInfo:
    """Registration-origin metadata persisted once when the user row is created.

    Only applied on the create branch of ``find_or_create_wechat_user``;
    existing users keep their original signup fields untouched. The columns
    answer "where did this account originate", not "last login channel".
    """

    source: str  # wechat_app / wechat_miniprogram / wechat_h5
    platform: str | None = None  # ios / android / harmony / devtools / ...
    os_version: str | None = None
    app_version: str | None = None

    def user_create_fields(self) -> dict[str, str]:
        fields: dict[str, str] = {"signupSource": self.source}
        if self.platform:
            fields["signupPlatform"] = self.platform
        if self.os_version:
            fields["signupOsVersion"] = self.os_version
        if self.app_version:
            fields["signupAppVersion"] = self.app_version
        return fields


class WeChatLoginError(Exception):
    """Raised for expected WeChat OAuth failures that are safe to show generically."""


def _hash_for_username(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _provider_username(provider_account_id: str) -> str:
    return f"wx_{_hash_for_username(provider_account_id)}"


def _wechat_configured() -> bool:
    return bool(
        settings.wechat_login_enabled
        and settings.wechat_mobile_app_id.strip()
        and settings.wechat_mobile_app_secret.strip()
    )


def _wechat_mini_configured() -> bool:
    return bool(
        settings.wechat_login_enabled
        and settings.wechat_mini_app_id.strip()
        and settings.wechat_mini_app_secret.strip()
    )


def _identity_lookup_where(token: WeChatTokenPayload) -> dict[str, Any]:
    candidates: list[dict[str, str]] = [
        {"providerAccountId": token.provider_account_id},
        {"openid": token.openid},
    ]
    if token.unionid:
        candidates.append({"unionid": token.unionid})
    return {"provider": _WECHAT_PROVIDER, "OR": candidates}


async def _find_identity_preferring_union(token: WeChatTokenPayload):
    """Deterministic identity lookup: unionid > providerAccountId > openid.

    Before the Mini Program was bound to the Open Platform, its logins carried
    only an openid, so a duplicate identity/user may exist alongside the mobile
    app's unionid identity. Once binding activates, both rows match an
    unordered OR lookup — which account wins becomes arbitrary, and updating
    the openid-only row's providerAccountId to the unionid collides with the
    mobile row's unique(provider, providerAccountId). Preferring the unionid
    match routes cross-platform logins into the one canonical account.
    """
    if token.unionid:
        found = await db.authidentity.find_first(
            where={"provider": _WECHAT_PROVIDER, "unionid": token.unionid}
        )
        if found:
            return found
    found = await db.authidentity.find_first(
        where={
            "provider": _WECHAT_PROVIDER,
            "providerAccountId": token.provider_account_id,
        }
    )
    if found:
        return found
    return await db.authidentity.find_first(
        where={"provider": _WECHAT_PROVIDER, "openid": token.openid}
    )


def _safe_wechat_profile(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "openid",
        "unionid",
        "nickname",
        "sex",
        "province",
        "city",
        "country",
        "headimgurl",
        "privilege",
    }
    return {key: value for key, value in payload.items() if key in allowed}


async def _fetch_wechat_userinfo(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    openid: str,
) -> dict[str, Any]:
    response = await client.get(
        _USERINFO_URL,
        params={
            "access_token": access_token,
            "openid": openid,
            "lang": "zh_CN",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errcode") is not None:
        logger.info(
            "WeChat userinfo request rejected",
            extra={"event": "wechat_userinfo_rejected", "errcode": payload.get("errcode")},
        )
        return {}
    return _safe_wechat_profile(payload)


def _wechat_h5_configured() -> bool:
    return bool(
        settings.wechat_login_enabled
        and settings.wechat_h5_app_id.strip()
        and settings.wechat_h5_app_secret.strip()
    )


async def exchange_wechat_h5_code(code: str) -> WeChatTokenPayload:
    """Exchange an Official Account web-page OAuth code (公众号网页授权).

    Identical sns/oauth2 + sns/userinfo flow as the mobile app — only the
    credentials differ — so it reuses ``exchange_wechat_code`` and therefore the
    same ``users`` / ``auth_identities`` unionid continuity.
    """
    if not _wechat_h5_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="微信登录暂未开放",
        )
    return await exchange_wechat_code(
        code,
        app_id=settings.wechat_h5_app_id.strip(),
        app_secret=settings.wechat_h5_app_secret.strip(),
    )


async def exchange_wechat_code(
    code: str,
    *,
    app_id: str | None = None,
    app_secret: str | None = None,
) -> WeChatTokenPayload:
    # Default credentials = mobile app (Open Platform OAuth); callers may pass
    # another appid/secret pair (e.g. the Official Account for H5 login).
    if app_id is None or app_secret is None:
        if not _wechat_configured():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="微信登录暂未开放",
            )
        app_id = settings.wechat_mobile_app_id.strip()
        app_secret = settings.wechat_mobile_app_secret.strip()

    params = {
        "appid": app_id,
        "secret": app_secret,
        "code": code,
        "grant_type": "authorization_code",
    }
    timeout = httpx.Timeout(settings.wechat_oauth_timeout_s, connect=3.0)
    profile: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(_TOKEN_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            access_token = payload.get("access_token")
            openid_for_profile = payload.get("openid")
            scope_for_profile = payload.get("scope")
            if (
                isinstance(access_token, str)
                and access_token
                and isinstance(openid_for_profile, str)
                and openid_for_profile
                and isinstance(scope_for_profile, str)
                and "snsapi_userinfo" in scope_for_profile.split(",")
            ):
                try:
                    profile = await _fetch_wechat_userinfo(
                        client,
                        access_token=access_token,
                        openid=openid_for_profile,
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    logger.info(
                        "WeChat userinfo fetch failed",
                        extra={
                            "event": "wechat_userinfo_failed",
                            "reason": type(exc).__name__,
                        },
                    )
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "WeChat code exchange failed",
            extra={"event": "wechat_login_exchange_failed", "reason": type(exc).__name__},
        )
        raise WeChatLoginError("wechat_exchange_failed") from exc

    errcode = payload.get("errcode")
    if errcode is not None:
        logger.info(
            "WeChat rejected login code",
            extra={"event": "wechat_login_rejected", "errcode": errcode},
        )
        raise WeChatLoginError("wechat_rejected_code")

    openid = payload.get("openid")
    if not isinstance(openid, str) or not openid:
        logger.warning("WeChat response missing openid", extra={"event": "wechat_missing_openid"})
        raise WeChatLoginError("wechat_missing_openid")

    unionid = payload.get("unionid") or profile.get("unionid")
    scope = payload.get("scope")
    return WeChatTokenPayload(
        openid=openid,
        unionid=unionid if isinstance(unionid, str) and unionid else None,
        scope=scope if isinstance(scope, str) and scope else None,
        raw={
            "openid": openid,
            "unionid": unionid if isinstance(unionid, str) else None,
            "scope": scope if isinstance(scope, str) else None,
            **profile,
        },
    )


async def exchange_wechat_miniprogram_code(code: str) -> WeChatTokenPayload:
    """Exchange a Mini Program ``wx.login()`` code via ``jscode2session``.

    Returns the same ``WeChatTokenPayload`` shape as the mobile OAuth flow so it
    can reuse ``find_or_create_wechat_user`` and write to the identical
    ``users`` / ``auth_identities`` tables. ``unionid`` is only present when the
    Mini Program is bound to the same WeChat Open Platform account as the mobile
    app; that shared ``unionid`` is what keeps the account (and its
    conversations) continuous across Mini Program and the future app.

    The ``session_key`` returned by WeChat is deliberately NOT persisted: it is a
    sensitive credential only needed for decrypting encrypted payloads, which we
    do not use here.
    """
    if not _wechat_mini_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="微信登录暂未开放",
        )

    params = {
        "appid": settings.wechat_mini_app_id.strip(),
        "secret": settings.wechat_mini_app_secret.strip(),
        "js_code": code,
        "grant_type": "authorization_code",
    }
    timeout = httpx.Timeout(settings.wechat_oauth_timeout_s, connect=3.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(_JSCODE2SESSION_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "WeChat mini code exchange failed",
            extra={"event": "wechat_mini_exchange_failed", "reason": type(exc).__name__},
        )
        raise WeChatLoginError("wechat_exchange_failed") from exc

    errcode = payload.get("errcode")
    if errcode:
        logger.info(
            "WeChat rejected mini login code",
            extra={"event": "wechat_mini_login_rejected", "errcode": errcode},
        )
        raise WeChatLoginError("wechat_rejected_code")

    openid = payload.get("openid")
    if not isinstance(openid, str) or not openid:
        logger.warning(
            "WeChat mini response missing openid",
            extra={"event": "wechat_mini_missing_openid"},
        )
        raise WeChatLoginError("wechat_missing_openid")

    unionid = payload.get("unionid")
    unionid = unionid if isinstance(unionid, str) and unionid else None
    return WeChatTokenPayload(
        openid=openid,
        unionid=unionid,
        scope="miniprogram",
        raw={
            "openid": openid,
            "unionid": unionid,
            "source": "miniprogram",
        },
    )


async def update_wechat_profile(
    user_id: str,
    *,
    nickname: str | None = None,
    avatar_url: str | None = None,
) -> tuple[str | None, str | None]:
    """Merge user-supplied nickname/avatar into the WeChat identity rawProfile.

    Backs the Mini Program "头像昵称填写能力": WeChat no longer returns nickname /
    avatar silently, so the client collects them via chooseAvatar + nickname
    input and we persist them here (into ``nickname`` / ``headimgurl`` so
    ``_build_auth_response`` surfaces them as display name / avatar).

    Returns the resulting ``(display_name, avatar_url)``. No-op for users without
    a WeChat identity (e.g. password accounts).
    """
    identity = await db.authidentity.find_first(
        where={"userId": user_id, "provider": _WECHAT_PROVIDER},
        order={"updatedAt": "desc"},
    )
    if not identity:
        return None, None

    profile = dict(getattr(identity, "rawProfile", None) or {})
    if nickname is not None:
        cleaned = nickname.strip()
        if cleaned:
            profile["nickname"] = cleaned[:64]
    if avatar_url is not None and avatar_url.strip():
        profile["headimgurl"] = avatar_url.strip()

    await db.authidentity.update(
        where={"id": identity.id},
        data={"rawProfile": Json(profile), "updatedAt": datetime.now(UTC)},
    )
    return profile.get("nickname"), profile.get("headimgurl")


def _merged_raw_profile(existing: object, token: WeChatTokenPayload) -> dict[str, Any]:
    """Merge this login's raw payload into the stored profile (never wipe it).

    The Mini Program flow's ``token.raw`` carries only ``openid/unionid/source``;
    replacing ``rawProfile`` wholesale would erase the nickname/headimgurl saved
    via the 头像昵称填写 step, sending returning users back to the profile page
    on every login. Keys the new payload actually provides (e.g. fresh
    nickname/headimgurl from the mobile OAuth userinfo fetch) still win.
    """
    profile = dict(existing) if isinstance(existing, dict) else {}
    for key, value in token.raw.items():
        if value is not None and value != "":
            profile[key] = value
    return profile


async def find_or_create_wechat_user(
    token: WeChatTokenPayload,
    *,
    signup: SignupInfo | None = None,
):
    identity = await _find_identity_preferring_union(token)
    if identity:
        try:
            await db.authidentity.update(
                where={"id": identity.id},
                data={
                    "providerAccountId": token.provider_account_id,
                    "openid": token.openid,
                    "unionid": token.unionid,
                    "scope": token.scope,
                    "rawProfile": Json(
                        _merged_raw_profile(getattr(identity, "rawProfile", None), token)
                    ),
                    "lastLoginAt": datetime.now(UTC),
                },
            )
        except UniqueViolationError:
            # Another identity already owns this providerAccountId (duplicate
            # pre-binding account) — route the login into that canonical row
            # instead of failing the whole login.
            logger.warning(
                "WeChat identity update conflict; falling back to canonical row",
                extra={"event": "wechat_identity_conflict"},
            )
            canonical = await db.authidentity.find_first(
                where={
                    "provider": _WECHAT_PROVIDER,
                    "providerAccountId": token.provider_account_id,
                }
            )
            if canonical:
                return await db.user.find_unique(where={"id": canonical.userId})
            raise
        return await db.user.find_unique(where={"id": identity.userId})

    username = _provider_username(token.provider_account_id)
    user_data: dict[str, Any] = {
        "username": username,
        "hashedPassword": None,
        "role": "user",
    }
    if signup is not None:
        user_data.update(signup.user_create_fields())
    try:
        async with db.tx() as tx:
            user = await tx.user.create(data=user_data)
            await tx.authidentity.create(
                data={
                    "user": {"connect": {"id": user.id}},
                    "provider": _WECHAT_PROVIDER,
                    "providerAccountId": token.provider_account_id,
                    "openid": token.openid,
                    "unionid": token.unionid,
                    "scope": token.scope,
                    "rawProfile": Json(token.raw),
                    "lastLoginAt": datetime.now(UTC),
                }
            )
            return user
    except UniqueViolationError:
        identity = await _find_identity_preferring_union(token)
        if not identity:
            raise
        await db.authidentity.update(
            where={"id": identity.id},
            data={
                "providerAccountId": token.provider_account_id,
                "openid": token.openid,
                "unionid": token.unionid,
                "scope": token.scope,
                "rawProfile": Json(
                    _merged_raw_profile(getattr(identity, "rawProfile", None), token)
                ),
                "lastLoginAt": datetime.now(UTC),
            },
        )
        return await db.user.find_unique(where={"id": identity.userId})
