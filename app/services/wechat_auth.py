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


async def exchange_wechat_code(code: str) -> WeChatTokenPayload:
    if not _wechat_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="微信登录暂未开放",
        )

    params = {
        "appid": settings.wechat_mobile_app_id.strip(),
        "secret": settings.wechat_mobile_app_secret.strip(),
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


async def find_or_create_wechat_user(token: WeChatTokenPayload):
    identity = await db.authidentity.find_first(where=_identity_lookup_where(token))
    if identity:
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

    username = _provider_username(token.provider_account_id)
    try:
        async with db.tx() as tx:
            user = await tx.user.create(
                data={
                    "username": username,
                    "hashedPassword": None,
                    "role": "user",
                }
            )
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
        identity = await db.authidentity.find_first(where=_identity_lookup_where(token))
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
