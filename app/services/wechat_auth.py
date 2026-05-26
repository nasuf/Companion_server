from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import HTTPException, status
from prisma.errors import UniqueViolationError

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)

_WECHAT_PROVIDER = "wechat"
_TOKEN_URL = "https://api.weixin.qq.com/sns/oauth2/access_token"


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


def _identity_lookup_where(token: WeChatTokenPayload) -> dict[str, Any]:
    candidates: list[dict[str, str]] = [
        {"providerAccountId": token.provider_account_id},
        {"openid": token.openid},
    ]
    if token.unionid:
        candidates.append({"unionid": token.unionid})
    return {"provider": _WECHAT_PROVIDER, "OR": candidates}


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
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(_TOKEN_URL, params=params)
            response.raise_for_status()
            payload = response.json()
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

    unionid = payload.get("unionid")
    scope = payload.get("scope")
    return WeChatTokenPayload(
        openid=openid,
        unionid=unionid if isinstance(unionid, str) and unionid else None,
        scope=scope if isinstance(scope, str) and scope else None,
        raw={
            "openid": openid,
            "unionid": unionid if isinstance(unionid, str) else None,
            "scope": scope if isinstance(scope, str) else None,
        },
    )


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
                "rawProfile": token.raw,
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
                    "userId": user.id,
                    "provider": _WECHAT_PROVIDER,
                    "providerAccountId": token.provider_account_id,
                    "openid": token.openid,
                    "unionid": token.unionid,
                    "scope": token.scope,
                    "rawProfile": token.raw,
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
                "rawProfile": token.raw,
                "lastLoginAt": datetime.now(UTC),
            },
        )
        return await db.user.find_unique(where={"id": identity.userId})
