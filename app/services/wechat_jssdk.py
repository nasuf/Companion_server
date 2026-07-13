"""WeChat Official Account JS-SDK ticket cache and page signature generation."""

from __future__ import annotations

import hashlib
import secrets
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import settings
from app.redis_client import get_redis

_TOKEN_ENDPOINT = "https://api.weixin.qq.com/cgi-bin/stable_token"
_TICKET_ENDPOINT = "https://api.weixin.qq.com/cgi-bin/ticket/getticket"
_CACHE_SAFETY_SECONDS = 300


class WeChatJSSDKError(RuntimeError):
    """Configuration or upstream WeChat error safe to translate at the API edge."""


def normalize_and_validate_url(url: str) -> str:
    """Return the exact no-fragment URL that may be signed for an allowed origin."""
    text = (url or "").strip()
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise WeChatJSSDKError("页面地址格式不正确") from exc
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise WeChatJSSDKError("页面地址格式不正确")
    origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    allowed = settings.wechat_jssdk_origins()
    if origin not in allowed:
        raise WeChatJSSDKError("页面域名未获准使用微信扫一扫")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def _cache_key(kind: str, app_id: str) -> str:
    app_hash = hashlib.sha256(app_id.encode("utf-8")).hexdigest()[:16]
    return f"wechat:jssdk:{kind}:{app_hash}"


def _configured_credentials() -> tuple[str, str]:
    app_id = settings.wechat_h5_app_id.strip()
    app_secret = settings.wechat_h5_app_secret.strip()
    if not settings.wechat_login_enabled or not app_id or not app_secret:
        raise WeChatJSSDKError("微信服务号 JS-SDK 尚未配置")
    return app_id, app_secret


async def _access_token(app_id: str, app_secret: str) -> str:
    redis = await get_redis()
    key = _cache_key("access-token", app_id)
    cached = await redis.get(key)
    if cached:
        return str(cached)

    timeout = httpx.Timeout(settings.wechat_oauth_timeout_s, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.post(
            _TOKEN_ENDPOINT,
            json={
                "grant_type": "client_credential",
                "appid": app_id,
                "secret": app_secret,
                "force_refresh": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise WeChatJSSDKError(
            f"获取微信 access_token 失败 ({payload.get('errcode', 'unknown')})"
        )
    expires_in = max(60, int(payload.get("expires_in") or 7200) - _CACHE_SAFETY_SECONDS)
    await redis.set(key, token, ex=expires_in)
    return token


async def _jsapi_ticket(app_id: str, app_secret: str) -> str:
    redis = await get_redis()
    key = _cache_key("ticket", app_id)
    cached = await redis.get(key)
    if cached:
        return str(cached)

    token = await _access_token(app_id, app_secret)
    timeout = httpx.Timeout(settings.wechat_oauth_timeout_s, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.get(
            _TICKET_ENDPOINT,
            params={"access_token": token, "type": "jsapi"},
        )
        response.raise_for_status()
        payload = response.json()
    ticket = payload.get("ticket")
    if payload.get("errcode") not in (None, 0) or not isinstance(ticket, str) or not ticket:
        raise WeChatJSSDKError(
            f"获取微信 jsapi_ticket 失败 ({payload.get('errcode', 'unknown')})"
        )
    expires_in = max(60, int(payload.get("expires_in") or 7200) - _CACHE_SAFETY_SECONDS)
    await redis.set(key, ticket, ex=expires_in)
    return ticket


async def build_config(url: str) -> dict:
    """Build the wx.config fields for one exact, allowlisted page URL."""
    normalized_url = normalize_and_validate_url(url)
    app_id, app_secret = _configured_credentials()
    ticket = await _jsapi_ticket(app_id, app_secret)
    timestamp = int(time.time())
    nonce = secrets.token_hex(16)
    source = (
        f"jsapi_ticket={ticket}&noncestr={nonce}&timestamp={timestamp}"
        f"&url={normalized_url}"
    )
    signature = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return {
        "app_id": app_id,
        "timestamp": timestamp,
        "nonce_str": nonce,
        "signature": signature,
        "url": normalized_url,
    }
