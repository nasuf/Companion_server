from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import settings
from app.redis_client import get_redis
from app.services.offline.providers.gift_types import GiftProviderError

logger = logging.getLogger(__name__)

_ACCESS_TOKEN_KEY = "ali1688:access_token"
_REFRESH_TOKEN_KEY = "ali1688:refresh_token"
# 1688 OAuth2 刷新接口（http 协议组）：用 refresh_token 换新 access_token。
_REFRESH_ENDPOINT = "https://gw.open.1688.com/openapi/http/1/system.oauth2/getToken/{app_key}"
# access_token 过期前留这么多秒缓冲，避免临界点请求拿到刚过期的 token。
_EXPIRY_BUFFER_S = 300


async def get_access_token() -> str:
    """运行时取当前有效 access_token：Redis 优先（cron 刷新写入），回退 .env 初始值。

    作为 callable 注入 Ali1688Client，使 provider 始终用最新 token，无需重启进程。
    """
    cached = await _redis_get(_ACCESS_TOKEN_KEY)
    return cached or settings.ali1688_access_token


async def refresh_access_token() -> dict[str, Any]:
    """用 refresh_token 换新 access_token 并写回 Redis。失败抛 GiftProviderError。"""
    app_key = settings.ali1688_app_key.strip()
    app_secret = settings.ali1688_app_secret.strip()
    refresh_token = (await _get_refresh_token()).strip()
    if not (app_key and app_secret and refresh_token):
        raise GiftProviderError(
            "刷新 1688 token 需要 ALI1688_APP_KEY/APP_SECRET/REFRESH_TOKEN 都已配置"
        )

    url = _REFRESH_ENDPOINT.format(app_key=app_key)
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": app_key,
        "client_secret": app_secret,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise GiftProviderError(f"1688 token 刷新请求失败: {exc}") from exc

    if not isinstance(data, dict):
        raise GiftProviderError("1688 token 刷新响应非 JSON 对象")
    access_token = str(data.get("access_token") or "")
    if not access_token:
        error = data.get("error_message") or data.get("error") or data
        raise GiftProviderError(f"1688 token 刷新失败: {error}")

    new_refresh = str(data.get("refresh_token") or refresh_token)
    expires_in = _to_int(data.get("expires_in"), 86400)  # 默认按 1 天

    # Redis 写失败不致命：本次换到的 token 已是最新，下次 cron 会重试。
    await _redis_set(_ACCESS_TOKEN_KEY, access_token, ttl_s=max(_EXPIRY_BUFFER_S, expires_in - _EXPIRY_BUFFER_S))
    await _redis_set(_REFRESH_TOKEN_KEY, new_refresh, ttl_s=None)
    logger.info("[ali1688-token] access_token 刷新成功 expires_in=%ss", expires_in)
    return {"refreshed": True, "expires_in": expires_in}


async def _get_refresh_token() -> str:
    cached = await _redis_get(_REFRESH_TOKEN_KEY)
    return cached or settings.ali1688_refresh_token


async def _redis_get(key: str) -> str | None:
    try:
        client = await get_redis()
        return await client.get(key)
    except Exception as exc:  # noqa: BLE001 — Redis 不可用时回退配置，不阻断
        logger.warning("[ali1688-token] 读 Redis %s 失败，回退配置: %s", key, exc)
        return None


async def _redis_set(key: str, value: str, *, ttl_s: int | None) -> None:
    try:
        client = await get_redis()
        await client.set(key, value, ex=ttl_s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ali1688-token] 写 Redis %s 失败: %s", key, exc)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
