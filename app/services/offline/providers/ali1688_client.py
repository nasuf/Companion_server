from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.services.offline.providers.gift_types import GiftProviderError

logger = logging.getLogger(__name__)


class Ali1688Client:
    """1688 开放平台 (gw.open.1688.com) param2 风格签名客户端。

    鉴权三件套：
      - app_key     应用标识（公开）
      - app_secret  签名密钥（保密，仅服务端持有）
      - access_token 通过 OAuth2.0 授权采购账号后获得（会过期，需用 refresh_token 续期）

    签名算法（param2）：
      1. 业务参数 + access_token 组成 map，按 key 的 ASCII 升序排列；
      2. 拼成 "k1v1k2v2..." 串（无分隔符），前缀 URL path
         "param2/{version}/{namespace}/{api_name}/{app_key}"；
      3. HMAC-SHA1(app_secret) 后转十六进制大写，作为 _aop_signature。

    ⚠️ 接口名(namespace/api_name)与请求/响应字段名以你拿到 appkey 后在
    open.1688.com「API 列表」看到的官方文档为准。本客户端封装的是「通用调用 +
    正确签名」，具体接口名通过参数传入，便于按文档微调而不改签名逻辑。
    """

    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        access_token: str = "",
        access_token_getter: Callable[[], Awaitable[str]] | None = None,
        base_url: str = "https://gw.open.1688.com/openapi",
        timeout_s: float = 12.0,
    ) -> None:
        self._app_key = app_key.strip()
        self._app_secret = app_secret.strip()
        self._access_token = access_token.strip()
        self._access_token_getter = access_token_getter
        self._base_url = base_url.strip().rstrip("/")
        self._timeout_s = timeout_s
        if not (self._app_key and self._app_secret):
            raise GiftProviderError("ALI1688_APP_KEY / ALI1688_APP_SECRET 必须配置")
        if not (self._access_token or self._access_token_getter):
            raise GiftProviderError("必须提供 access_token 或 access_token_getter 之一")

    async def _resolve_access_token(self) -> str:
        token = (await self._access_token_getter()) if self._access_token_getter else self._access_token
        token = (token or "").strip()
        if not token:
            raise GiftProviderError("1688 access_token 为空（可能未授权或刷新失败）")
        return token

    def _sign(self, url_path: str, params: dict[str, str]) -> str:
        ordered = "".join(f"{key}{params[key]}" for key in sorted(params))
        message = f"{url_path}{ordered}".encode()
        digest = hmac.new(self._app_secret.encode(), message, hashlib.sha1).hexdigest()
        return digest.upper()

    async def call(
        self,
        *,
        namespace: str,
        api_name: str,
        version: str = "1",
        biz_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url_path = f"param2/{version}/{namespace}/{api_name}/{self._app_key}"
        params: dict[str, str] = {
            key: _to_param(value)
            for key, value in (biz_params or {}).items()
            if value is not None and value != ""
        }
        params["access_token"] = await self._resolve_access_token()
        params["_aop_signature"] = self._sign(url_path, params)

        url = f"{self._base_url}/{url_path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(url, data=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise GiftProviderError(f"1688 接口 {api_name} 调用失败: {exc}") from exc

        if not isinstance(data, dict):
            raise GiftProviderError(f"1688 接口 {api_name} 返回非 JSON 对象")
        # 1688 网关错误统一在 error_code / errorCode 字段
        error_code = data.get("error_code") or data.get("errorCode")
        if error_code:
            message = data.get("error_message") or data.get("errorMessage") or "unknown"
            raise GiftProviderError(f"1688 接口 {api_name} 业务报错 [{error_code}] {message}")
        return data


def _to_param(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)
