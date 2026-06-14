"""Minimal APNs HTTP/2 provider client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jwt

from app.config import settings


class ApnsConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApnsResult:
    ok: bool
    apns_id: str | None = None
    status_code: int | None = None
    reason: str | None = None
    unregister: bool = False


class ApnsClient:
    def __init__(self) -> None:
        self._jwt: str | None = None
        self._jwt_iat = 0

    @property
    def configured(self) -> bool:
        return (
            settings.apns_enabled
            and bool(settings.apns_team_id.strip())
            and bool(settings.apns_key_id.strip())
            and bool(settings.apns_topic.strip())
            and bool(self._private_key())
        )

    @property
    def environment(self) -> str:
        return "sandbox" if settings.apns_use_sandbox else "production"

    @property
    def _base_url(self) -> str:
        if settings.apns_use_sandbox:
            return "https://api.sandbox.push.apple.com"
        return "https://api.push.apple.com"

    def _private_key(self) -> str:
        inline = settings.apns_auth_key.strip()
        if inline:
            return inline.replace("\\n", "\n")
        path = settings.apns_auth_key_path.strip()
        if not path:
            return ""
        return Path(path).read_text(encoding="utf-8")

    def _provider_token(self) -> str:
        now = int(time.time())
        if self._jwt and now - self._jwt_iat < 50 * 60:
            return self._jwt
        key = self._private_key()
        if not key:
            raise ApnsConfigurationError("APNs auth key is not configured")
        self._jwt_iat = now
        self._jwt = jwt.encode(
            {"iss": settings.apns_team_id.strip(), "iat": now},
            key,
            algorithm="ES256",
            headers={"kid": settings.apns_key_id.strip()},
        )
        return self._jwt

    async def send_alert(
        self,
        *,
        token: str,
        title: str,
        body: str,
        payload: dict[str, Any],
        topic: str | None = None,
        collapse_id: str | None = None,
        thread_id: str | None = None,
    ) -> ApnsResult:
        if not self.configured:
            raise ApnsConfigurationError("APNs is disabled or incomplete")

        aps: dict[str, Any] = {
            "alert": {"title": title[:120], "body": body[:180]},
            "sound": "default",
        }
        if thread_id:
            aps["thread-id"] = thread_id[:64]
        body_json = {"aps": aps, **payload}
        headers = {
            "authorization": f"bearer {self._provider_token()}",
            "apns-topic": topic or settings.apns_topic.strip(),
            "apns-push-type": "alert",
            "apns-priority": "10",
        }
        if collapse_id:
            headers["apns-collapse-id"] = collapse_id[:64]

        async with httpx.AsyncClient(http2=True, timeout=8.0) as client:
            response = await client.post(
                f"{self._base_url}/3/device/{token}",
                json=body_json,
                headers=headers,
            )
        if 200 <= response.status_code < 300:
            return ApnsResult(
                ok=True,
                apns_id=response.headers.get("apns-id"),
                status_code=response.status_code,
            )
        reason = ""
        try:
            reason = str(response.json().get("reason") or "")
        except Exception:
            reason = response.text[:200]
        return ApnsResult(
            ok=False,
            apns_id=response.headers.get("apns-id"),
            status_code=response.status_code,
            reason=reason,
            unregister=response.status_code == 410 or reason in {"BadDeviceToken", "Unregistered"},
        )


apns_client = ApnsClient()
