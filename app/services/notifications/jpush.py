"""JPush Android transport. Credentials stay on the server, per app package."""
from __future__ import annotations

import json

import httpx

from app.config import settings
from app.services.notifications.apns import ApnsResult


class JPushClient:
    def credentials(self, package: str | None) -> dict | None:
        if settings.jpush_enabled is False or not package:
            return None
        try:
            apps = json.loads(settings.jpush_apps)
            if not isinstance(apps, dict):
                return None
            # Explicit per-package entries override the single-app configuration.
            # An invalid entry fails closed rather than silently using other keys.
            if package in apps:
                value = apps[package]
                if (isinstance(value, dict) and isinstance(value.get("app_key"), str)
                        and isinstance(value.get("master_secret"), str)
                        and value["app_key"].strip() and value["master_secret"].strip()):
                    return {"app_key": value["app_key"].strip(),
                            "master_secret": value["master_secret"].strip()}
                return None
        except (ValueError, TypeError):
            return None
        if package == settings.jiguang_push_package_name.strip():
            app_key = settings.jiguang_push_app_key.strip()
            master_secret = settings.jiguang_push_master_secret.get_secret_value().strip()
            if app_key and master_secret:
                return {"app_key": app_key, "master_secret": master_secret}
        return None

    async def send_alert(self, *, token: str, package: str | None,
                         title: str, body: str, payload: dict) -> ApnsResult:
        credentials = self.credentials(package)
        if not credentials:
            return ApnsResult(ok=False, reason="jpush_not_configured")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.jpush.cn/v3/push",
                    auth=(credentials["app_key"], credentials["master_secret"]),
                    json={
                        "platform": ["android"],
                        "audience": {"registration_id": [token]},
                        "notification": {"android": {
                            "title": title[:120], "alert": body[:180],
                            "extras": {key: value for key, value in payload.items()
                                       if isinstance(value, (str, int, float, bool))},
                        }},
                        "options": {"time_to_live": 3600},
                    },
                )
            data = response.json()
        except (httpx.HTTPError, ValueError):
            # Do not persist HTTP exception strings: they may contain credentials.
            return ApnsResult(ok=False, reason="jpush_transport_error")
        if response.status_code == 200 and isinstance(data, dict) and data.get("msg_id"):
            return ApnsResult(ok=True, apns_id=str(data["msg_id"]), status_code=200)
        error = data.get("error", {}) if isinstance(data, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        # 1011 means no valid audience, not proof of token expiration.
        return ApnsResult(ok=False, status_code=response.status_code,
                          reason=f"jpush_error_{code or 'unknown'}")


jpush_client = JPushClient()
