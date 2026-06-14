"""Device token persistence for remote notifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.db import db


@dataclass(frozen=True)
class PushDevice:
    id: str
    token: str
    environment: str
    bundle_id: str | None


def normalize_apns_token(token: str) -> str:
    return re.sub(r"[^0-9a-fA-F]", "", token or "").lower()


def _field(row: Any, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


async def register_device(
    *,
    user_id: str,
    platform: str,
    token: str,
    environment: str,
    bundle_id: str | None = None,
    device_id: str | None = None,
    app_version: str | None = None,
) -> str:
    normalized_token = normalize_apns_token(token)
    if not normalized_token:
        raise ValueError("empty push token")
    provider = "apns" if platform == "ios" else "unknown"
    env = environment if environment in {"sandbox", "production"} else "sandbox"
    rows = await db.query_raw(
        """
        INSERT INTO push_devices (
            user_id, platform, provider, token, environment, bundle_id,
            device_id, app_version, enabled, disabled_at, last_seen_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, TRUE, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (provider, token) DO UPDATE
        SET user_id = EXCLUDED.user_id,
            platform = EXCLUDED.platform,
            environment = EXCLUDED.environment,
            bundle_id = EXCLUDED.bundle_id,
            device_id = EXCLUDED.device_id,
            app_version = EXCLUDED.app_version,
            enabled = TRUE,
            disabled_at = NULL,
            last_seen_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
        """,
        user_id,
        platform,
        provider,
        normalized_token,
        env,
        bundle_id,
        device_id,
        app_version,
    )
    return str(_field(rows[0], "id")) if rows else ""


async def disable_device_for_user(*, user_id: str, token: str) -> None:
    normalized_token = normalize_apns_token(token)
    if not normalized_token:
        return
    await db.execute_raw(
        """
        UPDATE push_devices
        SET enabled = FALSE, disabled_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1 AND token = $2
        """,
        user_id,
        normalized_token,
    )


async def disable_device_token(token: str) -> None:
    normalized_token = normalize_apns_token(token)
    if not normalized_token:
        return
    await db.execute_raw(
        """
        UPDATE push_devices
        SET enabled = FALSE, disabled_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
        WHERE token = $1
        """,
        normalized_token,
    )


async def list_enabled_apns_devices(
    *,
    user_id: str,
    environment: str,
) -> list[PushDevice]:
    rows = await db.query_raw(
        """
        SELECT id, token, environment, bundle_id AS "bundleId"
        FROM push_devices
        WHERE user_id = $1
          AND platform = 'ios'
          AND provider = 'apns'
          AND enabled = TRUE
          AND environment = $2
        ORDER BY last_seen_at DESC
        """,
        user_id,
        environment,
    )
    return [
        PushDevice(
            id=str(_field(row, "id")),
            token=str(_field(row, "token")),
            environment=str(_field(row, "environment")),
            bundle_id=_field(row, "bundleId"),
        )
        for row in rows
    ]
