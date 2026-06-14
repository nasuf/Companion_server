"""Notification outbox dispatcher."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import settings
from app.db import db
from app.services.notifications.apns import ApnsConfigurationError, apns_client
from app.services.notifications.devices import disable_device_token, list_enabled_apns_devices
from app.services.notifications.presence import is_user_foreground

logger = logging.getLogger(__name__)


def _field(row: Any, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _json_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    data = getattr(value, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


async def dispatch_due_notifications(limit: int | None = None) -> int:
    batch_size = limit or settings.notification_dispatch_batch_size
    await db.execute_raw(
        """
        UPDATE notification_events
        SET status = 'pending', updated_at = CURRENT_TIMESTAMP
        WHERE status = 'dispatching'
          AND updated_at < CURRENT_TIMESTAMP - INTERVAL '5 minutes'
        """
    )
    rows = await db.query_raw(
        """
        SELECT
            id, user_id AS "userId", agent_id AS "agentId",
            workspace_id AS "workspaceId", conversation_id AS "conversationId",
            type, title, body, payload, dedupe_key AS "dedupeKey", attempts
        FROM notification_events
        WHERE status = 'pending'
          AND scheduled_for <= CURRENT_TIMESTAMP
        ORDER BY scheduled_for ASC, created_at ASC
        LIMIT $1
        """,
        batch_size,
    )
    processed = 0
    for row in rows:
        try:
            if not await _claim_event(str(_field(row, "id"))):
                continue
            await _dispatch_one(row)
            processed += 1
        except Exception as e:
            event_id = str(_field(row, "id"))
            logger.warning(f"[PUSH] dispatch failed event={event_id}: {e}")
            await _mark_retry_or_failed(row, f"{type(e).__name__}: {str(e)[:180]}")
    return processed


async def _claim_event(event_id: str) -> bool:
    rows = await db.query_raw(
        """
        UPDATE notification_events
        SET status = 'dispatching', updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND status = 'pending'
        RETURNING id
        """,
        event_id,
    )
    return bool(rows)


async def _dispatch_one(row: Any) -> None:
    event_id = str(_field(row, "id"))
    user_id = str(_field(row, "userId"))
    workspace_id = _field(row, "workspaceId")
    conversation_id = _field(row, "conversationId")

    if await is_user_foreground(
        user_id=user_id,
        workspace_id=str(workspace_id) if workspace_id else None,
        conversation_id=str(conversation_id) if conversation_id else None,
    ):
        await _mark_status(event_id, "suppressed", "app_foreground")
        return

    if not apns_client.configured:
        await _mark_status(event_id, "suppressed", "apns_not_configured")
        return

    devices = await list_enabled_apns_devices(
        user_id=user_id,
        environment=apns_client.environment,
    )
    if not devices:
        await _mark_status(event_id, "suppressed", "no_enabled_devices")
        return

    payload = _json_dict(_field(row, "payload"))
    collapse_id = f"{_field(row, 'type')}:{conversation_id or user_id}"
    thread_id = str(workspace_id or conversation_id or user_id)
    successes = 0
    errors: list[str] = []
    last_apns_id = None
    for device in devices:
        try:
            result = await apns_client.send_alert(
                token=device.token,
                title=str(_field(row, "title")),
                body=str(_field(row, "body")),
                payload=payload,
                topic=device.bundle_id or None,
                collapse_id=collapse_id,
                thread_id=thread_id,
            )
        except ApnsConfigurationError as e:
            await _mark_status(event_id, "suppressed", str(e))
            return
        if result.ok:
            successes += 1
            last_apns_id = result.apns_id
        else:
            errors.append(f"{result.status_code}:{result.reason or 'unknown'}")
            if result.unregister:
                await disable_device_token(device.token)

    if successes:
        await db.execute_raw(
            """
            UPDATE notification_events
            SET status = 'sent',
                provider_message_id = $1,
                attempts = attempts + 1,
                sent_at = CURRENT_TIMESTAMP,
                error = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            """,
            last_apns_id,
            event_id,
        )
        return
    await _mark_retry_or_failed(row, "; ".join(errors)[:240] or "apns_send_failed")


async def _mark_retry_or_failed(row: Any, error: str) -> None:
    event_id = str(_field(row, "id"))
    attempts = int(_field(row, "attempts") or 0) + 1
    status = "failed" if attempts >= settings.notification_max_attempts else "pending"
    await db.execute_raw(
        """
        UPDATE notification_events
        SET status = $1,
            attempts = $2,
            error = $3,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $4
        """,
        status,
        attempts,
        error[:240],
        event_id,
    )


async def _mark_status(event_id: str, status: str, error: str | None) -> None:
    await db.execute_raw(
        """
        UPDATE notification_events
        SET status = $1,
            error = $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $3
        """,
        status,
        error[:240] if error else None,
        event_id,
    )
