from __future__ import annotations

import logging

import pytest

from app.services.notifications import dispatcher
from app.services.notifications.apns import ApnsClient, ApnsResult
from app.services.notifications.devices import PushDevice


def test_apns_base_url_can_follow_device_environment():
    client = ApnsClient()

    assert client._base_url_for("sandbox") == "https://api.sandbox.push.apple.com"
    assert client._base_url_for("production") == "https://api.push.apple.com"


@pytest.mark.asyncio
async def test_dispatch_sends_each_device_to_its_registered_environment(monkeypatch):
    sent_environments: list[str | None] = []

    class _FakeApnsClient:
        configured = True

        async def send_alert(self, **kwargs):
            sent_environments.append(kwargs.get("environment"))
            return ApnsResult(ok=True, apns_id=f"apns-{len(sent_environments)}")

    async def _foreground(**kwargs):
        return False

    async def _devices(**kwargs):
        assert kwargs == {"user_id": "user-1"}
        return [
            PushDevice(
                id="device-sandbox",
                token="sandbox-token",
                environment="sandbox",
                bundle_id=None,
            ),
            PushDevice(
                id="device-production",
                token="production-token",
                environment="production",
                bundle_id=None,
            ),
        ]

    updates: list[tuple] = []

    async def _execute_raw(*args):
        updates.append(args)

    monkeypatch.setattr(dispatcher, "apns_client", _FakeApnsClient())
    monkeypatch.setattr(dispatcher, "is_user_foreground", _foreground)
    monkeypatch.setattr(dispatcher, "list_enabled_apns_devices", _devices)
    monkeypatch.setattr(dispatcher.db, "execute_raw", _execute_raw)

    await dispatcher._dispatch_one(
        {
            "id": "event-1",
            "userId": "user-1",
            "workspaceId": None,
            "conversationId": "conversation-1",
            "type": "agent_message",
            "title": "小芜",
            "body": "你好呀",
            "payload": {"type": "agent_message"},
            "attempts": 0,
        }
    )

    assert sent_environments == ["sandbox", "production"]
    assert updates
    assert updates[-1][-1] == "event-1"


@pytest.mark.asyncio
async def test_dispatch_logs_partial_failure_and_names_the_disabled_bundle(
    monkeypatch, caplog
):
    """One install losing delivery must leave a trace even when another succeeds.

    The dev and prod flavors are separate app records on the same device, so a
    rejected token silently disables pushes for one of them; before this the
    error list was dropped whenever any device succeeded.
    """

    class _FakeApnsClient:
        configured = True

        async def send_alert(self, **kwargs):
            if kwargs.get("topic") == "com.bansheng.dev":
                return ApnsResult(
                    ok=False,
                    status_code=410,
                    reason="Unregistered",
                    unregister=True,
                )
            return ApnsResult(ok=True, apns_id="apns-ok")

    async def _foreground(**kwargs):
        return False

    async def _devices(**kwargs):
        return [
            PushDevice(
                id="device-dev",
                token="dev-token",
                environment="production",
                bundle_id="com.bansheng.dev",
            ),
            PushDevice(
                id="device-prod",
                token="prod-token",
                environment="production",
                bundle_id="com.bansheng.prod",
            ),
        ]

    disabled: list[str] = []

    async def _disable(token: str):
        disabled.append(token)

    async def _execute_raw(*args):
        return None

    monkeypatch.setattr(dispatcher, "apns_client", _FakeApnsClient())
    monkeypatch.setattr(dispatcher, "is_user_foreground", _foreground)
    monkeypatch.setattr(dispatcher, "list_enabled_apns_devices", _devices)
    monkeypatch.setattr(dispatcher, "disable_device_token", _disable)
    monkeypatch.setattr(dispatcher.db, "execute_raw", _execute_raw)

    with caplog.at_level(logging.WARNING, logger="app.services.notifications.dispatcher"):
        await dispatcher._dispatch_one(
            {
                "id": "event-2",
                "userId": "user-1",
                "workspaceId": None,
                "conversationId": "conversation-1",
                "type": "agent_message",
                "title": "小芜",
                "body": "你好呀",
                "payload": {"type": "agent_message"},
                "attempts": 0,
            }
        )

    assert disabled == ["dev-token"]

    messages = " | ".join(record.getMessage() for record in caplog.records)
    assert "partial_failure" in messages
    assert "device_disabled" in messages
    # The bundle id is the point: without it you cannot tell which install lost
    # delivery.
    assert "com.bansheng.dev" in messages
    assert "Unregistered" in messages
