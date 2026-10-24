from __future__ import annotations

import logging

import pytest

from app.services.notifications import dispatcher
from app.services.notifications.apns import ApnsClient, ApnsResult
from app.services.notifications.devices import PushDevice
from unittest.mock import AsyncMock


def test_apns_base_url_can_follow_device_environment():
    client = ApnsClient()

    assert client._base_url_for("sandbox") == "https://api.sandbox.push.apple.com"
    assert client._base_url_for("production") == "https://api.push.apple.com"


@pytest.mark.parametrize('kind',['admin_new_user','admin_takeover_message','admin_redis_capacity'])
async def test_admin_alert_is_not_lost_to_foreground_presence_or_collapse(monkeypatch,kind):
    from types import SimpleNamespace
    send=AsyncMock(return_value=ApnsResult(ok=True,apns_id='synthetic'))
    monkeypatch.setattr(dispatcher,'apns_client',SimpleNamespace(configured=True,send_alert=send))
    presence=AsyncMock(return_value=True)
    monkeypatch.setattr(dispatcher,'is_user_foreground',presence)
    monkeypatch.setattr(dispatcher,'list_enabled_apns_devices',AsyncMock(return_value=[PushDevice('d','token','production',None)]))
    monkeypatch.setattr(dispatcher.db,'execute_raw',AsyncMock())
    monkeypatch.setattr(dispatcher.db,'query_raw',AsyncMock(return_value=[{'ok':1}]))
    for ident in ['event-one','event-two']:
        await dispatcher._dispatch_one({'id':ident,'userId':'admin','type':kind,'title':'用户姓名','body':'hello','payload':{}})
    presence.assert_not_awaited()
    assert [call.kwargs['collapse_id'] for call in send.await_args_list]==['event-one','event-two']


async def test_revoked_admin_never_receives_queued_private_chat(monkeypatch):
    monkeypatch.setattr(dispatcher.db,'query_raw',AsyncMock(return_value=[]))
    mark=AsyncMock()
    monkeypatch.setattr(dispatcher,'_mark_status',mark)
    await dispatcher._dispatch_one({'id':'event','userId':'former-admin','type':'admin_takeover_message'})
    mark.assert_awaited_once_with('event','suppressed','admin_access_revoked')


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


@pytest.fixture(autouse=True)
def _no_android_devices_by_default(monkeypatch):
    monkeypatch.setattr(dispatcher, "list_enabled_jpush_devices", AsyncMock(return_value=[]))


async def test_android_sends_without_apns_configuration(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(dispatcher, "apns_client", SimpleNamespace(configured=False))
    monkeypatch.setattr(dispatcher, "is_user_foreground", AsyncMock(return_value=False))
    monkeypatch.setattr(dispatcher, "list_enabled_apns_devices", AsyncMock(return_value=[]))
    monkeypatch.setattr(dispatcher, "list_enabled_jpush_devices", AsyncMock(return_value=[
        PushDevice("android", "registration_123", "production", "com.bansheng.prod", "jpush")]))
    send = AsyncMock(return_value=ApnsResult(ok=True, apns_id="jpush-1"))
    monkeypatch.setattr(dispatcher, "jpush_client", SimpleNamespace(send_alert=send))
    update = AsyncMock()
    monkeypatch.setattr(dispatcher.db, "execute_raw", update)
    await dispatcher._dispatch_one({"id":"event", "userId":"user", "type":"agent_message", "title":"hi", "body":"hello"})
    send.assert_awaited_once()
    assert send.await_args.kwargs["package"] == "com.bansheng.prod"
    assert update.await_args.args[1] == "jpush-1"
    assert update.await_args.args[3] == "jpush"


async def test_unconfigured_ios_does_not_block_android_and_remains_observable(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(dispatcher, 'apns_client', SimpleNamespace(configured=False))
    monkeypatch.setattr(dispatcher, 'is_user_foreground', AsyncMock(return_value=False))
    monkeypatch.setattr(dispatcher, 'list_enabled_apns_devices', AsyncMock(return_value=[
        PushDevice('ios', 'ios-token', 'production', None)]))
    monkeypatch.setattr(dispatcher, 'list_enabled_jpush_devices', AsyncMock(return_value=[
        PushDevice('android', 'rid-12345', 'production', 'com.bansheng.prod', 'jpush')]))
    monkeypatch.setattr(dispatcher, 'jpush_client', SimpleNamespace(
        send_alert=AsyncMock(return_value=ApnsResult(ok=True, apns_id='jpush-1'))))
    update = AsyncMock()
    monkeypatch.setattr(dispatcher.db, 'execute_raw', update)
    await dispatcher._dispatch_one({'id': 'event', 'userId': 'user', 'type': 'agent_message'})
    assert update.await_args.args[2] == 'apns_not_configured'
    assert update.await_args.args[3] == 'jpush'


async def test_failed_android_dispatch_is_retried_instead_of_reported_sent(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(dispatcher, 'is_user_foreground', AsyncMock(return_value=False))
    monkeypatch.setattr(dispatcher, 'list_enabled_apns_devices', AsyncMock(return_value=[]))
    monkeypatch.setattr(dispatcher, 'list_enabled_jpush_devices', AsyncMock(return_value=[
        PushDevice('android', 'rid-12345', 'production', 'com.bansheng.prod', 'jpush')]))
    monkeypatch.setattr(dispatcher, 'jpush_client', SimpleNamespace(
        send_alert=AsyncMock(return_value=ApnsResult(ok=False, reason='jpush_transport_error'))))
    update = AsyncMock()
    monkeypatch.setattr(dispatcher.db, 'execute_raw', update)
    await dispatcher._dispatch_one({'id': 'event', 'userId': 'user', 'type': 'agent_message', 'attempts': 0})
    assert update.await_args.args[1] == 'pending'
    assert 'jpush_transport_error' in update.await_args.args[3]
