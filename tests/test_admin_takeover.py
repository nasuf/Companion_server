"""Human takeover stays off the model path until a claim writes the Redis flag."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.realtime import ws
from app.services.admin_takeover import service


class _Redis:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.sets = []

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, **kwargs):
        self.values[key] = value
        self.sets.append((key, value))

    def pipeline(self, **kwargs):
        parent = self
        class Pipeline:
            def __init__(self): self.ops=[]
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            def set(self,*args): self.ops.append(args)
            async def execute(self):
                for args in self.ops: await parent.set(*args)
        return Pipeline()


@pytest.mark.asyncio
async def test_missing_flag_checks_authoritative_claim(monkeypatch):
    redis = _Redis()
    queried = AsyncMock(return_value=[])
    monkeypatch.setattr(service, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(service.db, "query_raw", queried)

    assert await service.is_conversation_taken_over("conv-1") is False
    queried.assert_awaited_once()


@pytest.mark.asyncio
async def test_active_flag_is_taken_over(monkeypatch):
    redis = _Redis({service.active_conversation_key("conv-1"): "1"})
    monkeypatch.setattr(service, "get_redis", AsyncMock(return_value=redis))

    assert await service.is_conversation_taken_over("conv-1") is True


@pytest.mark.asyncio
async def test_claim_writes_redis_only_after_the_row_updates(monkeypatch):
    redis = _Redis()
    monkeypatch.setattr(service, "get_redis", AsyncMock(return_value=redis))

    async def query_raw(sql, *args):
        if "SET status = 'active'" in sql:
            return [{
                "id": "t1",
                "userId": "user-1",
                "conversationId": "conv-1",
                "agentId": "agent-1",
                "workspaceId": "ws-1",
            }]
        if "FROM human_takeovers t" in sql:
            return [{
                "id": "t1",
                "userId": "user-1",
                "conversationId": "conv-1",
                "agentId": "agent-1",
                "workspaceId": "ws-1",
                "status": "active",
                "ownerAdminId": "admin-1",
                "username": "new",
                "displayName": "新用户",
                "agentName": "小星",
            }]
        return []

    monkeypatch.setattr(service.db, "query_raw", query_raw)
    detail = await service.claim_takeover("t1", "admin-1")

    assert detail["status"] == "active"
    assert redis.values[service.active_conversation_key("conv-1")] == "1"
    assert redis.values[service.active_user_key("user-1")] == "1"


@pytest.mark.asyncio
async def test_second_claim_does_not_write_a_flag(monkeypatch):
    redis = _Redis()
    monkeypatch.setattr(service, "get_redis", AsyncMock(return_value=redis))

    async def query_raw(sql, *args):
        if "SET status = 'active'" in sql:
            return []
        return [{
            "status": "active",
            "conversationId": "conv-1",
            "ownerDisplayName": "管理员甲",
            "ownerUsername": "admin-a",
        }]

    monkeypatch.setattr(service.db, "query_raw", query_raw)
    with pytest.raises(service.TakeoverClaimError) as caught:
        await service.claim_takeover("t1", "admin-2")

    assert caught.value.code == "already_taken_over"
    assert caught.value.owner_name == "管理员甲"
    assert redis.sets == []


@pytest.mark.asyncio
async def test_flag_lookup_failure_stays_on_the_model_path(monkeypatch, caplog):
    async def redis_down():
        raise ConnectionError("redis down")

    async def db_down(*_args, **_kwargs):
        raise RuntimeError("table missing")

    async def takeover(*_args, **_kwargs):
        raise AssertionError("takeover path")

    async def entered_legacy(_user_id):
        raise RuntimeError("entered-legacy-quota")

    monkeypatch.setattr(service, "get_redis", redis_down)
    monkeypatch.setattr(service.db, "query_raw", db_down)
    monkeypatch.setattr(ws, "_handle_takeover_user_message", takeover)
    monkeypatch.setattr(ws.wallet, "is_vip", entered_legacy)
    socket = MagicMock()
    socket.send_json = AsyncMock()

    assert await service.is_conversation_taken_over("conv-1") is False
    await ws._handle_message(
        socket,
        "conv-1",
        "user-1",
        SimpleNamespace(id="agent-1"),
        "你好",
    )
    assert "entered-legacy-quota" in caplog.text


@pytest.mark.asyncio
async def test_status_overlay_lookup_failure_returns_none(monkeypatch):
    async def redis_down():
        raise ConnectionError("redis down")

    async def db_down(*_args, **_kwargs):
        raise RuntimeError("table missing")

    monkeypatch.setattr(service, "get_redis", redis_down)
    monkeypatch.setattr(service.db, "query_raw", db_down)

    assert await service.display_status_override("conv-1") is None


@pytest.mark.asyncio
async def test_unflagged_message_stays_on_the_legacy_path(monkeypatch, caplog):
    async def not_taken(_conversation_id):
        return False

    async def takeover(*_args, **_kwargs):
        raise AssertionError("takeover path")

    async def entered_legacy(_user_id):
        raise RuntimeError("entered-legacy-quota")

    monkeypatch.setattr(service, "is_conversation_taken_over", not_taken)
    monkeypatch.setattr(ws, "_handle_takeover_user_message", takeover)
    monkeypatch.setattr(ws.wallet, "is_vip", entered_legacy)
    socket = MagicMock()
    socket.send_json = AsyncMock()

    await ws._handle_message(
        socket,
        "conv-1",
        "user-1",
        SimpleNamespace(id="agent-1"),
        "你好",
    )
    assert "entered-legacy-quota" in caplog.text


@pytest.mark.asyncio
async def test_flagged_message_returns_before_the_model(monkeypatch):
    called = {}

    async def taken(_conversation_id):
        return True

    async def takeover(*_args, **_kwargs):
        called["takeover"] = True

    async def entered_legacy(_user_id):
        raise AssertionError("legacy quota")

    monkeypatch.setattr(service, "is_conversation_taken_over", taken)
    monkeypatch.setattr(ws, "_handle_takeover_user_message", takeover)
    monkeypatch.setattr(ws.wallet, "is_vip", entered_legacy)

    await ws._handle_message(
        MagicMock(),
        "conv-1",
        "user-1",
        SimpleNamespace(id="agent-1"),
        "你好",
    )
    assert called == {"takeover": True}


def test_activity_previews_keep_distinct_places():
    from app.services.admin_takeover.activities import public_preview, select_preview_cards

    cards = select_preview_cards([
        {"location_name": "甲", "title": "甲"},
        {"location_name": "甲", "title": "甲的分店"},
        {"location_name": "乙", "title": "乙"},
        {"title": ""},
        {"location_name": "丙", "title": "丙"},
        {"location_name": "丁", "title": "丁"},
        {"location_name": "戊", "title": "戊"},
        {"location_name": "己", "title": "己"},
    ])
    names = [item["location_name"] for item in cards]
    assert len(names) == 5
    assert len(set(names)) == 5
    view = public_preview("abc", {
        "title": "甲", "summary": "看看", "description": "详细",
        "location_name": "甲", "address": "路 1 号", "city": "苏州",
        "suitable": "走走", "category": "公园", "image_urls": ["https://img.example/a.jpg"],
    })
    assert view["preview_id"] == "abc"
    assert view["image_url"] == "https://img.example/a.jpg"


def test_takeover_read_cursor_only_covers_human_messages():
    assert service.takeover_message_read("assistant", {"human_takeover": True, "takeover_sequence": 1}, 5) is None
    assert service.takeover_message_read("user", {"human_takeover": True, "takeover_sequence": 2}, 2) is True
    assert service.takeover_message_read("user", {"human_takeover": True, "takeover_sequence": 3}, 2) is False
    assert service.takeover_message_read("user", {"takeover_sequence": 1}, 5) is None


def test_empty_music_invite_remains_a_valid_quota_turn():
    from app.services.admin_takeover.ingress import quota_request_without_card
    from app.services.runtime.chat_entrypoint import parse_client_chat_input

    card = {
        "version": 1,
        "type": "music_track",
        "title": "歌",
        "subtitle": "歌手",
        "body": "频道",
        "footer": "邀请一起听",
        "accent": "#1f6fff",
        "payload": {
            "intent": "invite",
            "source": "music_page",
            "track": {"id": "t1", "title": "歌"},
        },
    }
    incoming = parse_client_chat_input({
        "message": "",
        "client_id": "music-invite-1",
        "attachments": [],
        "component_card": card,
    })
    original = json.loads(incoming.request.input_json)
    quota_request = quota_request_without_card(incoming.request, original)
    quota_input = json.loads(quota_request.input_json)

    assert incoming.request.fingerprint != quota_request.fingerprint
    assert incoming.request.request_key == quota_request.request_key
    assert json.loads(incoming.request.input_json)["component_card"]["type"] == "music_track"
    assert quota_input["component_card"] is None
    assert quota_input["text"] == "［分享］"
    assert quota_input["attachment_ids"] == []

    written = parse_client_chat_input({
        "message": "一起听这首",
        "client_id": "music-invite-2",
        "attachments": [],
        "component_card": card,
    })
    kept = quota_request_without_card(
        written.request, json.loads(written.request.input_json),
    )
    assert json.loads(kept.input_json)["text"] == "一起听这首"


def test_avatar_urls_prefer_stored_portraits_over_wechat():
    user_url, agent_url = service.avatar_urls_from_row({
        "userAvatarKey": "subject_abc.jpg",
        "wechatAvatarUrl": "http://thirdwx.qlogo.cn/mmopen/a",
        "agentAvatarKey": "companion-female-01",
        "agentAvatarUrl": "/agents/avatar/stale.png",
    })

    assert user_url == "/users/avatar/subject_abc.jpg"
    assert agent_url == "/agents/avatar/companion-female-01.png"


def test_avatar_urls_fall_back_when_local_portraits_are_missing():
    user_url, agent_url = service.avatar_urls_from_row({
        "wechatAvatarUrl": "http://thirdwx.qlogo.cn/mmopen/a",
        "avatarUrl": "https://cdn.example/agent.png",
    })

    assert user_url == "https://thirdwx.qlogo.cn/mmopen/a"
    assert agent_url == "https://cdn.example/agent.png"


@pytest.mark.asyncio
async def test_transcript_notify_reaches_the_owning_admin(monkeypatch):
    from app.services.runtime.ws_manager import manager

    sent = AsyncMock()
    monkeypatch.setattr(
        service.db, "query_raw",
        AsyncMock(return_value=[{"id": "t1", "ownerAdminId": "admin-1"}]),
    )
    monkeypatch.setattr(manager, "send_to_admin", sent)

    await service.notify_takeover_transcript("c1", "m1")

    sent.assert_awaited_once_with("admin-1", "takeover_transcript", {
        "takeover_id": "t1",
        "conversation_id": "c1",
        "message_id": "m1",
    })


@pytest.mark.asyncio
async def test_transcript_notify_skips_when_the_chat_is_not_taken_over(monkeypatch):
    from app.services.runtime.ws_manager import manager

    sent = AsyncMock()
    queried = AsyncMock(return_value=[])
    monkeypatch.setattr(service.db, "query_raw", queried)
    monkeypatch.setattr(manager, "send_to_admin", sent)

    await service.notify_takeover_transcript("c1", "m1")
    await service.notify_takeover_transcript("", "m1")

    queried.assert_awaited_once()
    sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcript_notify_swallows_lookup_failure(monkeypatch):
    async def down(*_args, **_kwargs):
        raise RuntimeError("db")

    monkeypatch.setattr(service.db, "query_raw", down)

    await service.notify_takeover_transcript("c1", "m1")


@pytest.mark.asyncio
async def test_recognition_mark_keeps_the_sql_column_and_notifies(monkeypatch):
    from app.services.offline import repository

    queried = AsyncMock(return_value=[{"conversation_id": "c1"}])
    notified = AsyncMock()
    monkeypatch.setattr(repository.db, "query_raw", queried)
    monkeypatch.setattr(service, "notify_takeover_transcript", notified)

    await repository.mark_message_recognized("m1", "gold")

    sql, message_id, tier = queried.await_args.args
    assert "RETURNING conversation_id" in sql
    assert '"conversationId"' not in sql
    assert message_id == "m1"
    assert tier == "gold"
    notified.assert_awaited_once_with("c1", "m1")


@pytest.mark.asyncio
async def test_activity_emit_notifies_after_the_message_is_stored(monkeypatch):
    from app.services.offline import chat_emit

    notified = AsyncMock()
    monkeypatch.setattr(
        chat_emit, "emit_proactive_message", AsyncMock(return_value="m1"),
    )
    monkeypatch.setattr(chat_emit, "_notify_takeover_transcript", notified)

    result = await chat_emit.emit_assistant(
        conversation_id="c1", user_id="u1", agent_id="a1",
        workspace_id="w1", message="一起去看看",
        real_world_type="activity", source_id="act1",
        trigger_type="offline_activity_recommendation",
    )

    assert result == "m1"
    notified.assert_awaited_once_with("c1", "m1")


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [{"id": "t1"}, {"owner_admin_id": "admin-1"}])
async def test_transcript_notify_missing_owner_or_takeover_never_broadcasts(monkeypatch, row):
    from app.services.runtime.ws_manager import manager
    sent = AsyncMock()
    monkeypatch.setattr(service.db, "query_raw", AsyncMock(return_value=[row]))
    monkeypatch.setattr(manager, "send_to_admin", sent)
    await service.notify_takeover_transcript("c1", "m1")
    sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcript_delivery_failure_does_not_fail_stored_message(monkeypatch):
    from app.services.runtime.ws_manager import manager
    from app.services.offline import chat_emit, repository
    sent = AsyncMock(side_effect=RuntimeError("synthetic websocket outage"))
    monkeypatch.setattr(service.db, "query_raw", AsyncMock(return_value=[{"id": "t1", "owner_admin_id": "admin-1"}]))
    monkeypatch.setattr(manager, "send_to_admin", sent)
    await service.notify_takeover_transcript("c1", "m1")
    sent.assert_awaited_once()
    notify = AsyncMock(side_effect=RuntimeError("synthetic notification outage"))
    monkeypatch.setattr(service, "notify_takeover_transcript", notify)
    await chat_emit._notify_takeover_transcript(None, "m1")
    notify.assert_not_awaited()
    await chat_emit._notify_takeover_transcript("c1", "m1")
    monkeypatch.setattr(repository.db, "query_raw", AsyncMock(return_value=[{"conversation_id": "c1"}]))
    await repository.mark_message_recognized("m1", "gold")
    assert notify.await_count == 2
