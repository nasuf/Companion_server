"""Human takeover stays off the model path until a claim writes the Redis flag."""

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
