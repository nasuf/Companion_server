"""ConnectionManager 跨进程 Pub/Sub 测试.

覆盖:
- Fast path (本地命中): 不 publish, 直接 ws.send_json
- Slow path (本地未命中): publish 到 ws:conv:{conv_id} channel
- 订阅者收到 publish 后查本地 _connections 命中则推送 (跨 worker 流)
- workspace 维度路由: 多 agent 用户不跨 agent 广播 (业务正确性)
- Redis 异常降级: subscriber 启动失败 / publish 失败仍工作
- Channel hash tag 命名: ws:conv:{conv_id} 含花括号, 预留 sharded pubsub
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.websockets import WebSocketState

from app.services.runtime.ws_manager import (
    ConnectionManager,
    _conv_channel,
    _parse_channel,
    _workspace_channel,
)


def _make_ws() -> MagicMock:
    """模拟一个已连接的 WebSocket."""
    ws = MagicMock()
    ws.client_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


# ────────────────────────── channel naming ──────────────────────────

def test_conv_channel_uses_hash_tag():
    assert _conv_channel("abc-123") == "ws:conv:{abc-123}"


def test_workspace_channel_uses_hash_tag():
    assert _workspace_channel("ws-xyz") == "ws:workspace:{ws-xyz}"


def test_parse_channel_recovers_kind_and_id():
    assert _parse_channel("ws:conv:{abc}") == ("conv", "abc")
    assert _parse_channel("ws:workspace:{xyz}") == ("workspace", "xyz")
    assert _parse_channel("invalid") is None
    assert _parse_channel("ws:user:{u1}") is None  # user 维度已废弃


# ────────────────────────── fast path / slow path ──────────────────────────

@pytest.mark.asyncio
async def test_send_event_fast_path_local_hit_skips_publish():
    """本地有连接 → 直送 ws.send_json, 不调用 redis.publish."""
    mgr = ConnectionManager()
    ws = _make_ws()
    await mgr.connect("conv-1", "user-1", ws, workspace_id="ws-1")
    fake_redis = AsyncMock()
    with patch("app.services.runtime.ws_manager.get_redis", AsyncMock(return_value=fake_redis)):
        result = await mgr.send_event("conv-1", "delay", {"duration": 5})
    assert result is True
    ws.send_json.assert_awaited_once_with({"type": "delay", "data": {"duration": 5}})
    fake_redis.publish.assert_not_called()


@pytest.mark.asyncio
async def test_send_event_slow_path_local_miss_publishes():
    """本地无连接 → publish 到 ws:conv:{conv_id} 让其他 worker 处理."""
    mgr = ConnectionManager()
    fake_redis = AsyncMock()
    fake_redis.publish = AsyncMock()
    with patch("app.services.runtime.ws_manager.get_redis", AsyncMock(return_value=fake_redis)):
        result = await mgr.send_event("conv-elsewhere", "stream", {"chunk": "abc"})
    assert result is True
    fake_redis.publish.assert_awaited_once()
    args = fake_redis.publish.await_args.args
    assert args[0] == "ws:conv:{conv-elsewhere}"
    payload = json.loads(args[1])
    assert payload == {"type": "stream", "data": {"chunk": "abc"}}


@pytest.mark.asyncio
async def test_publish_redis_exception_returns_false():
    """Redis publish 异常 → 返 False, 不抛."""
    mgr = ConnectionManager()
    broken_redis = AsyncMock()
    broken_redis.publish = AsyncMock(side_effect=Exception("redis down"))
    with patch("app.services.runtime.ws_manager.get_redis", AsyncMock(return_value=broken_redis)):
        result = await mgr.send_event("conv-x", "delay", {"duration": 1})
    assert result is False


# ────────────────────────── subscriber dispatch ──────────────────────────

@pytest.mark.asyncio
async def test_dispatch_local_hits_local_connection_for_conv():
    """收到 publish 后 conv channel 命中本地 → 推送."""
    mgr = ConnectionManager()
    ws = _make_ws()
    await mgr.connect("conv-1", "user-1", ws, workspace_id="ws-1")
    msg = {
        "type": "pmessage",
        "channel": b"ws:conv:{conv-1}",
        "data": json.dumps({"type": "stream", "data": {"chunk": "hi"}}).encode(),
    }
    await mgr._handle_message(msg)
    ws.send_json.assert_awaited_once_with({"type": "stream", "data": {"chunk": "hi"}})


@pytest.mark.asyncio
async def test_dispatch_local_misses_silently_when_conv_not_local():
    """收到 publish 但 conv 不在本地 → 静默 drop, 不重新 publish 防循环."""
    mgr = ConnectionManager()
    msg = {
        "type": "pmessage",
        "channel": b"ws:conv:{not-local}",
        "data": json.dumps({"type": "stream", "data": {}}).encode(),
    }
    # 应该不抛
    await mgr._handle_message(msg)


@pytest.mark.asyncio
async def test_dispatch_workspace_pushes_to_all_conv_in_workspace():
    """workspace channel 命中 → 推送到该 workspace 下所有本地 conv."""
    mgr = ConnectionManager()
    ws_a = _make_ws()
    ws_b = _make_ws()
    await mgr.connect("conv-a", "user-1", ws_a, workspace_id="ws-shared")
    await mgr.connect("conv-b", "user-1", ws_b, workspace_id="ws-shared")
    msg = {
        "type": "pmessage",
        "channel": b"ws:workspace:{ws-shared}",
        "data": json.dumps({"type": "proactive", "data": {"text": "hi"}}).encode(),
    }
    await mgr._handle_message(msg)
    ws_a.send_json.assert_awaited_once()
    ws_b.send_json.assert_awaited_once()


# ────────────────────────── workspace isolation (业务正确性) ──────────────────────────

@pytest.mark.asyncio
async def test_send_to_workspace_isolates_other_agents():
    """用户拥有两个 agent 时, send_to_workspace(A) 不会推送到 agent B 的 conv."""
    mgr = ConnectionManager()
    ws_a = _make_ws()
    ws_b = _make_ws()
    await mgr.connect("conv-a", "user-1", ws_a, workspace_id="ws-A")
    await mgr.connect("conv-b", "user-1", ws_b, workspace_id="ws-B")
    fake_redis = AsyncMock()
    fake_redis.publish = AsyncMock()
    with patch("app.services.runtime.ws_manager.get_redis", AsyncMock(return_value=fake_redis)):
        local_count = await mgr.send_to_workspace("ws-A", "proactive", {"text": "hi"})
    assert local_count == 1
    ws_a.send_json.assert_awaited_once()
    ws_b.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_send_to_workspace_publishes_workspace_channel():
    """send_to_workspace 同时本地 + publish, 让远端 worker 持有的 conv 也能收."""
    mgr = ConnectionManager()
    fake_redis = AsyncMock()
    fake_redis.publish = AsyncMock()
    with patch("app.services.runtime.ws_manager.get_redis", AsyncMock(return_value=fake_redis)):
        await mgr.send_to_workspace("ws-A", "proactive", {"text": "hi"})
    fake_redis.publish.assert_awaited_once()
    assert fake_redis.publish.await_args.args[0] == "ws:workspace:{ws-A}"


@pytest.mark.asyncio
async def test_send_to_workspace_none_falls_back_to_user_dim():
    """workspace_id=None (历史 conv) → fallback 到 send_to_user."""
    mgr = ConnectionManager()
    ws = _make_ws()
    await mgr.connect("conv-legacy", "user-1", ws, workspace_id=None)
    local_count = await mgr.send_to_workspace(None, "proactive", {"user_id": "user-1", "text": "x"})
    assert local_count == 1
    ws.send_json.assert_awaited_once()


# ────────────────────────── connect / disconnect lifecycle ──────────────────────────

@pytest.mark.asyncio
async def test_connect_replaces_existing_with_close_4001():
    """同 conv 第二次 connect → 旧连接被 close(4001)."""
    mgr = ConnectionManager()
    old_ws = _make_ws()
    new_ws = _make_ws()
    await mgr.connect("conv-1", "user-1", old_ws, workspace_id="ws-1")
    await mgr.connect("conv-1", "user-1", new_ws, workspace_id="ws-1")
    old_ws.close.assert_awaited_once()
    assert mgr.get("conv-1") is new_ws


@pytest.mark.asyncio
async def test_disconnect_cleans_workspace_and_user_indexes():
    mgr = ConnectionManager()
    ws = _make_ws()
    await mgr.connect("conv-1", "user-1", ws, workspace_id="ws-1")
    assert mgr._workspace_convs["ws-1"] == {"conv-1"}
    await mgr.disconnect("conv-1")
    assert "ws-1" not in mgr._workspace_convs
    assert "user-1" not in mgr._user_convs


# ────────────────────────── subscriber lifecycle ──────────────────────────

@pytest.mark.asyncio
async def test_start_subscriber_redis_exception_does_not_throw():
    """Redis 挂时 start_subscriber 不抛 (warning 后退化到本地直送)."""
    mgr = ConnectionManager()
    broken_redis = AsyncMock()
    broken_redis.pubsub = MagicMock(side_effect=Exception("redis down"))
    with patch("app.services.runtime.ws_manager.get_redis", AsyncMock(return_value=broken_redis)):
        await mgr.start_subscriber()
        # 让 subscriber loop 跑一轮
        await asyncio.sleep(0.05)
    await mgr.stop_subscriber()
    # 没抛, 没崩 → pass


@pytest.mark.asyncio
async def test_stop_subscriber_idempotent():
    mgr = ConnectionManager()
    await mgr.stop_subscriber()  # 没启动也不抛
    await mgr.stop_subscriber()


@pytest.mark.asyncio
async def test_close_pubsub_safely_calls_aclose():
    """_close_pubsub_safely 必须调用 aclose 释放 Redis 连接 (regression for
    Too many connections cascade)."""
    aclose_calls = 0

    ps = MagicMock()
    async def _punsubscribe():
        pass
    async def _unsubscribe():
        pass
    async def _aclose():
        nonlocal aclose_calls
        aclose_calls += 1
    ps.punsubscribe = _punsubscribe
    ps.unsubscribe = _unsubscribe
    ps.aclose = _aclose

    await ConnectionManager._close_pubsub_safely(ps)
    assert aclose_calls == 1


@pytest.mark.asyncio
async def test_close_pubsub_safely_swallows_exceptions():
    """关闭过程中任何步骤异常都不抛 (best-effort cleanup)."""
    ps = MagicMock()

    async def _raise():
        raise Exception("redis disconnected")

    ps.punsubscribe = _raise
    ps.unsubscribe = _raise
    ps.aclose = _raise
    # 应该不抛
    await ConnectionManager._close_pubsub_safely(ps)


@pytest.mark.asyncio
async def test_close_pubsub_safely_handles_none():
    """pubsub=None 时 (subscribe 还没建好就抛了) 不应崩."""
    await ConnectionManager._close_pubsub_safely(None)


def test_pubsub_client_does_not_inherit_business_socket_timeout():
    """pubsub 用独立 client 不带 socket_timeout: listen() 永久阻塞读, 不会
    每 5s 被业务 pool 的 socket_timeout 打断 (regression for Timeout 循环 bug)."""
    client = ConnectionManager._make_pubsub_client()
    pool_kwargs = client.connection_pool.connection_kwargs
    # 不应该有 socket_timeout (None 或 missing 都行)
    assert pool_kwargs.get("socket_timeout") is None
    # 应启 keepalive + 健康检查
    assert pool_kwargs.get("socket_keepalive") is True
    assert pool_kwargs.get("health_check_interval") == 30


@pytest.mark.asyncio
async def test_close_pubsub_safely_closes_client_too():
    """_close_pubsub_safely(pubsub, client) 必须关 client (不然连接泄漏)."""
    client_aclose = 0
    pubsub_aclose = 0

    ps = MagicMock()
    async def _no_op():
        pass
    async def _ps_close():
        nonlocal pubsub_aclose
        pubsub_aclose += 1
    ps.punsubscribe = _no_op
    ps.unsubscribe = _no_op
    ps.aclose = _ps_close

    cl = MagicMock()
    async def _cl_close():
        nonlocal client_aclose
        client_aclose += 1
    cl.aclose = _cl_close

    await ConnectionManager._close_pubsub_safely(ps, cl)
    assert pubsub_aclose == 1
    assert client_aclose == 1
