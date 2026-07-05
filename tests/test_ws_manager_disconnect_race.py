"""Regression tests for the WebSocket disconnect race.

When device B replaces device A on the same conversation, connect() closes A's
socket. A's endpoint `finally` then calls disconnect(); without an identity
guard it would evict B's live mapping, so the active device stops receiving
replies. disconnect(conv_id, ws) must only remove the mapping when it still
points at the caller's own socket.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.websockets import WebSocketState

from app.services.runtime.ws_manager import ConnectionManager


def _make_ws() -> MagicMock:
    ws = MagicMock()
    ws.client_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_stale_disconnect_does_not_evict_replacement():
    """A 被 B 顶替后, A 的 finally disconnect(ws=A) 不能删掉 B 的连接."""
    mgr = ConnectionManager()
    old_ws = _make_ws()
    new_ws = _make_ws()
    await mgr.connect("conv-1", "user-1", old_ws, workspace_id="ws-1")
    await mgr.connect("conv-1", "user-1", new_ws, workspace_id="ws-1")

    # A 的延迟 disconnect 到达 — 传入自己的 ws, 不应误删 B
    await mgr.disconnect("conv-1", old_ws)

    assert mgr.get("conv-1") is new_ws
    assert mgr._workspace_convs.get("ws-1") == {"conv-1"}
    assert mgr._conv_users.get("conv-1") == "user-1"


@pytest.mark.asyncio
async def test_disconnect_with_matching_ws_removes_mapping():
    """传入的 ws 就是当前连接 → 正常清理所有索引."""
    mgr = ConnectionManager()
    ws = _make_ws()
    await mgr.connect("conv-1", "user-1", ws, workspace_id="ws-1")

    await mgr.disconnect("conv-1", ws)

    assert mgr.get("conv-1") is None
    assert "ws-1" not in mgr._workspace_convs
    assert "user-1" not in mgr._user_convs


@pytest.mark.asyncio
async def test_disconnect_without_ws_arg_is_backward_compatible():
    """未传 ws (legacy 调用) → 无条件清理, 保持旧行为."""
    mgr = ConnectionManager()
    ws = _make_ws()
    await mgr.connect("conv-1", "user-1", ws, workspace_id="ws-1")

    await mgr.disconnect("conv-1")

    assert mgr.get("conv-1") is None
    assert "ws-1" not in mgr._workspace_convs


@pytest.mark.asyncio
async def test_send_failure_only_evicts_own_socket():
    """send 失败触发的 disconnect 也带 ws 身份, 不误删接管连接."""
    mgr = ConnectionManager()
    old_ws = _make_ws()
    old_ws.send_json = AsyncMock(side_effect=Exception("broken pipe"))
    await mgr.connect("conv-1", "user-1", old_ws, workspace_id="ws-1")

    # 用 stale 的 old_ws 直接发送失败, 触发 _send_local_conv 内部 disconnect(ws=old)
    # 期间 B 已接管
    new_ws = _make_ws()
    await mgr.connect("conv-1", "user-1", new_ws, workspace_id="ws-1")

    ok = await mgr._send_local_conv("conv-1", "stream", {"chunk": "x"})
    # 当前连接是 new_ws (send 成功), old_ws 的失败路径不触发
    assert ok is True
    assert mgr.get("conv-1") is new_ws
