"""WS "已读" ack event 测试.

工程扩展 (2026-05): 之前用户发消息到 AI 流式回复中间 1-5s 无任何反馈, 体感
"消息石沉大海". 加 ack event: persist 落库后立刻发, 前端在气泡旁标 ✓✓.

详见 CLAUDE.md §6 偏离表.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.realtime import ws as ws_mod
from app.services.interaction import user_turn_aggregation as turn_mod
from app.services.interaction.user_turn_aggregation import UserMessageAggregationPlan


@pytest.fixture
def fake_ws():
    """Mock WebSocket 捕获 send_json 调用."""
    ws = MagicMock()
    ws.send_json = AsyncMock()
    return ws


def _ack_payloads(ws_mock) -> list[dict]:
    """从 send_json 调用历史里提取所有 type='ack' 的 payload."""
    return [
        call.args[0]
        for call in ws_mock.send_json.call_args_list
        if isinstance(call.args[0], dict) and call.args[0].get("type") == "ack"
    ]


def _aggregation_plan(
    route: str,
    *,
    text: str = "测试消息",
    metadata: dict | None = None,
) -> UserMessageAggregationPlan:
    return UserMessageAggregationPlan(
        route=route,
        agent_id="a1",
        user_id="user-1",
        conversation_id="conv-1",
        text=text,
        metadata=metadata or {"queued": True},
        final_message=text,
        final_context={},
        fallback_message=text,
        fallback_context={},
    )


@pytest.mark.asyncio
async def test_turn_aggregation_bypass_for_record_requests():
    """提醒/记忆类控制消息不等待 turn quiet window。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value=None),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert await turn_mod.should_bypass_user_turn_aggregation("conv-1", "明天提醒我交报告")


@pytest.mark.asyncio
async def test_turn_aggregation_keeps_current_state_queries_in_turn_window():
    """询问 AI 当前状态仍是聊天回合的一部分, 需要等待短 turn window 合并追问。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value=None),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert not await turn_mod.should_bypass_user_turn_aggregation("conv-1", "你现在在干嘛")


@pytest.mark.asyncio
async def test_turn_aggregation_keeps_schedule_queries_in_turn_window():
    """计划查询是只读聊天意图, 连续追问应先聚合, 避免重复播报日程。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value=None),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert not await turn_mod.should_bypass_user_turn_aggregation("conv-1", "你明天忙吗")


@pytest.mark.asyncio
async def test_turn_aggregation_bypass_for_pending_confirmations():
    """删除/矛盾等跨消息确认状态下, 下一句需要立即进入 preflight。"""
    with (
        patch.object(turn_mod, "is_crisis_message", return_value=False),
        patch.object(turn_mod, "check_banned_keywords", return_value=[]),
        patch.object(turn_mod, "load_pending_contradiction",
                     new_callable=AsyncMock, return_value={"id": "pending"}),
        patch.object(turn_mod, "load_pending_action",
                     new_callable=AsyncMock, return_value=None),
    ):
        assert await turn_mod.should_bypass_user_turn_aggregation("conv-1", "对，更新吧")


@pytest.mark.asyncio
async def test_send_ack_happy_path(fake_ws):
    """_send_ack 直接调用 → ws 发出 type=ack envelope, 含 client_id +
    message_id + ISO received_at."""
    await ws_mod._send_ack(fake_ws, message_id="db-123", client_id="client-uuid-456")

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1
    data = acks[0]["data"]
    assert data["message_id"] == "db-123"
    assert data["client_id"] == "client-uuid-456"
    assert "received_at" in data
    # ISO format check
    assert "T" in data["received_at"]


@pytest.mark.asyncio
async def test_send_ack_without_client_id(fake_ws):
    """前端没传 client_id (旧客户端兼容) → ack 仍发出, client_id=None.
    前端可 fallback 按时间顺序匹配气泡."""
    await ws_mod._send_ack(fake_ws, message_id="db-789", client_id=None)

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1
    assert acks[0]["data"]["message_id"] == "db-789"
    assert acks[0]["data"]["client_id"] is None


@pytest.mark.asyncio
async def test_send_ack_failure_does_not_raise(fake_ws):
    """ws 已断 / send 抛 → ack 失败不影响主流程 (主路径 reply 仍会发).
    日志 WARN 但不冒泡异常."""
    fake_ws.send_json = AsyncMock(side_effect=ConnectionError("WS closed"))

    # 不抛 — _send_ack 内部 try/except
    await ws_mod._send_ack(fake_ws, message_id="db-1", client_id="c-1")


@pytest.mark.asyncio
async def test_handle_message_fragment_sends_ack_with_client_id(fake_ws):
    """碎片分支 (短消息聚合): persist → 立刻 send_ack (含 client_id) → 然后
    push_pending → 发 pending:aggregating. 顺序很重要 (ack 必须在 pending 前)."""
    agent = SimpleNamespace(id="a1", name="A")
    captured_ack_msg_id = None

    async def _fake_persist(*args, **kwargs):
        return "db-msg-fragment-1"

    plan = _aggregation_plan(
        "fragment_window",
        text="嗯",
        metadata={"fragment": True},
    )

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True),
        # 非空 schedule 让 _handle_message 跳过 generate_daily_schedule (后者
        # 会真的调 redis, 测试不该接触外部资源)
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "嗯",
            client_id="client-frag-uuid",
        )

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1, f"碎片分支必须发 1 个 ack; got {acks}"
    assert acks[0]["data"]["client_id"] == "client-frag-uuid"
    assert acks[0]["data"]["message_id"] == "db-msg-fragment-1"

    # ack 必须在 pending:aggregating 之前发 — 用户体感"已读 → 处理中"
    all_payloads = [
        call.args[0] for call in fake_ws.send_json.call_args_list
        if isinstance(call.args[0], dict)
    ]
    types = [p.get("type") for p in all_payloads]
    ack_idx = types.index("ack")
    pending_idx = types.index("pending")
    assert ack_idx < pending_idx, (
        f"ack 必须在 pending 前发, got types order: {types}"
    )


@pytest.mark.asyncio
async def test_handle_message_fragment_joins_open_turn_window(fake_ws):
    """普通消息后紧接 1-2 字碎片时, 碎片应追加到 turn window, 不另起 5s fragment window。"""
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-fragment-join-1"

    plan = _aggregation_plan("turn_window", text="吗")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True) as enqueue_plan,
        patch.object(ws_mod, "_queue_reply", new_callable=AsyncMock) as queue_reply,
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={"delay_seconds": 0}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "吗",
            client_id="client-frag-join-uuid",
        )

    enqueue_plan.assert_awaited_once()
    assert enqueue_plan.await_args.args[0].route == "turn_window"
    queue_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_non_fragment_sends_ack_with_client_id(fake_ws):
    """非碎片分支 (直接入延迟队列): persist → ack → _queue_reply.
    ack 仍带 client_id."""
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-msg-full-1"

    plan = _aggregation_plan("immediate", text="完整一句话哈哈")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "_queue_reply", new_callable=AsyncMock),
        # 非空 schedule 让 _handle_message 跳过 generate_daily_schedule (后者
        # 会真的调 redis, 测试不该接触外部资源)
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "完整一句话哈哈",
            client_id="client-full-uuid",
        )

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1, f"非碎片分支必须发 1 个 ack; got {acks}"
    assert acks[0]["data"]["client_id"] == "client-full-uuid"
    assert acks[0]["data"]["message_id"] == "db-msg-full-1"


@pytest.mark.asyncio
async def test_handle_message_normal_turn_aggregates_before_queue(fake_ws):
    """普通非碎片消息先进入 turn quiet window, 不立刻触发 reply。"""
    agent = SimpleNamespace(id="a1", name="A")
    queue_reply = AsyncMock()

    async def _fake_persist(*args, **kwargs):
        return "db-msg-turn-1"

    plan = _aggregation_plan("turn_window", text="我最近在看一部美剧")

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True) as enqueue_plan,
        patch.object(ws_mod, "_queue_reply", queue_reply),
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={"delay_seconds": 0}),
    ):
        await ws_mod._handle_message(
            fake_ws, "conv-1", "user-1", agent, "我最近在看一部美剧",
            client_id="client-turn-uuid",
        )

    enqueue_plan.assert_awaited_once()
    assert enqueue_plan.await_args.kwargs["message_id"] == "db-msg-turn-1"
    queue_reply.assert_not_awaited()
    payloads = [
        call.args[0] for call in fake_ws.send_json.call_args_list
        if isinstance(call.args[0], dict)
    ]
    assert any(
        p.get("type") == "pending" and p.get("data", {}).get("status") == "aggregating"
        for p in payloads
    )


@pytest.mark.asyncio
async def test_handle_message_default_client_id_none(fake_ws):
    """旧前端不传 client_id (向后兼容): _handle_message 默认 None,
    ack 仍发出 (仅缺 client_id 字段)."""
    agent = SimpleNamespace(id="a1", name="A")

    async def _fake_persist(*args, **kwargs):
        return "db-1"

    plan = _aggregation_plan(
        "fragment_window",
        text="嗯",
        metadata={"fragment": True},
    )

    with (
        patch.object(ws_mod, "_persist_user_message", side_effect=_fake_persist),
        patch.object(ws_mod, "plan_user_message_aggregation",
                     new_callable=AsyncMock, return_value=plan),
        patch.object(ws_mod, "enqueue_planned_user_message",
                     new_callable=AsyncMock, return_value=True),
        # 非空 schedule 让 _handle_message 跳过 generate_daily_schedule (后者
        # 会真的调 redis, 测试不该接触外部资源)
        patch.object(ws_mod, "get_cached_schedule",
                     new_callable=AsyncMock,
                     return_value=[{"activity": "自由时间", "type": "leisure"}]),
        patch.object(ws_mod, "get_current_status",
                     return_value={"activity": "自由时间", "type": "leisure", "status": "idle"}),
        patch.object(ws_mod, "build_reply_timing_context",
                     new_callable=AsyncMock, return_value={}),
    ):
        # 不传 client_id (default=None)
        await ws_mod._handle_message(fake_ws, "conv-1", "user-1", agent, "嗯")

    acks = _ack_payloads(fake_ws)
    assert len(acks) == 1
    assert acks[0]["data"]["client_id"] is None
    assert acks[0]["data"]["message_id"] == "db-1"
