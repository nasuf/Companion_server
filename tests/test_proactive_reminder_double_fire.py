"""Regression: proactive orchestrator + reminder fire 同分钟双发 + reminder
缺 trace_id.

生产 bug 复现 (2026-05-03 14:00:23): 用户设的"下午提醒出去活动" reminder 在
14:00 触发, 同时 proactive orchestrator scheduled_scene 也在 14:00 fire,
用户连收两条相似的"出去走走"消息 + 第二条没 Trace 按钮.

根因:
1. trigger_scan (15s) 跑 reminder fire → emit_proactive_message
2. proactive_orchestrator (1min) 跑 scheduled_scene → emit_proactive_message
3. orchestrator 的 mutex 检查只看 has_recent_user_activity (user 消息), 漏了
   "AI 自己刚发过 proactive 消息" 的场景

修复:
- 加 has_recent_proactive_or_reminder 检查最近 30min 的 proactive 类 AI 消息
- orchestrator scan 检查通过后再 fire, 否则 defer 下窗口
- reminder fire 也通过 traced_usage_session 包住, emit 时带 trace_id, 让
  Trace 按钮可点
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_TZ = timezone.utc


def test_weekly_habit_next_occurrence_uses_selected_weekdays():
    """A Mon/Wed/Fri habit fired on Monday should renew to Wednesday, not +7d."""
    from app.services.proactive import triggers as trig_mod

    current = datetime(2026, 6, 1, 9, 0, tzinfo=_TZ)  # Monday
    next_occur = trig_mod._next_habit_weekday_occurrence(current, [1, 3, 5])

    assert next_occur == datetime(2026, 6, 3, 9, 0, tzinfo=_TZ)


def test_weekly_habit_next_occurrence_wraps_to_next_week():
    """A Mon/Wed/Fri habit fired on Friday should renew to next Monday."""
    from app.services.proactive import triggers as trig_mod

    current = datetime(2026, 6, 5, 9, 0, tzinfo=_TZ)  # Friday
    next_occur = trig_mod._next_habit_weekday_occurrence(current, [1, 3, 5])

    assert next_occur == datetime(2026, 6, 8, 9, 0, tzinfo=_TZ)


# ═══════════════════════════════════════════════════════════════════
# has_recent_proactive_or_reminder
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_has_recent_proactive_returns_true_when_proactive_msg_exists():
    """proactive AI 消息在窗口内 → True (orchestrator 该 defer)."""
    from app.services.proactive import state as state_mod

    async def _fake_query(_sql, *_args):
        return [{"1": 1}]  # 任意非空行 → True

    with patch.object(state_mod, "db") as mock_db:
        mock_db.query_raw = AsyncMock(side_effect=_fake_query)
        result = await state_mod.has_recent_proactive_or_reminder(
            "ws-1", window_minutes=30,
        )
    assert result is True


@pytest.mark.asyncio
async def test_has_recent_proactive_returns_false_when_empty():
    """无 proactive 消息 → False (orchestrator 可继续)."""
    from app.services.proactive import state as state_mod

    async def _fake_query(_sql, *_args):
        return []

    with patch.object(state_mod, "db") as mock_db:
        mock_db.query_raw = AsyncMock(side_effect=_fake_query)
        result = await state_mod.has_recent_proactive_or_reminder("ws-1")
    assert result is False


@pytest.mark.asyncio
async def test_has_recent_proactive_query_filters_by_metadata_proactive_true():
    """SQL where 子句必须按 metadata->>'proactive'=true 过滤, 不能把普通
    chat reply 也算成 proactive."""
    from app.services.proactive import state as state_mod

    captured_sql = []
    captured_args = []

    async def _fake_query(sql, *args):
        captured_sql.append(sql)
        captured_args.append(args)
        return []

    with patch.object(state_mod, "db") as mock_db:
        mock_db.query_raw = AsyncMock(side_effect=_fake_query)
        await state_mod.has_recent_proactive_or_reminder("ws-1")

    assert len(captured_sql) == 1
    sql = captured_sql[0]
    assert "role = 'assistant'" in sql, "必须只看 AI 消息"
    assert "created_at >= $2::timestamp" in sql
    assert "metadata" in sql and "proactive" in sql, (
        "必须按 metadata.proactive 过滤, 防普通 chat reply 误算"
    )
    assert isinstance(captured_args[0][1], str)


@pytest.mark.asyncio
async def test_has_recent_user_activity_query_casts_since_as_timestamp():
    """prisma-py 会把 datetime 参数传成 text, timestamp 比较必须显式 cast."""
    from app.services.proactive import state as state_mod

    captured_sql = []
    captured_args = []
    now = datetime(2026, 5, 3, 14, 1, tzinfo=_TZ)

    async def _fake_query(sql, *args):
        captured_sql.append(sql)
        captured_args.append(args)
        return []

    with patch.object(state_mod, "db") as mock_db:
        mock_db.query_raw = AsyncMock(side_effect=_fake_query)
        await state_mod.has_recent_user_activity("ws-1", now=now)

    # 两条查询: 先看用户消息, 没有再看对局 —— 一起玩游戏也算在场, 而游戏全程
    # 只写 assistant 消息, 只查 messages 会把一局 20 分钟的棋当成"沉默了 20 分钟"。
    assert len(captured_sql) == 2
    assert "role = 'user'" in captured_sql[0]
    assert "FROM game_sessions" in captured_sql[1]
    for sql, args in zip(captured_sql, captured_args, strict=True):
        assert ">= $2::timestamp" in sql, f"缺 cast: {sql}"
        assert isinstance(args[1], str)


@pytest.mark.asyncio
async def test_has_recent_proactive_db_failure_returns_false():
    """DB 挂 → 返 False (兼容现状, 不冒泡 — orchestrator 主流程不该被这个 mutex 阻塞)."""
    from app.services.proactive import state as state_mod

    async def _fake_query(*_args, **_kw):
        raise ConnectionError("db down")

    with patch.object(state_mod, "db") as mock_db:
        mock_db.query_raw = AsyncMock(side_effect=_fake_query)
        result = await state_mod.has_recent_proactive_or_reminder("ws-1")
    assert result is False


# ═══════════════════════════════════════════════════════════════════
# Orchestrator wiring: 检查通过后 defer
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_orchestrator_defers_when_recent_proactive_exists():
    """orchestrator 看到 has_recent_proactive_or_reminder=True → 走 defer 路径
    (event_type='window_deferred', reason='recent_proactive_activity'), 不发新消息.
    生产 bug 直接复现: reminder 14:00:23 fire 后, orchestrator 14:01 scan
    应 defer 不要再发 scheduled_scene."""
    from app.services.proactive import orchestrator as orch_mod

    advance_calls = []

    async def _capture_advance(state, *, now, event_type, payload=None):
        advance_calls.append({"event_type": event_type, "payload": payload})

    fake_state = MagicMock(
        workspace_id="ws-1",
        agent_id="a1",
        user_id="u1",
    )

    with (
        # workspace 检查通过 (mock 整个 db 模块属性, query_raw 是 read-only 不能直接 patch)
        patch.object(orch_mod, "db") as mock_db,
        patch.object(orch_mod, "stop_proactive_state", new_callable=AsyncMock),
        # 关键: 用户活动检查 → False (放过这关), proactive 检查 → True (该被它拦)
        patch.object(orch_mod, "has_recent_user_activity",
                     new_callable=AsyncMock, return_value=False),
        patch.object(orch_mod, "has_recent_proactive_or_reminder",
                     new_callable=AsyncMock, return_value=True),
        patch.object(orch_mod, "advance_to_next_window",
                     side_effect=_capture_advance),
        patch.object(orch_mod, "generate_and_send_proactive",
                     new_callable=AsyncMock) as mock_send,
    ):
        mock_db.query_raw = AsyncMock(return_value=[{"status": "active"}])
        await orch_mod._process_due_state(fake_state, now=datetime(2026, 5, 3, 14, 1, tzinfo=_TZ))

    # 必须 defer, 不发消息
    assert len(advance_calls) == 1, f"应 defer 1 次, got {advance_calls}"
    assert advance_calls[0]["event_type"] == "window_deferred"
    assert advance_calls[0]["payload"]["reason"] == "recent_proactive_activity"
    assert mock_send.call_count == 0, "不该 fire generate_and_send_proactive"


@pytest.mark.asyncio
async def test_sender_blocks_when_fatigue_score_is_high():
    """主动消息发送前应看用户级 fatigue score, 不只看固定日上限."""
    from app.services.proactive import sender as sender_mod

    fake_state = MagicMock(
        id="state-1",
        workspace_id="ws-1",
        agent_id="a1",
        user_id="u1",
        conversation_id="conv-1",
        current_window_index=2,
        stage="warming",
    )

    events = []

    async def _capture_event(**kwargs):
        events.append(kwargs)

    with (
        patch.object(sender_mod, "can_send_proactive",
                     new_callable=AsyncMock, return_value=True),
        patch.object(sender_mod, "get_proactive_fatigue_score",
                     new_callable=AsyncMock,
                     return_value={
                         "score": 0.91,
                         "threshold": 0.85,
                         "block": True,
                         "components": {"today_count": 2},
                     }),
        patch.object(sender_mod, "get_active_workspace_context",
                     new_callable=AsyncMock) as mock_workspace,
        patch.object(sender_mod, "log_proactive_event", side_effect=_capture_event),
    ):
        result = await sender_mod._check_send_eligibility(fake_state, "memory_proactive")

    assert result is None
    mock_workspace.assert_not_called()
    assert events[0]["event_type"] == "send_skipped"
    assert events[0]["payload"]["reason"] == "fatigue_score"
    assert events[0]["payload"]["score"] == 0.91


# ═══════════════════════════════════════════════════════════════════
# Reminder fire 带 trace_id (前端 Trace 按钮可点)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_reminder_fire_passes_trace_id_to_emit():
    """生产 bug 复现 (2026-05-03 14:00): reminder bubble 没 Trace 按钮, 因为
    emit_proactive_message(trace_id=None). 修复: reminder fire 用
    traced_usage_session 包住, 把 tracer.safe_trace_id 传给 emit."""
    from app.services.proactive import triggers as trig_mod
    from types import SimpleNamespace

    now = datetime(2026, 5, 3, 14, 0, 5, tzinfo=_TZ)
    trigger = SimpleNamespace(
        id="t-trace",
        aiAgentId="agent-A",
        userId="u1",
        actionData={"summary": "出去活动", "memory_id": "m1", "recurrence": "once"},
        actionType="reminder",
        triggerTime=now - timedelta(seconds=5),  # 已到点
        lastFired=None,
        repeatRule=None,
        isActive=True,
    )

    captured_emit_kwargs = {}

    async def _capture_emit(**kwargs):
        captured_emit_kwargs.update(kwargs)
        return "msg-id"

    async def _needed(**_):
        return {"state": "needed", "new_time": None, "reason": ""}

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "_fetch_recent_messages_for_reminder",
                     new_callable=AsyncMock, return_value=""),
        patch("app.services.chat.intent_replies.reminder_pre_check",
              side_effect=_needed),
        patch("app.services.chat.intent_replies.reminder_message",
              new_callable=AsyncMock, return_value="该出去走走啦~"),
        patch.object(trig_mod, "resolve_workspace_id",
                     new_callable=AsyncMock, return_value="ws-1"),
        patch("app.services.proactive.state.get_active_workspace_context",
              new_callable=AsyncMock,
              return_value={"conversation_id": "conv-1"}),
        patch("app.services.proactive.emit.emit_proactive_message",
              side_effect=_capture_emit),
        patch.object(trig_mod, "get_redis", new_callable=AsyncMock,
                     return_value=MagicMock(incr=AsyncMock(), expire=AsyncMock())),
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock()
        mock_db.aiagent.find_unique = AsyncMock(
            return_value=SimpleNamespace(id="agent-A", name="A"),
        )
        await trig_mod._handle_reminder_trigger(trigger, now)

    # emit 必须被调用
    assert captured_emit_kwargs, "emit_proactive_message 应被调用"
    # **关键**: trace_id 必须有显式 key (即便 langsmith 关闭返 None, key 也得在 — 表明
    # 调用方意图传 trace, 不是漏传). 跟 sender.py 主动消息路径一致.
    assert "trace_id" in captured_emit_kwargs, (
        "reminder fire 必须传 trace_id= 给 emit_proactive_message, "
        "否则前端 Trace 按钮不显示 (生产 bug 2026-05-03)"
    )


@pytest.mark.asyncio
async def test_system_only_checkin_does_not_generate_ai_chat():
    """打卡页创建但未发聊天的事项只走系统通知/状态刷新, 不调用 AI pre-check、
    不生成主动聊天消息。"""
    from app.services.proactive import triggers as trig_mod
    from types import SimpleNamespace

    now = datetime(2026, 5, 3, 14, 0, 5, tzinfo=_TZ)
    trigger = SimpleNamespace(
        id="t-system-only",
        aiAgentId="agent-A",
        userId="u1",
        actionData={
            "summary": "喝水",
            "memory_id": "m1",
            "recurrence": "once",
            "sent_to_ai": False,
            "conversation_id": "conv-1",
        },
        actionType="reminder",
        triggerTime=now - timedelta(seconds=5),
        lastFired=None,
        repeatRule=None,
        isActive=True,
    )

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "get_redis", new_callable=AsyncMock,
                     return_value=MagicMock(incr=AsyncMock(), expire=AsyncMock())),
        patch.object(trig_mod, "_fetch_recent_messages_for_reminder",
                     new_callable=AsyncMock) as fetch_recent,
        patch("app.services.chat.intent_replies.reminder_pre_check",
              new_callable=AsyncMock) as pre_check,
        patch("app.services.chat.intent_replies.reminder_message",
              new_callable=AsyncMock) as reminder_message,
        patch("app.services.proactive.emit.emit_proactive_message",
              new_callable=AsyncMock) as emit_message,
        patch("app.services.reminder.scheduling.notify_reminder_changed",
              new_callable=AsyncMock) as notify_changed,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock()
        await trig_mod._handle_reminder_trigger(trigger, now)

    fetch_recent.assert_not_awaited()
    pre_check.assert_not_awaited()
    reminder_message.assert_not_awaited()
    emit_message.assert_not_awaited()
    mock_db.timetrigger.update.assert_awaited_once_with(
        where={"id": "t-system-only"},
        data={"isActive": False, "lastFired": now},
    )
    notify_changed.assert_awaited_once_with(
        "conv-1", kind="fired", trigger_id="t-system-only",
    )


@pytest.mark.asyncio
async def test_precheck_completed_without_memory_id_still_deactivates_trigger():
    """LLM pre-check 判定已完成/取消时, 即使旧数据缺 memory_id 也不能继续发提醒。"""
    from app.services.proactive import triggers as trig_mod
    from types import SimpleNamespace

    now = datetime(2026, 5, 3, 14, 0, 5, tzinfo=_TZ)
    trigger = SimpleNamespace(
        id="t-no-memory",
        aiAgentId="agent-A",
        userId="u1",
        actionData={"summary": "喝水", "recurrence": "once", "sent_to_ai": True},
        actionType="reminder",
        triggerTime=now - timedelta(seconds=5),
        lastFired=None,
        repeatRule=None,
        isActive=True,
    )

    with (
        patch.object(trig_mod, "db") as mock_db,
        patch.object(trig_mod, "get_redis", new_callable=AsyncMock,
                     return_value=MagicMock(incr=AsyncMock(), expire=AsyncMock())),
        patch.object(trig_mod, "_fetch_recent_messages_for_reminder",
                     new_callable=AsyncMock, return_value="刚刚已经喝过了"),
        patch("app.services.chat.intent_replies.reminder_pre_check",
              new_callable=AsyncMock,
              return_value={"state": "completed", "reason": "用户已完成"}),
        patch("app.services.chat.intent_replies.reminder_message",
              new_callable=AsyncMock) as reminder_message,
        patch("app.services.proactive.emit.emit_proactive_message",
              new_callable=AsyncMock) as emit_message,
    ):
        mock_db.timetrigger = MagicMock()
        mock_db.timetrigger.update = AsyncMock()
        await trig_mod._handle_reminder_trigger(trigger, now)

    mock_db.timetrigger.update.assert_awaited_once_with(
        where={"id": "t-no-memory"},
        data={"isActive": False, "lastFired": now},
    )
    reminder_message.assert_not_awaited()
    emit_message.assert_not_awaited()
