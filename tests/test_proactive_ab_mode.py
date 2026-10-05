"""《主动聊天机制（新增）》A/B 双模式的接线与修复点.

- 回合收尾统一 arm (主路径 + 所有短路; 边界系统不 arm)
- 判定窗 = AI 最后一句 + 5 分钟
- 状态机 CAS: 用户在生成期间回来, 后续状态写全部落空
- metadata 合并 (记忆冷却跨聊天存活)
- 门槛: A 模式 30 分钟用户活跃互斥不套到 B 模式
- 发送守卫: 生成期间用户回来 → 不插入
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.proactive import gates, state as state_mod
from app.services.proactive.gates import Gate
from app.services.proactive.state import (
    ARM_REASON_FAREWELL,
    ARM_REASON_REPLY,
    ARM_REASON_SYSTEM,
    FOLLOWUP_JUDGE_DELAY_S,
    JUDGE_WINDOW_INDEX,
    short_circuit_arm_reason,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 29, 4, 0, tzinfo=UTC)  # 上海 12:00


_BASE_RECORD = state_mod.ProactiveStateRecord(
    id="st-1", workspace_id="ws-1", user_id="u-1", agent_id="a-1",
    conversation_id="c-1", status="processing", stage="warming",
    silence_level_n=0, followup_plan_type="normal", remaining_forced_triggers=None,
    current_window_index=1, window_due_at=None, response_deadline_at=None,
    t0_at=NOW, last_proactive_at=None, last_user_reply_at=None,
    last_assistant_reply_at=None, last_attempt_at=None,
    daily_scene_triggered_at=None, stop_reason=None, metadata={},
)


def _record(**kw) -> state_mod.ProactiveStateRecord:
    return replace(_BASE_RECORD, **kw)


# ── arm 原因 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("conversation_end", ARM_REASON_FAREWELL),
        # 危机短路之后不主动 (与改造前一致)
        ("crisis_reply", None),
        ("crisis", None),
        # 系统确认类问句挂着 pending 等用户答: 照常 arm, 但不做 B 追问
        ("deletion_delete", ARM_REASON_SYSTEM),
        ("record_request_ask_time", ARM_REASON_SYSTEM),
        ("current_state", "short_circuit:current_state"),
        (None, "short_circuit"),
    ],
)
def test_short_circuit_arm_reason(kind, expected):
    assert short_circuit_arm_reason(kind) == expected


# ── 判定窗 & metadata 合并 ──────────────────────────────────────────────

async def test_arm_opens_judge_window_five_minutes_after_ai_reply(monkeypatch):
    query = AsyncMock(return_value=[{"id": "st-1"}])
    monkeypatch.setattr(state_mod.db, "query_raw", query)
    monkeypatch.setattr(state_mod, "determine_proactive_stage", AsyncMock(return_value="warming"))
    monkeypatch.setattr(state_mod, "log_proactive_event", AsyncMock())

    await state_mod.start_or_restart_proactive_session(
        workspace_id="ws-1", conversation_id="c-1", user_id="u-1", agent_id="a-1",
        now=NOW, reason=ARM_REASON_REPLY, first_window_index=JUDGE_WINDOW_INDEX,
    )

    sql, *params = query.await_args.args
    assert params[6] == JUDGE_WINDOW_INDEX
    assert params[7] == (NOW + timedelta(seconds=FOLLOWUP_JUDGE_DELAY_S)).isoformat()
    # 记忆冷却 (spec §9) 必须跨越多轮聊天: 合并而非覆盖
    assert "COALESCE(proactive_states.metadata, '{}'::jsonb) || EXCLUDED.metadata" in sql


async def test_workspace_activation_still_starts_at_window_one(monkeypatch):
    query = AsyncMock(return_value=[{"id": "st-1"}])
    monkeypatch.setattr(state_mod.db, "query_raw", query)
    monkeypatch.setattr(state_mod, "determine_proactive_stage", AsyncMock(return_value="warming"))
    monkeypatch.setattr(state_mod, "log_proactive_event", AsyncMock())
    await state_mod.start_or_restart_proactive_session(
        workspace_id="ws-1", conversation_id="c-1", user_id="u-1", agent_id="a-1",
        now=NOW, reason="workspace_activated",
    )
    due = datetime.fromisoformat(query.await_args.args[8])
    assert timedelta(minutes=30) <= due - NOW <= timedelta(hours=1)


async def test_arm_resolves_workspace_when_missing(monkeypatch):
    start = AsyncMock()
    monkeypatch.setattr(state_mod, "start_or_restart_proactive_session", start)
    monkeypatch.setattr(
        "app.services.workspace.workspaces.resolve_workspace_id", AsyncMock(return_value="ws-9"),
    )
    await state_mod.arm_after_assistant_turn(
        conversation_id="c-1", user_id="u-1", agent_id="a-1", reason="farewell",
    )
    kwargs = start.await_args.kwargs
    assert kwargs["workspace_id"] == "ws-9"
    assert kwargs["first_window_index"] == JUDGE_WINDOW_INDEX
    assert kwargs["reason"] == "farewell"


async def test_user_reply_notes_session_even_without_proactive_row(monkeypatch):
    note = AsyncMock()
    monkeypatch.setattr(
        "app.services.interaction.topic_continuity.note_user_message", note,
    )
    monkeypatch.setattr(state_mod.db, "query_raw", AsyncMock(return_value=[]))
    await state_mod.mark_user_replied_for_conversation("c-1", replied_at=NOW)
    note.assert_awaited_once_with("c-1", now=NOW)


# ── CAS ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "call",
    [
        lambda s: state_mod.advance_to_next_window(s, now=NOW),
        lambda s: state_mod.advance_to_next_window(s, now=NOW, restart_cycle=True),
        lambda s: state_mod.stop_proactive_state(s, reason="workspace_inactive", now=NOW),
        lambda s: state_mod._escalate_silence_level(s, now=NOW),
    ],
)
async def test_transitions_are_noops_after_user_came_back(monkeypatch, call):
    """用户在 processing 期间发消息 → 行已是 idle → UPDATE 影响 0 行 → 不记事件."""
    execute = AsyncMock(return_value=0)
    log = AsyncMock()
    monkeypatch.setattr(state_mod.db, "execute_raw", execute)
    monkeypatch.setattr(state_mod, "log_proactive_event", log)

    await call(_record())

    sql, *params = execute.await_args.args
    assert "AND status = $" in sql
    assert params[-1] == "processing"
    log.assert_not_awaited()


async def test_mark_sent_logs_even_when_cas_misses(monkeypatch):
    """消息已经发出去了: 状态保持 idle, 但 message_sent 照记 (疲劳度/节律依赖)."""
    monkeypatch.setattr(state_mod.db, "execute_raw", AsyncMock(return_value=0))
    log = AsyncMock()
    monkeypatch.setattr(state_mod, "log_proactive_event", log)
    await state_mod.mark_proactive_sent(
        _record(), trigger_type="silence_wakeup", message="hi", assistant_message_id="m",
        now=NOW,
    )
    assert log.await_args.kwargs["event_type"] == "message_sent"


async def test_restart_cycle_resets_t0_to_now(monkeypatch):
    execute = AsyncMock(return_value=1)
    monkeypatch.setattr(state_mod.db, "execute_raw", execute)
    monkeypatch.setattr(state_mod, "log_proactive_event", AsyncMock())
    old_t0 = NOW - timedelta(minutes=5)
    await state_mod.advance_to_next_window(
        _record(current_window_index=0, t0_at=old_t0), now=NOW, restart_cycle=True,
    )
    _sql, _id, next_index, due_at, new_t0, *_ = execute.await_args.args
    assert next_index == 1
    assert new_t0 == NOW
    assert timedelta(minutes=30) <= due_at - NOW <= timedelta(hours=1)


async def test_judge_window_advances_to_window_one_on_original_t0(monkeypatch):
    """话题已完结 → window 1 仍按 AI 最后一句的时刻算 (30min-1h)."""
    execute = AsyncMock(return_value=1)
    monkeypatch.setattr(state_mod.db, "execute_raw", execute)
    monkeypatch.setattr(state_mod, "log_proactive_event", AsyncMock())
    t0 = NOW - timedelta(minutes=5)
    await state_mod.advance_to_next_window(_record(current_window_index=0, t0_at=t0), now=NOW)
    _sql, _id, next_index, due_at, *_ = execute.await_args.args
    assert next_index == 1
    assert t0 + timedelta(minutes=30) <= due_at <= t0 + timedelta(hours=1)


# ── 门槛 ───────────────────────────────────────────────────────────────

@pytest.fixture
def open_gates(monkeypatch):
    monkeypatch.setattr(gates, "offline_activity_owns_channel", AsyncMock(return_value=False))
    monkeypatch.setattr(gates, "has_recent_game_activity", AsyncMock(return_value=False))
    monkeypatch.setattr(gates, "has_recent_proactive_or_reminder", AsyncMock(return_value=False))
    monkeypatch.setattr(gates, "_patience_is_normal", AsyncMock(return_value=True))
    monkeypatch.setattr(gates, "_reminder_due_soon", AsyncMock(return_value=False))
    user_active = AsyncMock(return_value=True)
    monkeypatch.setattr(gates, "has_recent_user_activity", user_active)
    return user_active


async def test_followup_gates_ignore_recent_user_activity(open_gates):
    """用户 5 分钟前刚说过话 —— A 模式的 30 分钟活跃互斥绝不能套到 B 上."""
    assert await gates.check_followup_gates(_record(), now=NOW) is None
    open_gates.assert_not_awaited()


@pytest.mark.parametrize(
    ("patch_name", "value", "reason"),
    [
        ("offline_activity_owns_channel", True, "offline_activity_active"),
        ("has_recent_game_activity", True, "game_in_progress"),
        ("has_recent_proactive_or_reminder", True, "cooldown"),
        ("_patience_is_normal", False, "patience_low"),
        # 提醒豁免冷却: 追问后一分钟又响提醒 = 连收两条
        ("_reminder_due_soon", True, "reminder_due_soon"),
    ],
)
async def test_followup_gate_reasons(monkeypatch, open_gates, patch_name, value, reason):
    monkeypatch.setattr(gates, patch_name, AsyncMock(return_value=value))
    assert await gates.check_followup_gates(_record(), now=NOW) == reason


async def test_followup_respects_quiet_hours(open_gates):
    night = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)  # 上海 23:30
    assert await gates.check_followup_gates(_record(), now=night) == "off_hours"


async def test_window_gates_order_and_actions(monkeypatch):
    monkeypatch.setattr(gates.db, "query_raw", AsyncMock(return_value=[{"status": "archived"}]))
    assert await gates.check_window_gates(_record(), now=NOW) == Gate("stop", "workspace_inactive")

    monkeypatch.setattr(gates.db, "query_raw", AsyncMock(return_value=[{"status": "active"}]))
    monkeypatch.setattr(gates, "offline_activity_owns_channel", AsyncMock(return_value=False))
    monkeypatch.setattr(gates, "has_recent_user_activity", AsyncMock(return_value=True))
    gate = await gates.check_window_gates(_record(), now=NOW)
    assert gate == Gate("defer", "recent_user_activity")


async def test_window_gates_off_hours_is_a_miss_with_local_hour(monkeypatch):
    monkeypatch.setattr(gates, "check_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(gates, "offline_activity_owns_channel", AsyncMock(return_value=False))
    monkeypatch.setattr(gates, "has_recent_user_activity", AsyncMock(return_value=False))
    monkeypatch.setattr(gates, "has_recent_proactive_or_reminder", AsyncMock(return_value=False))
    monkeypatch.setattr(gates, "_topic_fatigued", AsyncMock(return_value=False))
    night = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)
    gate = await gates.check_window_gates(_record(), now=night)
    assert gate == Gate("miss", "off_hours", {"local_hour": 23})


async def test_patience_read_failure_allows_followup(monkeypatch):
    monkeypatch.setattr(
        "app.services.interaction.boundary.get_patience", AsyncMock(side_effect=RuntimeError),
    )
    assert await gates._patience_is_normal(_record()) is True


# ── orchestrator 分派 ───────────────────────────────────────────────────

async def test_judge_window_is_dispatched_to_followup(monkeypatch):
    from app.services.proactive import orchestrator as orch

    followup = AsyncMock()
    window_gates = AsyncMock()
    monkeypatch.setattr(orch, "process_followup_window", followup)
    monkeypatch.setattr(orch, "check_window_gates", window_gates)
    await orch._process_due_state(_record(current_window_index=JUDGE_WINDOW_INDEX), now=NOW)
    followup.assert_awaited_once()
    window_gates.assert_not_awaited()


async def test_scan_processes_states_concurrently_but_bounded(monkeypatch):
    from app.services.proactive import orchestrator as orch

    states = [_record(id=f"st-{i}") for i in range(20)]
    in_flight = {"now": 0, "max": 0}

    async def _process(claimed, now=None):
        import asyncio

        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1

    async def _claim(state_id, now=None):
        return next(s for s in states if s.id == state_id)

    monkeypatch.setattr(orch, "reclaim_stale_processing_states", AsyncMock(return_value=0))
    monkeypatch.setattr(orch, "list_due_proactive_states", AsyncMock(return_value=states))
    monkeypatch.setattr(orch, "list_waiting_timeout_states", AsyncMock(return_value=[]))
    monkeypatch.setattr(orch, "claim_due_proactive_state", _claim)
    monkeypatch.setattr(orch, "_process_due_state", _process)
    monkeypatch.setattr(
        orch, "db", SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=None))),
    )
    await orch.scan_proactive_states(now=NOW)
    assert 1 < in_flight["max"] <= orch._SCAN_CONCURRENCY


# ── 回合收尾 ───────────────────────────────────────────────────────────

@pytest.fixture
def lifecycle(monkeypatch):
    from app.services.chat import turn_lifecycle

    fired: list = []
    monkeypatch.setattr(turn_lifecycle, "save_last_reply_timestamp", AsyncMock())
    monkeypatch.setattr(turn_lifecycle, "fire_background", lambda coro: fired.append(coro))
    arm = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(turn_lifecycle, "arm_after_assistant_turn", arm)
    monkeypatch.setattr(turn_lifecycle, "close_session", close)
    return turn_lifecycle, fired, arm, close


async def _drain(fired):
    for coro in fired:
        await coro


async def test_finish_turn_arms_proactive(lifecycle):
    mod, fired, arm, close = lifecycle
    await mod.finish_assistant_turn(
        conversation_id="c-1", agent_id="a-1", user_id="u-1", proactive_reason=ARM_REASON_REPLY,
    )
    await _drain(fired)
    assert arm.await_args.kwargs["reason"] == ARM_REASON_REPLY
    close.assert_not_awaited()


async def test_farewell_closes_topic_session(lifecycle):
    mod, fired, arm, close = lifecycle
    await mod.finish_assistant_turn(
        conversation_id="c-1", agent_id="a-1", user_id="u-1", proactive_reason=ARM_REASON_FAREWELL,
    )
    await _drain(fired)
    arm.assert_awaited_once()
    close.assert_awaited_once_with("c-1")


async def test_boundary_turn_does_not_arm(lifecycle):
    mod, fired, arm, _ = lifecycle
    await mod.finish_assistant_turn(
        conversation_id="c-1", agent_id="a-1", user_id="u-1", proactive_reason=None,
    )
    mod.save_last_reply_timestamp.assert_awaited_once()
    assert fired == []


async def test_short_circuit_reply_finishes_turn_with_reason(monkeypatch):
    """语气词表情 / 问当前状态这类短路回复以前不 arm, 状态永远停在 idle."""
    from app.services.chat import multi_intent

    finish = AsyncMock()
    monkeypatch.setattr(multi_intent, "finish_assistant_turn", finish)
    monkeypatch.setattr(multi_intent, "_fire_background", lambda coro: coro.close())

    async def _save(*_a, **_k):
        return None

    await multi_intent.short_circuit_reply("😄", "c-1", "a-1", "u-1", _save)
    assert finish.await_args.kwargs["proactive_reason"] == "short_circuit"

    await multi_intent.short_circuit_reply(
        "嗯", "c-1", "a-1", "u-1", _save, sub_intent_mode=True,
    )
    assert finish.await_count == 1  # 子意图由父调用收尾


async def test_boundary_replies_opt_out_of_proactive(monkeypatch):
    from app.services.chat import orchestrator

    impl = AsyncMock(return_value=[])
    monkeypatch.setattr(orchestrator, "_short_circuit_reply_impl", impl)
    await orchestrator._boundary_short_circuit_reply("走开", "c-1", "a-1", "u-1")
    assert impl.await_args.kwargs["proactive_reason"] is None
    await orchestrator._system_short_circuit_reply("好的已经帮你删掉啦", "c-1", "a-1", "u-1")
    assert impl.await_args.kwargs["proactive_reason"] == ARM_REASON_SYSTEM
    await orchestrator._short_circuit_reply("😄", "c-1", "a-1", "u-1")
    assert impl.await_args.kwargs["proactive_reason"] == "short_circuit"


def test_preflight_uses_system_reply_variant():
    """pending 流程 (矛盾回答 / 删除确认 / 撤回) 的回复不能触发 B 追问."""
    import inspect

    from app.services.chat import orchestrator

    # Public entry now selects an executor; the legacy phase wiring lives here.
    src = inspect.getsource(orchestrator._stream_legacy_response)
    preflight = src[src.index("preflight_ctx = PreflightCtx("):]
    assert "_system_short_circuit_reply" in preflight[:500]
    boundary = src[src.index("boundary_ctx = BoundaryPhaseCtx("):]
    assert "short_circuit_fn=_boundary_short_circuit_reply" in boundary[:600]


async def test_short_circuit_ctx_maps_kind_to_reason(monkeypatch):
    from app.services.chat import intent_handlers

    captured = {}

    async def _finalize(reply, **kwargs):
        captured.update(kwargs)
        if False:
            yield None

    monkeypatch.setattr(intent_handlers, "finalize_short_circuit", _finalize)
    ctx = intent_handlers.ShortCircuitCtx(
        conversation_id="c-1", agent_id="a-1", user_id="u-1", agent=None,
        reply_context=None, tracer=None, save_replies_fn=AsyncMock(),
        pending_sub_fragments={}, sub_intent_mode=False, reply_index_offset=0,
        cached_patience=100,
    )
    async for _ in ctx.finalize("晚安~", kind="conversation_end"):
        pass
    assert captured["proactive_reason"] == ARM_REASON_FAREWELL
    async for _ in ctx.finalize("抱抱你，我在", kind="crisis"):
        pass
    assert captured["proactive_reason"] is None


# ── 发送守卫 ───────────────────────────────────────────────────────────

async def test_emit_aborts_when_user_replied_during_generation(monkeypatch):
    from app.services.proactive import emit

    query = AsyncMock(return_value=[])  # INSERT ... WHERE NOT EXISTS 没插入
    create = AsyncMock()
    send = AsyncMock()
    monkeypatch.setattr(
        emit, "db", SimpleNamespace(query_raw=query, message=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(emit.manager, "send_to_workspace", send)
    monkeypatch.setattr(
        "app.services.speech_output.policy.should_generate_voice", AsyncMock(return_value=False),
    )

    since = datetime(2026, 9, 29, 4, 0, tzinfo=UTC)
    message_id = await emit.emit_proactive_message(
        conversation_id="c-1", user_id="u-1", agent_id="a-1", workspace_id="ws-1",
        message="对了，你更想去海边还是山里？", trigger_type="followup_unfinished",
        abort_if_user_replied_since=since,
    )

    assert message_id == ""
    create.assert_not_awaited()
    send.assert_not_awaited()
    sql, *params = query.await_args.args
    assert "WHERE NOT EXISTS" in sql and "user_message.role = 'user'" in sql
    assert params[5] == "2026-09-29T04:00:00"  # naive UTC, 与 messages.created_at 同口径


async def test_sender_reports_user_came_back(monkeypatch):
    """A 模式生成期间用户回来 → 不算发送, 也不关会话/不记日限."""
    from app.services.proactive import sender

    prep = sender._SendPrep(conversation_id="c-1", cooldown={}, exclude_memory_ids=set())
    ctx = {"source": "greeting", "stage": "warming", "topic_theme": "问候"}
    monkeypatch.setattr("app.services.runtime_config.bind_agent_context", AsyncMock())
    monkeypatch.setattr(sender, "_check_send_eligibility", AsyncMock(return_value=prep))
    monkeypatch.setattr(sender, "_resolve_topic", AsyncMock(return_value=ctx))
    monkeypatch.setattr(sender, "_attach_trending", AsyncMock(return_value=False))
    monkeypatch.setattr(sender, "_generate_message", AsyncMock(return_value="最近咋样呀"))
    monkeypatch.setattr(
        sender, "_prepare_attachments",
        AsyncMock(return_value=sender._Attachments(extra_metadata={})),
    )
    monkeypatch.setattr(sender, "emit_proactive_message", AsyncMock(return_value=""))
    skip = AsyncMock()
    monkeypatch.setattr(sender, "_log_skip", skip)
    count = AsyncMock()
    monkeypatch.setattr(sender, "increment_proactive_count", count)
    close = AsyncMock()
    monkeypatch.setattr(sender.topic_continuity, "close_session", close)

    class _Trace:
        safe_trace_id = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("app.services.llm.usage_tracker.traced_usage_session", lambda **_: _Trace())

    sent = await sender.generate_and_send_proactive(_record(), trigger_type="silence_wakeup", now=NOW)

    assert sent is False
    assert skip.await_args.args[2] == "user_replied_during_generation"
    count.assert_not_awaited()
    close.assert_not_awaited()


# ── A 模式新来源: 近两天带时间戳的对话 ─────────────────────────────────────

@pytest.mark.parametrize(
    ("delta_days", "text"),
    [(0, "今天"), (1, "昨天"), (2, "前天"), (4, "4天前"), (9, "上周"), (21, "3周前")],
)
def test_relative_day_text(delta_days, text):
    from app.services.proactive.context import relative_day_text

    assert relative_day_text(NOW - timedelta(days=delta_days), NOW) == text


def _mem(mid, content, *, occur=None, said=None, main="生活", sub="日常"):
    return SimpleNamespace(
        id=mid, content=content, occurTime=occur, statementTime=said,
        createdAt=said or NOW, mainCategory=main, subCategory=sub,
    )


async def test_recent_dialogue_is_stamped_and_denoised(monkeypatch):
    from app.services.proactive import context

    rows = [  # newest first, 同 SQL
        {"role": "assistant", "content": "那你周五加油！", "metadata": {},
         "created_at": datetime(2026, 9, 28, 13, 1)},
        {"role": "assistant", "content": "黑棋落子", "metadata": {"kind": "game_status"},
         "created_at": datetime(2026, 9, 28, 13, 0, 30)},
        {"role": "user", "content": "我周五要去面试", "metadata": None,
         "created_at": "2026-09-28T13:00:00"},
    ]
    query = AsyncMock(return_value=rows)
    monkeypatch.setattr(context, "db", SimpleNamespace(query_raw=query))
    text = await context._load_recent_dialogue("ws-1", now=NOW)
    assert text.splitlines() == [
        "[09-28 21:00] 用户: 我周五要去面试",
        "[09-28 21:01] AI: 那你周五加油！",
    ]
    # 窗口 = 近 48 小时
    assert query.await_args.args[2] == (NOW - timedelta(hours=48)).replace(tzinfo=None).isoformat()


async def test_recent_dialogue_needs_the_user_to_have_said_something(monkeypatch):
    """只有 AI 自己的主动消息 → 没有可追问的"对方的事"."""
    from app.services.proactive import context

    rows = [{"role": "assistant", "content": "早呀", "metadata": {"proactive": True},
             "created_at": datetime(2026, 9, 28, 1, 0)}]
    monkeypatch.setattr(context, "db", SimpleNamespace(query_raw=AsyncMock(return_value=rows)))
    assert await context._load_recent_dialogue("ws-1", now=NOW) == ""

    monkeypatch.setattr(
        context, "db", SimpleNamespace(query_raw=AsyncMock(side_effect=RuntimeError("db down"))),
    )
    assert await context._load_recent_dialogue("ws-1", now=NOW) == ""


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"crisis_guard_status": "crisis_followup"},
        {"crisis_guard_status": "released"},
        {"short_circuit_kind": "crisis"},
    ],
)
async def test_recent_dialogue_skips_windows_with_crisis_care(monkeypatch, diagnostics):
    """近两天有过危机照护: 不拿那段对话"轻度延伸", 换别的来源."""
    from app.services.proactive import context

    rows = [
        {"role": "assistant", "content": "我在的，你现在安全吗？",
         "metadata": {"response_diagnostics": diagnostics}, "created_at": datetime(2026, 9, 28, 13, 1)},
        {"role": "user", "content": "……", "metadata": None, "created_at": datetime(2026, 9, 28, 13, 0)},
    ]
    monkeypatch.setattr(context, "db", SimpleNamespace(query_raw=AsyncMock(return_value=rows)))
    assert await context._load_recent_dialogue("ws-1", now=NOW) == ""

    # 普通回复的诊断 (crisis_guard_status = none) 不受影响
    rows[0]["metadata"] = {"response_diagnostics": {"crisis_guard_status": "none"}}
    assert await context._load_recent_dialogue("ws-1", now=NOW) != ""


def _patch_context_deps(monkeypatch, context, memories):
    async def _load(**kwargs):
        return memories.get(kwargs["source"], ([], []))

    monkeypatch.setattr(context, "_load_proactive_memories", _load)
    monkeypatch.setattr(context, "_load_recent_dialogue", AsyncMock(return_value=""))
    monkeypatch.setattr(context, "load_core_memory_strings", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        context, "db",
        SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace()))),
    )
    for name in ("get_cached_schedule", "get_latest_portrait", "_load_recent_context", "load_ai_mood"):
        monkeypatch.setattr(context, name, AsyncMock(return_value=None))
    monkeypatch.setattr(context, "get_topic_intimacy", AsyncMock(return_value=60.0))


async def test_empty_recent_dialogue_falls_back_by_trigger_type(monkeypatch):
    from app.services.proactive import context

    _patch_context_deps(
        monkeypatch, context, {"ai_l1": (["[生活/爱好] 我最近在学做饭"], ["ai-mem"])},
    )
    common = dict(workspace_id="ws-1", user_id="u-1", agent_id="a-1", stage="warming",
                  source="recent_dialogue")
    # 记忆主动 → 回落 AI 自己的 L1 记忆
    ctx = await context.build_proactive_context(trigger_type="memory_proactive", **common)
    assert ctx["source"] == "ai_l1" and ctx["used_memory_ids"] == ["ai-mem"]
    # 沉默唤醒 → 回落打招呼
    ctx = await context.build_proactive_context(trigger_type="silence_wakeup", **common)
    assert ctx["source"] == "greeting" and ctx["proactive_memories"] == []


async def test_recent_dialogue_is_kept_when_present(monkeypatch):
    from app.services.proactive import context

    _patch_context_deps(monkeypatch, context, {})
    monkeypatch.setattr(
        context, "_load_recent_dialogue", AsyncMock(return_value="[09-28 21:00] 用户: 我周五面试"),
    )
    ctx = await context.build_proactive_context(
        workspace_id="ws-1", user_id="u-1", agent_id="a-1",
        trigger_type="memory_proactive", stage="warming", source="recent_dialogue",
    )
    assert ctx["source"] == "recent_dialogue"
    assert ctx["recent_dialogue"] == "[09-28 21:00] 用户: 我周五面试"


def test_recent_dialogue_routes_to_its_prompt():
    from app.services.proactive import sender

    for trigger in ("memory_proactive", "silence_wakeup"):
        assert sender._PROMPT_KEY_BY_SOURCE[(trigger, "recent_dialogue")] == "proactive.recent_dialogue"
    # 不是记忆来源: 对话为空时 context 已经换了来源, sender 不该再按"缺记忆"取消
    assert "recent_dialogue" not in sender._MEMORY_SOURCES


def test_recent_dialogue_prompt_renders_every_placeholder():
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    from app.services.proactive import sender

    prompt = sender._format_prompt(
        "proactive.recent_dialogue",
        {"__tpl": PROMPT_DEFINITION_MAP["proactive.recent_dialogue"].default_text,
         "recent_dialogue": "[09-28 21:00] 用户: 我周五面试"},
        "外向、温暖",
    )
    assert prompt is not None and "{" not in prompt
    assert "[09-28 21:00] 用户: 我周五面试" in prompt and "外向、温暖" in prompt
    assert any(approach in prompt for approach in sender._DIALOGUE_APPROACHES)


def test_time_hints_only_on_things_that_happen():
    """「三周前你说你 28 岁」这种时间感是在翻档案."""
    from app.services.proactive.context import _user_memory_time_hint

    said = NOW - timedelta(days=3)
    assert _user_memory_time_hint(_mem("m", "搬家了", said=said, main="生活"), NOW) == "[3天前聊到]"
    assert _user_memory_time_hint(_mem("m", "最近压力大", said=said, main="情绪"), NOW) == "[3天前聊到]"
    assert _user_memory_time_hint(_mem("m", "28岁", said=said, main="身份"), NOW) == ""
    assert _user_memory_time_hint(_mem("m", "喜欢猫", said=said, main="偏好"), NOW) == ""


async def test_reminder_lookahead_query(monkeypatch):
    find = AsyncMock(return_value=SimpleNamespace(id="t-1"))
    monkeypatch.setattr(gates, "db", SimpleNamespace(timetrigger=SimpleNamespace(find_first=find)))
    assert await gates._reminder_due_soon(_record(), NOW) is True
    where = find.await_args.kwargs["where"]
    assert where["actionType"] == "reminder" and where["isActive"] is True
    assert where["triggerTime"]["lte"] - where["triggerTime"]["gte"] == timedelta(minutes=15)

    monkeypatch.setattr(
        gates, "db",
        SimpleNamespace(timetrigger=SimpleNamespace(find_first=AsyncMock(side_effect=RuntimeError))),
    )
    assert await gates._reminder_due_soon(_record(), NOW) is False


def test_memory_source_distributions_sum_to_one():
    from app.services.proactive.policy import MEMORY_SOURCE_DIST

    for stage, dist in MEMORY_SOURCE_DIST.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-9, stage
    # 冷启动还不熟, 不追问对方的事
    assert "recent_dialogue" not in MEMORY_SOURCE_DIST["p1_cold"]
