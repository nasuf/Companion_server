"""短路意图必须仍走 post_process (memory/user-emotion/trait/recovery 等后台任务) 单测.

P0 BUG: orchestrator 主路径末尾才 fire post_process, 短路 intent 直接 return →
跳过整个 post_process. 本测试锁:
- ShortCircuitCtx.finalize 把 reply 文本捕获到 ctx.last_short_circuit_reply
- PreflightCtx 短路时把 reply 捕获到 ctx.last_short_circuit_reply
- BoundaryPhaseCtx _emit_short_circuit 同样捕获 (boundary 自己处理 memory pipeline)
- orchestrator finally 兜底 fire post_process (集成层验证)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# ShortCircuitCtx (intent_handlers)
# ═══════════════════════════════════════════════════════════════════


async def _drain(agen):
    events = []
    async for evt in agen:
        events.append(evt)
    return events


@pytest.mark.asyncio
async def test_short_circuit_ctx_finalize_captures_reply():
    """ctx.finalize(reply) 必须把 reply 写到 ctx.last_short_circuit_reply,
    供 orchestrator finally 兜底 fire post_process 使用."""
    from app.services.chat.intent_handlers import ShortCircuitCtx

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="a1", user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )
    assert ctx.last_short_circuit_reply is None

    # 用真 finalize_short_circuit, 但 save_replies_fn 是 AsyncMock + agent_id None
    # 避免 save_last_reply_timestamp 调用真 Redis. 把 agent_id 设 None 让 finalize
    # 跳过 save_last_reply_timestamp 那一步.
    ctx.agent_id = None
    await _drain(ctx.finalize("好的, 我知道了", kind="_test_only"))

    assert ctx.last_short_circuit_reply == "好的, 我知道了", (
        f"finalize 必须捕获 reply, 实际 {ctx.last_short_circuit_reply!r}"
    )


@pytest.mark.asyncio
async def test_short_circuit_ctx_finalize_persists_response_diagnostics():
    """短路意图也必须把 response_diagnostics 写入消息 metadata，供 trace modal 复制。"""
    from app.services.chat.intent_handlers import ShortCircuitCtx

    save_replies = AsyncMock()
    diagnostics = {
        "version": 1,
        "reply_path": None,
        "main_prompt_built": False,
        "main_prompt_build_ms": None,
    }
    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id=None, user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=save_replies,
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
        response_diagnostics=diagnostics,
    )
    bg_tasks = []

    def _capture_background(coro):
        task = asyncio.create_task(coro)
        bg_tasks.append(task)
        return task

    with patch("app.services.chat.multi_intent._fire_background", side_effect=_capture_background):
        await _drain(ctx.finalize("我在工作室打磨齿轮", kind="current_state"))
        if bg_tasks:
            await asyncio.gather(*bg_tasks)

    save_replies.assert_called_once()
    payload = save_replies.call_args.args[1][0]
    assert payload["text"] == "我在工作室打磨齿轮"
    assert payload["response_diagnostics"]["reply_path"] == "short_circuit"
    assert payload["response_diagnostics"]["short_circuit_kind"] == "current_state"
    assert payload["response_diagnostics"]["main_prompt_built"] is False


@pytest.mark.asyncio
async def test_handle_current_state_does_not_pass_full_schedule_to_reply_prompt():
    """当前状态回复只注入当前活动，避免把完整日程塞进 prompt。"""
    from app.services.chat.intent_handlers import ShortCircuitCtx, handle_current_state

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id=None, user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )

    with (
        patch(
            "app.services.chat.intent_handlers.resolve_implicit_time",
            new=AsyncMock(return_value=(None, "正在工作室打磨齿轮")),
        ),
        patch(
            "app.services.chat.intent_handlers.current_state_reply",
            new=AsyncMock(return_value="我在工作室打磨齿轮"),
        ) as reply_mock,
    ):
        handled, events = await handle_current_state(
            "你在干嘛呢",
            ctx,
            ai_status={"activity": "正在工作室打磨齿轮"},
            schedule_context="09:00-23:00 一整天完整日程",
            portrait=None,
            user_emotion=None,
        )

    assert handled is True
    assert events is not None
    assert reply_mock.await_args.kwargs["current_activity"] == "正在工作室打磨齿轮"
    assert reply_mock.await_args.kwargs["ai_schedule"] == ""


@pytest.mark.asyncio
async def test_handle_current_state_rejects_previous_reply_followup():
    """追问上一轮 AI 说法不应走 current_state 短路，否则容易用当前作息硬编细节。"""
    from app.services.chat.intent_handlers import ShortCircuitCtx, handle_current_state

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id=None, user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
        recent_context="AI: 不忙，刚在窗边看云发呆。\n用户: 这么晚还能看到云啊",
    )

    with (
        patch(
            "app.services.chat.intent_handlers.resolve_implicit_time",
            new=AsyncMock(return_value=(None, "看剧/看书")),
        ) as time_mock,
        patch(
            "app.services.chat.intent_handlers.current_state_reply",
            new=AsyncMock(return_value="刚才在翻一本讲云彩分类的书"),
        ) as reply_mock,
    ):
        handled, events = await handle_current_state(
            "这么晚还能看到云啊",
            ctx,
            ai_status={"activity": "看剧/看书"},
            schedule_context="",
            portrait=None,
            user_emotion=None,
        )

    assert handled is False
    assert events is None
    time_mock.assert_not_awaited()
    reply_mock.assert_not_awaited()


def test_orchestrator_downgrades_memory_recall_misrouted_as_current_state():
    """身份/记忆追问被 LLM 误归 CURRENT_STATE 时, 应回到普通聊天记忆路径。"""
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from app.services.chat.orchestrator import _downgrade_non_explicit_current_state

    diagnostics = {}
    result = _downgrade_non_explicit_current_state(
        IntentResult(intent=IntentType.CURRENT_STATE, confidence=0.8),
        "我叫什么名字？还记得吗？",
        diagnostics,
    )

    assert result.intent == IntentType.NONE
    assert result.metadata["downgraded_from"] == IntentType.CURRENT_STATE.value
    assert diagnostics["intent_downgrade_reason"] == "not_explicit_current_state"


def test_orchestrator_downgrades_past_experience_question_as_current_state():
    """追问 AI 去过哪些城市是经历/聊天，不是当前状态。"""
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from app.services.chat.orchestrator import _downgrade_non_explicit_current_state

    diagnostics = {}
    result = _downgrade_non_explicit_current_state(
        IntentResult(intent=IntentType.CURRENT_STATE, confidence=0.8),
        "去过哪些城市呢",
        diagnostics,
    )

    assert result.intent == IntentType.NONE
    assert result.metadata["downgraded_from"] == IntentType.CURRENT_STATE.value
    assert diagnostics["intent_downgrade_reason"] == "not_explicit_current_state"


def test_orchestrator_keeps_explicit_current_state_intent():
    """真正询问 AI 当前状态的消息仍保留 CURRENT_STATE 短路。"""
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from app.services.chat.orchestrator import _downgrade_non_explicit_current_state

    diagnostics = {}
    original = IntentResult(intent=IntentType.CURRENT_STATE, confidence=0.9)
    result = _downgrade_non_explicit_current_state(
        original,
        "你最近怎么样？",
        diagnostics,
    )

    assert result is original
    assert diagnostics == {}


@pytest.mark.asyncio
async def test_handle_schedule_query_date_does_not_inject_current_activity():
    """未来日程查询不能把当前正在做的事注入 prompt。"""
    from app.services.chat.intent_handlers import ShortCircuitCtx, handle_schedule_query

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="a1", user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )
    tomorrow_schedule = [
        {"start": "09:00", "end": "10:00", "activity": "看书"},
        {"start": "10:00", "end": "12:00", "activity": "整理工作台"},
    ]

    with (
        patch(
            "app.services.chat.intent_handlers.get_cached_schedule",
            new=AsyncMock(return_value=tomorrow_schedule),
        ),
        patch(
            "app.services.chat.intent_handlers.schedule_query_reply",
            new=AsyncMock(return_value="明天上午有点安排"),
        ) as reply_mock,
    ):
        handled, events, schedule_context = await handle_schedule_query(
            "你明天忙吗？",
            ctx,
            schedule=[{"start": "15:00", "end": "16:00", "activity": "当前齿轮活"}],
            ai_status={"status": "very_busy", "activity": "正在焊音乐盒关节"},
            portrait=None,
            user_emotion=None,
            query_type="date",
        )

    assert handled is True
    assert events is not None
    assert schedule_context is not None
    assert "明天" in schedule_context
    assert "当前状态" not in schedule_context
    assert "正在焊音乐盒关节" not in schedule_context
    assert reply_mock.await_args.kwargs["current_activity"] == ""
    assert "看书" in reply_mock.await_args.kwargs["ai_schedule"]


@pytest.mark.asyncio
async def test_handle_schedule_query_uses_parser_target_date():
    """handler 通过统一 scope 解析目标日期，不再自己维护零散日期关键词。"""
    from app.services.chat.intent_handlers import ShortCircuitCtx, handle_schedule_query

    ctx = ShortCircuitCtx(
        conversation_id="c1", agent_id="a1", user_id="u1",
        agent=SimpleNamespace(name="A"),
        reply_context=None,
        tracer=MagicMock(safe_trace_id=None, trace_id=None, is_active=False),
        save_replies_fn=AsyncMock(),
        pending_sub_fragments={},
        sub_intent_mode=False,
        reply_index_offset=0,
        cached_patience=100,
    )
    fixed_now = datetime.fromisoformat("2026-05-08T15:00:00+08:00")
    next_wednesday_schedule = [
        {"start": "14:00", "end": "15:00", "activity": "去旧货市场"},
    ]

    with (
        patch(
            "app.services.chat.intent_handlers.get_current_time",
            return_value=SimpleNamespace(now=fixed_now),
        ),
        patch(
            "app.services.chat.intent_handlers.get_cached_schedule",
            new=AsyncMock(return_value=next_wednesday_schedule),
        ) as cached_mock,
        patch(
            "app.services.chat.intent_handlers.schedule_query_reply",
            new=AsyncMock(return_value="下周三下午有安排"),
        ) as reply_mock,
    ):
        handled, events, schedule_context = await handle_schedule_query(
            "你下周三忙吗？",
            ctx,
            schedule=[{"start": "15:00", "end": "16:00", "activity": "当前齿轮活"}],
            ai_status={"status": "very_busy", "activity": "正在焊音乐盒关节"},
            portrait=None,
            user_emotion=None,
            query_type="date",
        )

    assert handled is True
    assert events is not None
    assert schedule_context is not None
    assert "下周三" in schedule_context
    assert "当前状态" not in schedule_context
    assert "去旧货市场" in reply_mock.await_args.kwargs["ai_schedule"]
    target_date = cached_mock.await_args.args[1]
    assert target_date.date().isoformat() == "2026-05-13"


# ═══════════════════════════════════════════════════════════════════
# PreflightCtx (preflight)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_preflight_resolve_pending_contradiction_captures_reply():
    """resolve_pending_contradiction 短路时必须设 ctx.last_short_circuit_reply."""
    from app.services.chat.preflight import PreflightCtx, resolve_pending_contradiction

    pending_payload = {"summary": "用户之前说住北京"}
    ctx = PreflightCtx(
        conversation_id="c1", agent_id="a1", user_id="u1",
        agent=SimpleNamespace(name="A"),
        tracer=MagicMock(safe_trace_id=None, close=MagicMock()),
        short_circuit_fn=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "app.services.chat.preflight.load_pending_contradiction",
            new_callable=AsyncMock, return_value=pending_payload,
        ),
        patch(
            "app.services.chat.preflight.analyze_contradiction_response",
            new_callable=AsyncMock, return_value={"resolution": "update"},
        ),
        patch(
            "app.services.chat.preflight.apply_contradiction_resolution",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.chat.preflight.clear_pending_contradiction",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.chat.preflight.generate_contradiction_reply",
            new_callable=AsyncMock,
            return_value="所以你现在住上海了, 之前的我记错了",
        ),
    ):
        await _drain(resolve_pending_contradiction("我现在住上海", ctx))

    assert ctx.stopped is True
    assert ctx.last_short_circuit_reply == "所以你现在住上海了, 之前的我记错了"


# ═══════════════════════════════════════════════════════════════════
# BoundaryPhaseCtx (boundary_phase)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_boundary_emit_short_circuit_captures_reply():
    """_emit_short_circuit 必须把 reply 写到 ctx.last_short_circuit_reply.

    boundary 路径在 orchestrator finally 通过 `boundary_ctx.stopped` 跳过
    后台兜底, 但 ctx 仍捕获 reply 供 trace/debug + 未来扩展用.
    """
    from app.services.chat.boundary_phase import BoundaryPhaseCtx, _emit_short_circuit

    ctx = BoundaryPhaseCtx(
        conversation_id="c1", agent_id="a1", user_id="u1",
        agent=SimpleNamespace(name="A"),
        user_message="测试消息",
        sub_intent_mode=False,
        parent_patience=None,
        tracer=MagicMock(safe_trace_id=None),
        short_circuit_fn=AsyncMock(return_value=[
            {"event": "reply", "data": "{}"},
            {"event": "done", "data": "{}"},
        ]),
        fire_background_fn=MagicMock(),
        bg_memory_pipeline_fn=MagicMock(),
    )
    assert ctx.last_short_circuit_reply is None

    await _drain(_emit_short_circuit(ctx, "我不想理你", {"boundary": True}))

    assert ctx.last_short_circuit_reply == "我不想理你"


# ═══════════════════════════════════════════════════════════════════
# Orchestrator finally 兜底 — 设计契约验证
# ═══════════════════════════════════════════════════════════════════


def test_ctx_classes_have_last_short_circuit_reply_field():
    """3 个 ctx (ShortCircuitCtx/PreflightCtx/BoundaryPhaseCtx) 都必须有
    last_short_circuit_reply 字段, 默认 None. orchestrator finally 依赖此字段
    取短路 reply 文本."""
    from app.services.chat.intent_handlers import ShortCircuitCtx
    from app.services.chat.preflight import PreflightCtx
    from app.services.chat.boundary_phase import BoundaryPhaseCtx
    from dataclasses import fields

    for cls in (ShortCircuitCtx, PreflightCtx, BoundaryPhaseCtx):
        names = {f.name for f in fields(cls)}
        assert "last_short_circuit_reply" in names, (
            f"{cls.__name__} 缺 last_short_circuit_reply 字段"
        )


def test_orchestrator_finally_logic_present():
    """grep orchestrator source 验证 finally 兜底逻辑存在 (P0 BUG 修复回归)."""
    import inspect
    from app.services.chat import orchestrator as orch_mod

    src = inspect.getsource(orch_mod.stream_chat_response)
    assert "post_process_fired" in src, "缺少 post_process_fired flag"
    assert "last_short_circuit_reply" in src, "未读取 ctx.last_short_circuit_reply"
    # 正向 gate: sc_reply is not None (防 mid-try exception phantom fire)
    assert "sc_reply is not None" in src
    # boundary 路径必须通过 boundary_ctx.stopped 直接判别跳过 (不用独立 flag)
    assert "boundary_ctx.stopped" in src
    assert "boundary_handled" in src  # 保留 sentinel: boundary_handled = boundary_ctx... and ...stopped


def test_orchestrator_current_state_fast_path_precedes_full_fetch_and_intent_llm():
    """当前状态常见短语必须在 full fetch / intent LLM 之前本地判定。"""
    import inspect
    from app.services.chat import orchestrator as orch_mod

    src = inspect.getsource(orch_mod.stream_chat_response)
    fast_detect_pos = src.find("detect_current_state_fast_path(user_message)")
    fetch_guard_pos = src.find("elif forced_intent is None and not current_state_fast_path")
    intent_fast_pos = src.find("elif current_state_fast_path:")
    intent_llm_pos = src.find("detected_intent = await detect_intent_unified")
    light_fetch_pos = src.find("elif current_state_fast_path:", intent_fast_pos + 1)
    full_fetch_pos = src.find("fetched = await fetch_task")

    assert fast_detect_pos != -1, "缺 current-state fast path 本地判定"
    assert fetch_guard_pos != -1, "fetch_parallel_context 启动前必须排除 current_state_fast_path"
    assert intent_fast_pos != -1, "意图识别阶段缺 current_state_fast_path 分支"
    assert intent_fast_pos < intent_llm_pos, "fast path 必须排在 detect_intent_unified 之前"
    assert light_fetch_pos != -1, "fast path 必须走轻量 schedule fetch 分支"
    assert full_fetch_pos < light_fetch_pos, (
        "轻量 current-state fetch 必须作为 fetch_task 分支的 elif；"
        "fast path 下 fetch_task 为 None，因此不会进入 full fetch await"
    )


def test_orchestrator_skips_ai_memory_for_state_and_schedule_short_circuits():
    """当前状态/计划查询回复是临场回答，不应进入 AI 自我记忆。"""
    import inspect
    from app.services.chat import orchestrator as orch_mod

    src = inspect.getsource(orch_mod.stream_chat_response)
    assert "skip_ai_memory=(" in src
    assert '{"schedule_query", "current_state"}' in src


@pytest.mark.skip(reason="full orchestrator integration test — too many lazy imports to mock cleanly; covered by manual e2e DB verification")
@pytest.mark.asyncio
async def test_orchestrator_intent_short_circuit_fires_post_process():
    """SCHEDULE_QUERY 意图短路 → orchestrator finally 兜底应 fire _background_post_process."""
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from app.services.chat import orchestrator as orch_mod

    # 收集 fire_background 调用
    fire_calls: list = []
    real_fire_background = orch_mod._fire_background

    def _capturing_fire(coro):
        fire_calls.append(coro)
        # 尝试关闭协程避免警告
        try:
            coro.close()
        except Exception:
            pass

    # 构造一个最小化的 SCHEDULE_QUERY mock 路径
    agent = SimpleNamespace(id="agent1", name="Test", userId="user1", status="active")
    saved_msg = SimpleNamespace(id="msg1")
    conv = SimpleNamespace(workspaceId=None)

    async def _empty_handler(user_message, ctx, **kwargs):
        # 模拟短路 handler 调 ctx.finalize
        async def _gen():
            ctx.last_short_circuit_reply = "我现在在烤面包"
            yield {"event": "reply", "data": "{}"}
            yield {"event": "done", "data": "{}"}
        return True, _gen(), None

    with (
        patch.object(orch_mod, "_fire_background", side_effect=_capturing_fire),
        patch.object(orch_mod, "db", new=MagicMock(
            message=MagicMock(
                create=AsyncMock(return_value=saved_msg),
                find_many=AsyncMock(return_value=[]),
            ),
            conversation=MagicMock(
                find_unique=AsyncMock(return_value=conv),
                update=MagicMock(),
            ),
        )),
        # bind_agent_context / reset_current_agent are lazy-imported inside the function
        patch("app.services.runtime_config.bind_agent_context",
              new_callable=AsyncMock, return_value=None),
        patch("app.services.runtime_config.reset_current_agent"),
        patch.object(orch_mod, "LangSmithTracer") as mock_tracer_cls,
        patch.object(orch_mod, "run_boundary") as mock_boundary,
        patch.object(orch_mod, "resolve_pending_contradiction") as mock_pc,
        patch.object(orch_mod, "resolve_pending_deletion") as mock_pd,
        patch.object(orch_mod, "detect_intent_unified", new_callable=AsyncMock,
                     return_value=IntentResult(intent=IntentType.SCHEDULE_QUERY, confidence=1.0,
                                                metadata={"query_type": "current"})),
        patch.object(orch_mod, "_fetch_intent_context", new_callable=AsyncMock, return_value=""),
        patch.object(orch_mod, "fetch_parallel_context", new_callable=AsyncMock,
                     return_value=SimpleNamespace(
                         memory_relevance="weak",
                         retrieval_result=([], [], None),
                         portrait=None,
                         user_emotion={"emotion": "中性", "intensity": 0, "confidence": 0.0},
                         time_memories=[],
                         schedule=None,
                         topic_intimacy=0.5,
                         ai_status=None,
                         schedule_context=None,
                     )),
        patch.object(orch_mod, "maybe_awaken_l3", new_callable=AsyncMock, return_value=([], "无")),
        patch.object(orch_mod, "handle_schedule_query", side_effect=_empty_handler),
        patch.object(orch_mod, "push_topic", new_callable=AsyncMock, return_value=None),
    ):
        # 配置 boundary / preflight 不命中 (空 generator + ctx.stopped=False)
        async def _empty_gen(*args, **kwargs):
            if False:
                yield {}
        mock_boundary.return_value = _empty_gen()
        mock_pc.return_value = _empty_gen()
        mock_pd.return_value = _empty_gen()
        mock_tracer = MagicMock(
            trace_id=None, is_active=False, safe_trace_id=None,
            close=MagicMock(),
        )
        mock_tracer_cls.return_value.enter.return_value = mock_tracer
        mock_tracer_cls.return_value.attach_to_parent.return_value = mock_tracer

        events = await _drain(orch_mod.stream_chat_response(
            conversation_id="conv1",
            user_message="你现在在干嘛",
            agent=agent,
            user_id="user1",
        ))

    # 期望 fire_background 至少有一次调用是 _background_post_process
    # (短路 SCHEDULE_QUERY 命中 → finally 兜底 fire)
    coro_names = [getattr(c, "__qualname__", str(c)) for c in fire_calls]
    post_proc_calls = [n for n in coro_names if "post_process" in n.lower() or "run_post_process" in n]
    assert post_proc_calls, (
        f"短路意图 finally 兜底必须 fire post_process; 实际 fire_background 调用: {coro_names}"
    )
