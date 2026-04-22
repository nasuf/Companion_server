"""Orchestrator 关键阶段（boundary_phase / reply_generate）的集成测试。

这些是 P2 新抽离模块的最少必要测试：mock 掉外部依赖（LLM、DB、Redis），
断言事件序列与状态输出 （ctx.stopped / cached_patience / 返回的 replies）。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════
# boundary_phase
# ═══════════════════════════════════════════════════════════════════


async def _drain(agen):
    events: list[dict] = []
    async for evt in agen:
        events.append(evt)
    return events


def _make_boundary_ctx(**overrides):
    from app.services.chat.boundary_phase import BoundaryPhaseCtx

    defaults = dict(
        conversation_id="conv1",
        agent_id="agent1",
        user_id="user1",
        agent=SimpleNamespace(name="Alice"),
        user_message="你这个傻X",
        sub_intent_mode=False,
        parent_patience=None,
        tracer=MagicMock(trace_id=None, is_active=False),
        short_circuit_fn=AsyncMock(return_value=[
            {"event": "reply", "data": "{}"},
            {"event": "done", "data": "{}"},
        ]),
        fire_background_fn=MagicMock(),
        bg_memory_pipeline_fn=MagicMock(return_value=None),
    )
    defaults.update(overrides)
    return BoundaryPhaseCtx(**defaults)


@pytest.mark.asyncio
async def test_boundary_sub_intent_mode_skips_redis():
    """sub_intent_mode=True 直接复用 parent_patience，不读 Redis、不产出事件。"""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx(sub_intent_mode=True, parent_patience=42)
    with patch("app.services.chat.boundary_phase.check_boundary") as mock_check:
        events = await _drain(run_boundary(ctx))

    mock_check.assert_not_called()
    assert events == []
    assert ctx.stopped is False
    assert ctx.cached_patience == 42


@pytest.mark.asyncio
async def test_boundary_blocked_zone_no_apology_falls_through():
    """zone=blocked + 用户不在道歉 → 走拉黑回复（spec §2.6 步骤 2.3）。"""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx()
    with patch(
        "app.services.chat.boundary_phase.check_boundary",
        AsyncMock(return_value=({"zone": "blocked", "fallback": "..."}, 0)),
    ), patch(
        "app.services.chat.boundary_phase.detect_apology",
        AsyncMock(return_value={"is_apology": False, "sincerity": 0.0}),
    ), patch(
        "app.services.chat.boundary_phase.generate_boundary_reply_llm",
        AsyncMock(return_value="滚开"),
    ):
        events = await _drain(run_boundary(ctx))

    ctx.short_circuit_fn.assert_awaited_once()
    kwargs = ctx.short_circuit_fn.call_args.kwargs
    assert kwargs["extra_metadata"] == {"boundary": True, "zone": "blocked"}
    # P3.1 后只 fire memory_pipeline（不再 fire apology_check）
    assert ctx.fire_background_fn.call_count == 1
    ctx.tracer.close.assert_called_once()
    assert ctx.stopped is True
    assert len(events) == 2
    assert ctx.cached_patience == 0


@pytest.mark.asyncio
async def test_boundary_blocked_apology_unblocks():
    """zone=blocked + 检测到道歉 → 调 handle_apology + apology_reply（spec §2.6 步骤 2.2）。"""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx(user_message="对不起，是我不对")
    mock_apology = AsyncMock(return_value=70)
    with patch(
        "app.services.chat.boundary_phase.check_boundary",
        AsyncMock(return_value=({"zone": "blocked", "fallback": "..."}, 0)),
    ), patch(
        "app.services.chat.boundary_phase.detect_apology",
        AsyncMock(return_value={"is_apology": True, "sincerity": 0.9}),
    ), patch(
        "app.services.chat.boundary_phase.handle_apology",
        mock_apology,
    ), patch(
        "app.services.chat.boundary_phase.apology_reply",
        AsyncMock(return_value="原谅你啦"),
    ), patch(
        "app.services.chat.boundary_phase.generate_boundary_reply_llm",
    ) as mock_blacklist:
        events = await _drain(run_boundary(ctx))

    # 调了 handle_apology 恢复耐心，没调拉黑回复 LLM
    mock_apology.assert_awaited_once_with("agent1", "user1")
    mock_blacklist.assert_not_called()
    # metadata 带 apology_unblock=True
    kwargs = ctx.short_circuit_fn.call_args.kwargs
    assert kwargs["extra_metadata"] == {
        "boundary": True, "zone": "blocked", "apology_unblock": True,
    }
    # 触发 memory_pipeline（无 apology_check）
    assert ctx.fire_background_fn.call_count == 1
    assert ctx.stopped is True
    assert len(events) == 2


@pytest.mark.asyncio
async def test_boundary_attack_ai_triggers_level_classification_and_violation():
    """zone=low + 攻击AI → spec §2.4 K4：final_warning 替代 K1/K2/K3 分级回复。"""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx()
    mock_reply = AsyncMock(return_value="你这样我很难过")
    with patch(
        "app.services.chat.boundary_phase.check_boundary",
        AsyncMock(return_value=({"zone": "low", "fallback": "..."}, 10)),
    ), patch(
        "app.services.chat.boundary_phase.attack_target_classify",
        AsyncMock(return_value="攻击AI"),
    ), patch(
        "app.services.chat.boundary_phase.attack_level_classify",
        AsyncMock(return_value="K2"),
    ), patch(
        "app.services.chat.boundary_phase.generate_boundary_reply_llm",
        mock_reply,
    ), patch(
        "app.services.chat.boundary_phase.process_boundary_violation",
    ) as mock_violation:
        await _drain(run_boundary(ctx))

    kwargs = ctx.short_circuit_fn.call_args.kwargs
    assert kwargs["extra_metadata"] == {
        "boundary": True, "zone": "low", "attack_level": "K2", "final_warning": True,
    }
    # generate_boundary_reply_llm 以 final_warning=True 调用，attack_level=None
    reply_kwargs = mock_reply.call_args.kwargs
    assert reply_kwargs["final_warning"] is True
    assert reply_kwargs["attack_level"] is None
    # process_boundary_violation 通过 fire_background_fn 调度，传入 K1/K2/K3
    mock_violation.assert_called_once_with("agent1", "user1", "K2")
    assert ctx.stopped is True


@pytest.mark.asyncio
async def test_boundary_attack_ai_zone_medium_uses_level_reply():
    """zone=medium + 攻击AI → 按 K1/K2/K3 分级回复，不触发最终警告。"""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx()
    mock_reply = AsyncMock(return_value="你这样我很难过")
    with patch(
        "app.services.chat.boundary_phase.check_boundary",
        AsyncMock(return_value=({"zone": "medium", "fallback": "..."}, 50)),
    ), patch(
        "app.services.chat.boundary_phase.attack_target_classify",
        AsyncMock(return_value="攻击AI"),
    ), patch(
        "app.services.chat.boundary_phase.attack_level_classify",
        AsyncMock(return_value="K2"),
    ), patch(
        "app.services.chat.boundary_phase.generate_boundary_reply_llm",
        mock_reply,
    ), patch(
        "app.services.chat.boundary_phase.process_boundary_violation",
    ):
        await _drain(run_boundary(ctx))

    reply_kwargs = mock_reply.call_args.kwargs
    assert reply_kwargs["final_warning"] is False
    assert reply_kwargs["attack_level"] == "K2"
    assert "final_warning" not in ctx.short_circuit_fn.call_args.kwargs["extra_metadata"]


@pytest.mark.asyncio
async def test_boundary_normal_zone_passthrough():
    """耐心满值、无违禁 → 不短路，cached_patience 原样传出。"""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx(user_message="今天天气真好")
    with patch(
        "app.services.chat.boundary_phase.check_boundary",
        AsyncMock(return_value=(None, 100)),
    ):
        events = await _drain(run_boundary(ctx))

    assert events == []
    assert ctx.stopped is False
    assert ctx.cached_patience == 100


@pytest.mark.asyncio
async def test_boundary_residual_medium_patience_short_circuits():
    """无违禁命中但耐心在 medium 区间 → 步骤 6 短路。"""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx(user_message="你好")
    with patch(
        "app.services.chat.boundary_phase.check_boundary",
        AsyncMock(return_value=(None, 50)),
    ), patch(
        "app.services.chat.boundary_phase.banned_word_check",
        AsyncMock(return_value=False),
    ), patch(
        "app.services.chat.boundary_phase.generate_boundary_reply_llm",
        AsyncMock(return_value="嗯"),
    ):
        events = await _drain(run_boundary(ctx))

    kwargs = ctx.short_circuit_fn.call_args.kwargs
    assert kwargs["extra_metadata"] == {"boundary": True, "zone": "medium"}
    assert ctx.stopped is True
    assert len(events) == 2


# ═══════════════════════════════════════════════════════════════════
# reply_generate
# ═══════════════════════════════════════════════════════════════════


def _make_reply_generate_kwargs(**overrides):
    from app.services.chat.intent_dispatcher import IntentResult, IntentType

    tier_weak = AsyncMock(return_value="弱相关回复")
    tier_strong = AsyncMock(return_value="强相关回复")
    tier_medium = AsyncMock(return_value="中相关回复")
    tier_l3 = AsyncMock(return_value="L3 回复")
    split_llm = AsyncMock(return_value=None)

    defaults = dict(
        contradiction_inquiry=None,
        detected_intent=IntentResult(intent=IntentType.NONE, confidence=0.0),
        memory_relevance="weak",
        relational_context=None,
        schedule_context=None,
        delay_context=None,
        l3_memories=[],
        classified_memories=[],
        messages_dicts=[{"role": "user", "content": "hi"}],
        portrait=None,
        prompt_user_emotion=None,
        user_message="hi",
        agent=SimpleNamespace(name="Alice"),
        chat_messages=[{"role": "system", "content": "..."}],
        reply_count=1,
        max_reply_count=3,
        max_total=150,
        tier_fns={
            "weak": tier_weak,
            "medium": tier_medium,
            "strong": tier_strong,
            "l3": tier_l3,
        },
        split_llm_fn=split_llm,
        truncate_fn=lambda t, n: t[:n],
        pipe_fallback_fn=lambda raw, n, per, total: [raw[:per]],
    )
    defaults.update(overrides)
    return defaults


@pytest.mark.asyncio
async def test_reply_generate_contradiction_inquiry_short_circuits_llm():
    """矛盾追问存在 → 直接用 inquiry 当 reply，不调 tier 也不调主 LLM。"""
    from app.services.chat.reply_generate import generate_reply

    kwargs = _make_reply_generate_kwargs(contradiction_inquiry="你之前不是说住苏州吗？")
    replies, raw = await generate_reply(**kwargs)

    assert replies == ["你之前不是说住苏州吗？"]
    assert raw == "你之前不是说住苏州吗？"
    kwargs["tier_fns"]["weak"].assert_not_called()
    kwargs["tier_fns"]["l3"].assert_not_called()


@pytest.mark.asyncio
async def test_reply_generate_tier_weak_bypasses_main_llm():
    """relevance=weak + intent=NONE 且无上下文 → 走 weak tier，不调主 LLM。"""
    from app.services.chat.reply_generate import generate_reply

    kwargs = _make_reply_generate_kwargs(memory_relevance="weak")
    with patch("app.services.chat.reply_generate.get_chat_model") as mock_model:
        replies, raw = await generate_reply(**kwargs)

    mock_model.assert_not_called()
    kwargs["tier_fns"]["weak"].assert_awaited_once()
    assert replies == ["弱相关回复"]
    assert raw == "弱相关回复"


@pytest.mark.asyncio
async def test_reply_generate_tier_l3_when_l3_memories_present():
    """有 l3_memories → 使用 l3 tier，即使 relevance=strong。"""
    from app.services.chat.reply_generate import generate_reply

    kwargs = _make_reply_generate_kwargs(
        memory_relevance="strong",
        l3_memories=["很久以前你说过喜欢下雨"],
    )
    replies, _ = await generate_reply(**kwargs)

    kwargs["tier_fns"]["l3"].assert_awaited_once()
    kwargs["tier_fns"]["strong"].assert_not_called()
    assert replies == ["L3 回复"]


@pytest.mark.asyncio
async def test_reply_generate_relational_context_disables_tier():
    """relational_context 存在 → can_use_tier=False，走主 LLM。"""
    from app.services.chat.reply_generate import generate_reply

    class _FakeChunk:
        def __init__(self, content):
            self.content = content

    async def _stream(_messages):
        for tok in ["主 LLM ", "回复"]:
            yield _FakeChunk(tok)

    mock_model = MagicMock()
    mock_model.astream = _stream

    kwargs = _make_reply_generate_kwargs(
        memory_relevance="weak",
        relational_context="用户很低落",
    )
    with patch(
        "app.services.chat.reply_generate.get_chat_model",
        return_value=mock_model,
    ), patch(
        "app.services.chat.reply_generate.convert_messages",
        side_effect=lambda m: m,
    ):
        replies, raw = await generate_reply(**kwargs)

    kwargs["tier_fns"]["weak"].assert_not_called()
    assert raw == "主 LLM 回复"
    assert len(replies) == 1


# ═══════════════════════════════════════════════════════════════════
# reply_post_process: AI reply emotion override (P3.2)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_emit_replies_uses_ai_reply_emotion_for_emoji():
    """spec §5 step 1：reply_emotion 提供时，pick_one_emoji 收到 LLM 出来的标签。"""
    from app.services.chat.reply_post_process import emit_replies

    captured: dict = {}

    def _capture(pleasure, arousal, primary_emotion=None):
        captured["primary_emotion"] = primary_emotion
        return "🎉"

    emitted: list[dict] = []
    with patch(
        "app.services.chat.reply_post_process.should_add_emoji", return_value=True,
    ), patch(
        "app.services.chat.reply_post_process.pick_one_emoji", side_effect=_capture,
    ), patch(
        "app.services.chat.reply_post_process.should_add_sticker", return_value=False,
    ):
        async for _ in emit_replies(
            ["hi"],
            reply_context=None,
            reply_index_offset=0,
            sub_intent_mode=False,
            emotion={"primary_emotion": "悲伤"},  # PAD 缓存说悲伤
            agent=SimpleNamespace(name="A"),
            user_message="嗨",
            delay_reply_fn=AsyncMock(return_value=None),
            fallback_fn=AsyncMock(return_value=""),
            emitted_replies=emitted,
            reply_emotion={"emotion": "高兴", "intensity": 80},  # LLM 说回复是高兴
        ):
            pass

    # LLM 输出的"高兴"应覆盖缓存的"悲伤"
    assert captured["primary_emotion"] == "高兴"


@pytest.mark.asyncio
async def test_emit_replies_falls_back_to_pad_emotion_when_no_reply_emotion():
    """reply_emotion 缺失/空 → 回退到 PAD primary_emotion。"""
    from app.services.chat.reply_post_process import emit_replies

    captured: dict = {}

    def _capture(pleasure, arousal, primary_emotion=None):
        captured["primary_emotion"] = primary_emotion
        return "😢"

    emitted: list[dict] = []
    with patch(
        "app.services.chat.reply_post_process.should_add_emoji", return_value=True,
    ), patch(
        "app.services.chat.reply_post_process.pick_one_emoji", side_effect=_capture,
    ), patch(
        "app.services.chat.reply_post_process.should_add_sticker", return_value=False,
    ):
        async for _ in emit_replies(
            ["hi"],
            reply_context=None,
            reply_index_offset=0,
            sub_intent_mode=False,
            emotion={"primary_emotion": "悲伤"},
            agent=SimpleNamespace(name="A"),
            user_message="嗨",
            delay_reply_fn=AsyncMock(return_value=None),
            fallback_fn=AsyncMock(return_value=""),
            emitted_replies=emitted,
            reply_emotion={},  # LLM 失败/空
        ):
            pass

    assert captured["primary_emotion"] == "悲伤"


@pytest.mark.asyncio
async def test_reply_generate_intent_schedule_adjust_disables_tier():
    """非 NONE/L3_RECALL 意图 → 走主 LLM 而非 tier。"""
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from app.services.chat.reply_generate import generate_reply

    class _FakeChunk:
        def __init__(self, content):
            self.content = content

    async def _stream(_messages):
        yield _FakeChunk("排期已调整")

    mock_model = MagicMock()
    mock_model.astream = _stream

    kwargs = _make_reply_generate_kwargs(
        memory_relevance="weak",
        detected_intent=IntentResult(intent=IntentType.SCHEDULE_ADJUST, confidence=1.0),
    )
    with patch(
        "app.services.chat.reply_generate.get_chat_model",
        return_value=mock_model,
    ), patch(
        "app.services.chat.reply_generate.convert_messages",
        side_effect=lambda m: m,
    ):
        replies, raw = await generate_reply(**kwargs)

    kwargs["tier_fns"]["weak"].assert_not_called()
    assert raw == "排期已调整"
    assert len(replies) == 1
