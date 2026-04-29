"""Orchestrator 关键阶段（boundary_phase / reply_generate）的集成测试。

这些是 P2 新抽离模块的最少必要测试：mock 掉外部依赖（LLM、DB、Redis），
断言事件序列与状态输出 （ctx.stopped / cached_patience / 返回的 replies）。
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
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


@contextmanager
def _patch_attack_ai_flow(
    *,
    zone: str,
    cached_patience: int,
    attack_level: str,
    reply_mock: AsyncMock,
    violation_mock: AsyncMock,
):
    """安装 _handle_attack_ai 流程需要的 5 个 mock。

    调用方只管 reply_mock / violation_mock 的返回值或异常, 拿到已激活的
    patch 栈后直接 run_boundary。
    """
    with ExitStack() as stack:
        stack.enter_context(patch(
            "app.services.chat.boundary_phase.check_boundary",
            AsyncMock(return_value=({"zone": zone, "fallback": "..."}, cached_patience)),
        ))
        stack.enter_context(patch(
            "app.services.chat.boundary_phase.attack_target_classify",
            AsyncMock(return_value="攻击AI"),
        ))
        stack.enter_context(patch(
            "app.services.chat.boundary_phase.attack_level_classify",
            AsyncMock(return_value=attack_level),
        ))
        stack.enter_context(patch(
            "app.services.chat.boundary_phase.generate_boundary_reply_llm",
            reply_mock,
        ))
        stack.enter_context(patch(
            "app.services.chat.boundary_phase.process_boundary_violation",
            violation_mock,
        ))
        yield


@pytest.mark.asyncio
async def test_boundary_sub_intent_mode_skips_redis_when_clean():
    """sub_intent_mode=True + 子片段干净 → 复用 parent_patience, 不读 Redis, 不产出事件.
    (spec §2.6: 父调用已检查整体, 这里仅对 fragment 做违禁词扫描.)
    """
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx(sub_intent_mode=True, parent_patience=42, user_message="你好")
    with patch("app.services.chat.boundary_phase.check_boundary") as mock_check, patch(
        "app.services.chat.boundary_phase._maybe_llm_banned_check",
        AsyncMock(return_value=None),
    ):
        events = await _drain(run_boundary(ctx))

    mock_check.assert_not_called()  # sub-intent 不重读 patience
    assert events == []
    assert ctx.stopped is False
    assert ctx.cached_patience == 42


@pytest.mark.asyncio
async def test_boundary_sub_intent_banned_keyword_caught():
    """sub_intent fragment 含违禁词 → 走攻击路径短路, 防多意图拆分后子片段绕过边界."""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx(sub_intent_mode=True, parent_patience=80, user_message="你这个垃圾AI")
    with patch(
        "app.services.chat.boundary_phase.check_banned_keywords",
        return_value=["垃圾AI"],
    ), patch(
        "app.services.chat.boundary_phase.attack_level_classify",
        AsyncMock(return_value="K1"),
    ), patch(
        "app.services.chat.boundary_phase.process_boundary_violation",
        AsyncMock(return_value=75),
    ), patch(
        "app.services.chat.boundary_phase.generate_boundary_reply_llm",
        AsyncMock(return_value="别这样说"),
    ):
        events = await _drain(run_boundary(ctx))

    # 触发了攻击短路
    ctx.short_circuit_fn.assert_awaited_once()
    assert ctx.stopped is True


@pytest.mark.asyncio
async def test_boundary_sub_intent_llm_fallback_catches_obfuscated():
    """sub_intent fragment 关键词没命中但 LLM 兜底判违禁 → 仍被拦.
    覆盖谐音/缩写/新造词 (e.g. 草字头变体) 绕过关键词表的场景.
    """
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx(sub_intent_mode=True, parent_patience=50, user_message="你这个**ai")
    with patch(
        "app.services.chat.boundary_phase.check_banned_keywords",
        return_value=[],  # 关键词表没命中
    ), patch(
        "app.services.chat.boundary_phase._maybe_llm_banned_check",
        AsyncMock(return_value={"zone": "medium", "blocked": False, "hits": [], "fallback": "..."}),
    ), patch(
        "app.services.chat.boundary_phase.attack_target_classify",
        AsyncMock(return_value="攻击AI"),
    ), patch(
        "app.services.chat.boundary_phase.attack_level_classify",
        AsyncMock(return_value="K2"),
    ), patch(
        "app.services.chat.boundary_phase.process_boundary_violation",
        AsyncMock(return_value=35),
    ), patch(
        "app.services.chat.boundary_phase.generate_boundary_reply_llm",
        AsyncMock(return_value="不喜欢你这样说话"),
    ):
        await _drain(run_boundary(ctx))

    # LLM 兜底判违禁 → 仍走攻击短路
    ctx.short_circuit_fn.assert_awaited_once()
    assert ctx.stopped is True


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
    # P1-5: 拉黑回复不再 fire memory_pipeline (防污染 AI 自我记忆)
    assert ctx.fire_background_fn.call_count == 0
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
@pytest.mark.parametrize(
    "zone, cached, level, new_patience, expect_final_warning, expect_level_in_reply",
    [
        # 扣分后 patience < 20 → final_warning 覆写 K1/K2/K3
        ("low", 25, "K2", 10, True, None),
        # 扣分后 patience 恰好 20 (不小于 20) → K1 分档回复
        ("low", 25, "K1", 20, False, "K1"),
        # medium 高值 → K2 分档回复
        ("medium", 50, "K2", 35, False, "K2"),
    ],
)
async def test_boundary_attack_ai_prompt_selection(
    zone, cached, level, new_patience, expect_final_warning, expect_level_in_reply,
):
    """spec §5.3 扣分在前 + §5.4 按 attack_level 选 prompt, PM 补丁规则: 扣分后 <20 用 final_warning."""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx()
    reply_mock = AsyncMock(return_value="...")
    violation_mock = AsyncMock(return_value=new_patience)
    with _patch_attack_ai_flow(
        zone=zone, cached_patience=cached, attack_level=level,
        reply_mock=reply_mock, violation_mock=violation_mock,
    ):
        await _drain(run_boundary(ctx))

    violation_mock.assert_awaited_once_with("agent1", "user1", level)
    reply_kwargs = reply_mock.call_args.kwargs
    assert reply_kwargs["final_warning"] is expect_final_warning
    assert reply_kwargs["attack_level"] == expect_level_in_reply

    meta = ctx.short_circuit_fn.call_args.kwargs["extra_metadata"]
    assert meta["zone"] == zone
    assert meta["attack_level"] == level
    assert meta.get("final_warning") is (True if expect_final_warning else None)
    assert ctx.cached_patience == new_patience
    assert ctx.stopped is True


@pytest.mark.asyncio
async def test_boundary_attack_to_zero_routes_to_blacklist_not_final_warning():
    """P2-6: 当条扣到 ≤ 0 → 走 blacklist 回复 + zone='blocked' + becomes_blocked=True.
    旧行为: 走 final_warning 让用户当条看到"最终警告", 下条才被拉黑 → 困惑.
    新行为: 立刻通知用户已进入拉黑态, 行为对齐 spec §2.6 patience ≤ 0 即拉黑.
    """
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx()
    reply_mock = AsyncMock(return_value="不想理你了")
    violation_mock = AsyncMock(return_value=0)  # 扣分后 patience=0
    with _patch_attack_ai_flow(
        zone="low", cached_patience=20, attack_level="K3",
        reply_mock=reply_mock, violation_mock=violation_mock,
    ):
        await _drain(run_boundary(ctx))

    # generate_boundary_reply_llm 应以 zone='blocked' 调用 (不是 'low')
    reply_kwargs = reply_mock.call_args.kwargs
    assert reply_kwargs["zone"] == "blocked"
    # 不该带 attack_level / final_warning kwargs (走 blacklist 路径不传这两个)
    assert "attack_level" not in reply_kwargs or reply_kwargs.get("attack_level") is None
    assert reply_kwargs.get("final_warning") in (False, None)

    meta = ctx.short_circuit_fn.call_args.kwargs["extra_metadata"]
    assert meta["zone"] == "blocked"
    assert meta["becomes_blocked"] is True
    assert meta["attack_level"] == "K3"  # 仍记录 attack_level 供审计
    assert ctx.stopped is True
    assert ctx.cached_patience == 0


@pytest.mark.asyncio
async def test_boundary_attack_ai_violation_failure_falls_back_to_cached_patience():
    """process_boundary_violation 抛异常 → 回退 ctx.cached_patience 做阈值判断, 不阻断回复."""
    from app.services.chat.boundary_phase import run_boundary

    ctx = _make_boundary_ctx()
    reply_mock = AsyncMock(return_value="...")
    with _patch_attack_ai_flow(
        zone="low", cached_patience=15, attack_level="K1",  # cached=15 <20
        reply_mock=reply_mock,
        violation_mock=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        await _drain(run_boundary(ctx))

    # cached_patience=15 <20 → final_warning 仍应触发
    assert reply_mock.call_args.kwargs["final_warning"] is True


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
    replies, raw, _ = await generate_reply(**kwargs)

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
        replies, raw, _ = await generate_reply(**kwargs)

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
    replies, _, _ = await generate_reply(**kwargs)

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
        replies, raw, _ = await generate_reply(**kwargs)

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
        replies, raw, _ = await generate_reply(**kwargs)

    kwargs["tier_fns"]["weak"].assert_not_called()
    assert raw == "排期已调整"
    assert len(replies) == 1


@pytest.mark.asyncio
async def test_main_reply_fallback_on_llm_failure():
    """spec-audit: primary + Ollama 都抛异常 → 返回静态兜底文本 + is_fallback=True."""
    from app.services.chat.reply_generate import generate_reply, _MAIN_REPLY_ULTIMATE_FALLBACK
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from app.services.llm.resilience import reset_breakers_for_testing
    reset_breakers_for_testing()

    async def _fail_stream(_messages):
        raise RuntimeError("primary dead")
        yield  # unreachable

    primary_model = MagicMock()
    primary_model.astream = _fail_stream
    fallback_model = MagicMock()
    fallback_model.astream = _fail_stream

    kwargs = _make_reply_generate_kwargs(
        memory_relevance="weak",
        detected_intent=IntentResult(intent=IntentType.SCHEDULE_ADJUST, confidence=1.0),
    )
    with patch(
        "app.services.chat.reply_generate.get_chat_model",
        return_value=primary_model,
    ), patch(
        "app.services.chat.reply_generate.get_fallback_chat_model",
        return_value=fallback_model,
    ), patch(
        "app.services.chat.reply_generate.convert_messages",
        side_effect=lambda m: m,
    ), patch(
        "app.services.chat.reply_generate.provider_name",
        return_value="dashscope",  # primary 不是 ollama, 才会尝试 fallback
    ):
        replies, raw, is_fallback = await generate_reply(**kwargs)

    assert is_fallback is True
    assert raw == _MAIN_REPLY_ULTIMATE_FALLBACK
    assert replies == [_MAIN_REPLY_ULTIMATE_FALLBACK]
