"""Spec §2.6：边界系统在聊天热路径的执行阶段。

从 orchestrator 抽离：sub_intent_mode 跳过 / 读耐心 → 关键词命中 → LLM 兜底违禁判断 →
拉黑短路 → 攻击目标分类 → 攻击级别扣分 → 步骤 6 中/低耐心短路。

调用方传入 `BoundaryPhaseCtx`，phase 通过 ctx.cached_patience / ctx.stopped 与调用方交换结果。
短路时本函数直接 yield reply/done 事件并关闭 trace；放行时保持 ctx.stopped=False。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from app.services.chat.intent_replies import (
    apology_reply,
    attack_level_classify,
    attack_target_classify,
    banned_word_check,
)

if TYPE_CHECKING:
    from app.services.chat.tracing import LangSmithTracer
from app.services.interaction.boundary import (
    APOLOGY_SINCERITY_MIN,
    PATIENCE_MAX,
    check_boundary,
    detect_apology,
    generate_boundary_reply_llm,
    get_patience_zone,
    handle_apology,
    process_boundary_violation,
)

logger = logging.getLogger(__name__)


@dataclass
class BoundaryPhaseCtx:
    """spec §2.6 phase 的输入 + 可变状态。"""

    conversation_id: str
    agent_id: str | None
    user_id: str
    agent: Any
    user_message: str
    sub_intent_mode: bool
    parent_patience: int | None
    tracer: "LangSmithTracer"
    short_circuit_fn: Callable[..., Awaitable[list[dict]]]
    fire_background_fn: Callable[[Any], None]
    bg_memory_pipeline_fn: Callable[..., Any]
    # 输出
    stopped: bool = False
    cached_patience: int = PATIENCE_MAX


def _personality_brief(agent: Any) -> str:
    return getattr(agent, "name", "") or ""


async def _maybe_llm_banned_check(
    user_message: str, cached_patience: int,
) -> dict | None:
    """关键词没命中 + 耐心非满 + 消息足够长时，调小模型语义违禁判断。"""
    if cached_patience >= PATIENCE_MAX or len(user_message.strip()) <= 3:
        return None
    try:
        if not await banned_word_check(user_message):
            return None
    except Exception as e:
        logger.warning(f"LLM banned word check failed: {e}")
        return None
    zone = get_patience_zone(cached_patience)
    logger.info(
        f"[BOUNDARY-LLM] banned_word_check hit for '{user_message[:40]}' "
        f"zone={zone} patience={cached_patience}"
    )
    return {
        "blocked": zone == "blocked",
        "zone": zone,
        "hits": [],
        "fallback": "...",
        "llm_detected": True,
    }


async def _emit_short_circuit(
    ctx: BoundaryPhaseCtx,
    reply: str,
    extra_metadata: dict,
) -> AsyncGenerator[dict, None]:
    """复用 ctx.short_circuit_fn 推 reply/done 事件。"""
    events = await ctx.short_circuit_fn(
        reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
        extra_metadata=extra_metadata,
    )
    for evt in events:
        yield evt


async def _handle_blocked(
    ctx: BoundaryPhaseCtx, boundary_result: dict,
) -> AsyncGenerator[dict, None]:
    """spec §2.6 步骤 2：拉黑。先检测道歉承诺；命中则恢复+和解回复，否则拉黑回复。"""
    apology = await detect_apology(ctx.user_message)
    if apology.get("is_apology") and apology.get("sincerity", 0) >= APOLOGY_SINCERITY_MIN:
        new_patience = await handle_apology(ctx.agent_id, ctx.user_id)
        reply = await apology_reply(
            message=ctx.user_message,
            personality_brief=_personality_brief(ctx.agent),
            new_patience=new_patience,
        ) or "好啦，我不生气了~"
        async for evt in _emit_short_circuit(
            ctx, reply,
            {"boundary": True, "zone": "blocked", "apology_unblock": True},
        ):
            yield evt
        ctx.fire_background_fn(ctx.bg_memory_pipeline_fn(ctx.user_id, [
            {"role": "user", "content": ctx.user_message},
            {"role": "assistant", "content": reply},
        ]))
        ctx.tracer.close()
        ctx.stopped = True
        return

    response = await generate_boundary_reply_llm(
        zone="blocked",
        message=ctx.user_message,
        personality_brief=_personality_brief(ctx.agent),
    ) or boundary_result.get("fallback", "...")
    async for evt in _emit_short_circuit(ctx, response, {"boundary": True, "zone": "blocked"}):
        yield evt
    ctx.fire_background_fn(ctx.bg_memory_pipeline_fn(ctx.user_id, [
        {"role": "user", "content": ctx.user_message},
        {"role": "assistant", "content": response},
    ]))
    ctx.tracer.close()
    ctx.stopped = True


async def _handle_attack_target_non_ai(
    ctx: BoundaryPhaseCtx, zone: str, attack_target: str, boundary_result: dict,
) -> AsyncGenerator[dict, None]:
    """spec §2.6 步骤 4：攻击目标不是 AI，按 zone 回复不扣分。"""
    response = await generate_boundary_reply_llm(
        zone=zone,
        message=ctx.user_message,
        personality_brief=_personality_brief(ctx.agent),
    ) or boundary_result.get("fallback", "...")
    metadata = {"boundary": True, "zone": zone, "attack_target": attack_target}
    async for evt in _emit_short_circuit(ctx, response, metadata):
        yield evt
    ctx.tracer.close()
    ctx.stopped = True


async def _handle_attack_ai(
    ctx: BoundaryPhaseCtx, zone: str, boundary_result: dict,
) -> AsyncGenerator[dict, None]:
    """spec §2.6 步骤 5：攻击 AI → 级别识别 + 扣分 + 分级回复。

    spec §2.4 K4（最终警告）：处于低耐心区（zone=low）时再次攻击 AI，
    发出最后一次警告（替代 K1/K2/K3 攻击分级回复）。
    """
    attack_level = await attack_level_classify(ctx.user_message)
    is_final_warning = zone == "low"
    response = await generate_boundary_reply_llm(
        zone=zone,
        message=ctx.user_message,
        personality_brief=_personality_brief(ctx.agent),
        attack_level=None if is_final_warning else attack_level,
        final_warning=is_final_warning,
    ) or boundary_result.get("fallback", "...")
    metadata = {"boundary": True, "zone": zone, "attack_level": attack_level}
    if is_final_warning:
        metadata["final_warning"] = True
    async for evt in _emit_short_circuit(ctx, response, metadata):
        yield evt
    if attack_level:
        ctx.fire_background_fn(process_boundary_violation(ctx.agent_id, ctx.user_id, attack_level))
    ctx.fire_background_fn(ctx.bg_memory_pipeline_fn(ctx.user_id, [
        {"role": "user", "content": ctx.user_message},
        {"role": "assistant", "content": response},
    ]))
    ctx.tracer.close()
    ctx.stopped = True


async def _handle_residual_patience(
    ctx: BoundaryPhaseCtx, patience_zone: str,
) -> AsyncGenerator[dict, None]:
    """spec §2.6 步骤 6：不含违禁词但耐心仍在中/低区间，按 zone 回复。"""
    try:
        response = await generate_boundary_reply_llm(
            zone=patience_zone,
            message=ctx.user_message,
            personality_brief=_personality_brief(ctx.agent),
        )
    except Exception as e:
        logger.warning(f"Medium/low patience short-circuit failed: {e}")
        return
    if not response:
        return
    async for evt in _emit_short_circuit(ctx, response, {"boundary": True, "zone": patience_zone}):
        yield evt
    ctx.tracer.close()
    ctx.stopped = True


async def run_boundary(ctx: BoundaryPhaseCtx) -> AsyncGenerator[dict, None]:
    """spec §2.6 全流程入口。写入 ctx.cached_patience；短路时 yield 并置 stopped=True。"""
    # sub_intent_mode：父调用已完成边界检查，直接复用传入的 parent_patience
    if ctx.sub_intent_mode:
        ctx.cached_patience = (
            ctx.parent_patience if ctx.parent_patience is not None else PATIENCE_MAX
        )
        return

    if not ctx.agent_id:
        # 无 agent_id 不做边界判断，`cached_patience` 保持默认 100
        return

    boundary_result, ctx.cached_patience = await check_boundary(
        ctx.agent_id, ctx.user_id, ctx.user_message,
    )
    if boundary_result is None:
        boundary_result = await _maybe_llm_banned_check(ctx.user_message, ctx.cached_patience)

    if boundary_result:
        zone = boundary_result["zone"]

        # 步骤 2：拉黑
        if zone == "blocked":
            async for evt in _handle_blocked(ctx, boundary_result):
                yield evt
            return

        # 步骤 4：攻击目标识别
        attack_target = await attack_target_classify(ctx.user_message)
        if attack_target and attack_target != "攻击AI":
            if zone in ("medium", "low"):
                async for evt in _handle_attack_target_non_ai(
                    ctx, zone, attack_target, boundary_result,
                ):
                    yield evt
                return
            # zone == "normal" + 非攻击AI → 放行到意图识别
            boundary_result = None

        # 步骤 5：攻击 AI
        if boundary_result:
            async for evt in _handle_attack_ai(ctx, zone, boundary_result):
                yield evt
            return

    # 步骤 6：非攻击场景下中/低耐心短路
    patience_zone = get_patience_zone(ctx.cached_patience)
    if patience_zone in ("medium", "low"):
        async for evt in _handle_residual_patience(ctx, patience_zone):
            yield evt
