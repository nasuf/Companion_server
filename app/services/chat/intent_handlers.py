"""Spec §3.4 各意图的短路处理器。

从 `orchestrator.stream_chat_response` 中抽出 7 个意图分支：每个 handler
只关心自己的输入/参考信息 + 生成 reply，尾部统一交给 `finalize_short_circuit`。

handler 作为 async generator 产出 SSE 事件，orchestrator 只需 `async for ...: yield`。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chat.tracing import LangSmithTracer

from app.services.chat.intent_replies import (
    apology_reply,
    current_state_reply,
    deletion_confirm_reply,
    end_reply,
    schedule_query_reply,
)
from app.services.chat.multi_intent import finalize_short_circuit
from app.services.memory.interaction.deletion import (
    detect_deletion_intent,
    find_matching_memories,
    generate_deletion_confirmation_prompt,
    save_pending_deletion,
)
from app.services.interaction.boundary import (
    APOLOGY_SINCERITY_MIN,
    PATIENCE_MAX,
    detect_apology,
    handle_apology,
)
from app.services.schedule_domain.schedule import (
    format_full_schedule_for_query,
    format_schedule_context,
    handle_schedule_adjustment,
    update_schedule_slot,
)

logger = logging.getLogger(__name__)


@dataclass
class ShortCircuitCtx:
    """短路分支共用的"尾部参数"；减少 handler 签名噪声。"""

    conversation_id: str
    agent_id: str | None
    user_id: str
    agent: Any
    reply_context: dict | None
    tracer: "LangSmithTracer"
    save_replies_fn: Callable[..., Any]
    pending_sub_fragments: dict[str, str]
    sub_intent_mode: bool
    reply_index_offset: int
    cached_patience: int

    async def finalize(self, reply: str) -> AsyncGenerator[dict, None]:
        async for evt in finalize_short_circuit(
            reply,
            conversation_id=self.conversation_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            agent=self.agent,
            reply_context=self.reply_context,
            tracer=self.tracer,
            save_replies_fn=self.save_replies_fn,
            pending_sub_fragments=self.pending_sub_fragments,
            sub_intent_mode=self.sub_intent_mode,
            reply_index_offset=self.reply_index_offset,
            cached_patience=self.cached_patience,
        ):
            yield evt


def _agent_name(agent) -> str:
    return getattr(agent, "name", "") or ""


# ═══════════════════════════════════════════════════════════════════
# §3.4.6 终结意图
# ═══════════════════════════════════════════════════════════════════


async def handle_conversation_end(
    user_message: str,
    ctx: ShortCircuitCtx,
    fallback_fn: Callable[..., Any],
) -> AsyncGenerator[dict, None]:
    farewell = await end_reply(
        message=user_message,
        personality_brief=_agent_name(ctx.agent),
    )
    if not farewell:
        farewell = await fallback_fn(
            ctx.agent, user_message,
            "用户要结束对话了。用你的性格风格生成一句简短的道别，不超过30字。不要用||分隔。",
        )
    async for evt in ctx.finalize(farewell):
        yield evt


# ═══════════════════════════════════════════════════════════════════
# §3.4.4 道歉承诺热路径
# ═══════════════════════════════════════════════════════════════════


async def handle_apology_promise(
    user_message: str,
    ctx: ShortCircuitCtx,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    """Spec §3.4.4: intent.unified 已分类为 apology_promise.

    Spec §2.6.2.1 要求道歉恢复耐心**必须过 sincerity >= 0.5 门禁** —
    即便 intent 分类判定是道歉, 也要小模型再看一眼诚意度 (防止 "对不起
    啦但我就是讨厌你" 这类低诚意道歉无条件恢复耐心). 门禁和 boundary_phase
    拉黑态道歉路径保持同一阈值, 两路径行为一致。
    """
    if not ctx.agent_id or ctx.cached_patience >= PATIENCE_MAX:
        return False, None
    try:
        apology = await detect_apology(user_message)
        if not (
            apology.get("is_apology")
            and apology.get("sincerity", 0) >= APOLOGY_SINCERITY_MIN
        ):
            # intent 识别为道歉但诚意不够 → 不短路, 落回正常 reply 流程
            return False, None
        new_patience = await handle_apology(ctx.agent_id, ctx.user_id)
        reply = await apology_reply(
            message=user_message,
            personality_brief=_agent_name(ctx.agent),
            new_patience=new_patience,
        ) or "好啦，我不生气了~"
        return True, ctx.finalize(reply)
    except Exception as e:
        logger.warning(f"Hot-path apology failed, falling through: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════
# §5.1-5.2 删除意图
# ═══════════════════════════════════════════════════════════════════


async def handle_deletion(
    user_message: str,
    ctx: ShortCircuitCtx,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    """返回 (handled, events_gen)。"""
    try:
        deletion_result = await detect_deletion_intent(user_message)
        description = (deletion_result or {}).get("target_description")
        if not description:
            return False, None

        candidates = await find_matching_memories(ctx.user_id, description)
        agent_name = ctx.agent.name if ctx.agent else "伙伴"
        if candidates:
            await save_pending_deletion(ctx.conversation_id, candidates)
            candidate_preview = "\n".join(
                f"{i + 1}. {c.get('content', c.get('summary', ''))[:60]}"
                for i, c in enumerate(candidates[:5])
            )
            reply = (
                await deletion_confirm_reply(
                    message=user_message,
                    personality_brief=agent_name,
                    candidate_memories=candidate_preview,
                )
                or await generate_deletion_confirmation_prompt(agent_name, candidates)
            )
        else:
            reply = "嗯...我好像没有关于这个的记忆呢。"
        return True, ctx.finalize(reply)
    except Exception as e:
        logger.warning(f"Hot-path deletion failed, falling through: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════
# §3.4.2 作息调整
# ═══════════════════════════════════════════════════════════════════


async def handle_schedule_adjust(
    user_message: str,
    ctx: ShortCircuitCtx,
    *,
    schedule: Any,
    ai_status: dict | None,
    topic_intimacy: float,
    mbti: dict | None,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    if not (ctx.agent_id and schedule and ai_status):
        return False, None
    try:
        adj_result = await handle_schedule_adjustment(
            agent_id=ctx.agent_id,
            request=user_message,
            current_status=ai_status,
            intimacy_score=float(topic_intimacy),
            mbti=mbti,
        )
        response = adj_result.get("response", "")
        if not response:
            return False, None
        if adj_result.get("accepted"):
            await update_schedule_slot(ctx.agent_id, schedule, ai_status)
        return True, ctx.finalize(response)
    except Exception as e:
        logger.warning(f"Schedule adjustment failed, falling through: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════
# §3.4.1 计划查询
# ═══════════════════════════════════════════════════════════════════


async def handle_schedule_query(
    user_message: str,
    ctx: ShortCircuitCtx,
    *,
    schedule: Any,
    ai_status: dict | None,
    portrait: Any,
    user_emotion: dict | None,
    query_type: str,
) -> tuple[bool, AsyncGenerator[dict, None] | None, str | None]:
    """返回 (handled, events_gen, schedule_context_for_prompt)。

    即使未 short-circuit，也会返回 `schedule_context` 让主流程注入 prompt。
    """
    if not schedule:
        return False, None, None
    schedule_context = format_full_schedule_for_query(schedule, query_type, ai_status)
    try:
        response = await schedule_query_reply(
            message=user_message,
            user_emotion=user_emotion,
            personality_brief=_agent_name(ctx.agent),
            user_portrait=str(portrait) if portrait else "(未知)",
            current_activity=format_schedule_context(ai_status) if ai_status else "(未知)",
            ai_schedule=schedule_context or "(未知)",
        )
        if not response:
            return False, None, schedule_context
        return True, ctx.finalize(response), schedule_context
    except Exception as e:
        logger.warning(f"Schedule query short-circuit failed, falling through: {e}")
        return False, None, schedule_context


# ═══════════════════════════════════════════════════════════════════
# §3.4.3 询问当前状态
# ═══════════════════════════════════════════════════════════════════


async def handle_current_state(
    user_message: str,
    ctx: ShortCircuitCtx,
    *,
    ai_status: dict | None,
    schedule_context: str | None,
    portrait: Any,
    user_emotion: dict | None,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    try:
        response = await current_state_reply(
            message=user_message,
            user_emotion=user_emotion,
            personality_brief=_agent_name(ctx.agent),
            user_portrait=str(portrait) if portrait else "(未知)",
            current_activity=format_schedule_context(ai_status) if ai_status else "(未知)",
            ai_schedule=schedule_context or "(未知)",
        )
        if not response:
            return False, None
        return True, ctx.finalize(response)
    except Exception as e:
        logger.warning(f"Current state short-circuit failed, falling through: {e}")
        return False, None
