"""交互前置校验：pending 跨消息状态 → 边界分流（spec §2.6 / §4 / §5）。

把 orchestrator 进入主流程前的两个独立判定块抽出：

- `resolve_pending_contradiction`：spec §4 step 3-5，处理上一轮的矛盾追问回答
- `resolve_pending_deletion`：spec §5 step 3，处理上一轮的删除确认/取消

两者都是 AsyncGenerator：命中就 yield reply/done 事件并通过 `ctx.stopped=True`
告诉 orchestrator 终止本次流程；否则不产出任何事件，orchestrator 继续下一阶段。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chat.tracing import LangSmithTracer

from app.services.chat.intent_replies import deletion_done_reply
from app.services.memory.interaction.contradiction import (
    analyze_contradiction_response,
    apply_contradiction_resolution,
    clear_pending_contradiction,
    generate_contradiction_reply,
    load_pending_contradiction,
)
from app.services.memory.interaction.deletion import (
    apply_reschedule,
    clear_pending_deletion,
    execute_confirmed_deletion,
    generate_deletion_reply,
    is_deletion_confirmed,
    load_pending_action,
)

logger = logging.getLogger(__name__)


@dataclass
class PreflightCtx:
    """前置阶段共享上下文。`stopped=True` 表示 orchestrator 必须立即返回。"""

    conversation_id: str
    agent_id: str | None
    user_id: str
    agent: Any
    tracer: "LangSmithTracer"
    short_circuit_fn: Callable[..., Awaitable[list[dict]]]
    stopped: bool = False
    # 短路 reply 文本回写: orchestrator finally 兜底 fire post_process 用
    last_short_circuit_reply: str | None = None


async def resolve_pending_contradiction(
    user_message: str,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """spec §4 step 3-5：若有待解决矛盾，分析用户回答 → 应用解析 → 生成矛盾回复。

    调用链：memory.contradiction_analysis（§4.3）→ apply_contradiction_resolution
    （§4.4 降级原 L1 → L2）→ memory.contradiction_reply（§4.5 自然拉回话题）。
    """
    pending = await load_pending_contradiction(ctx.conversation_id)
    if not pending:
        return
    try:
        analysis = await analyze_contradiction_response(user_message, pending)
        await apply_contradiction_resolution(pending, analysis)
        await clear_pending_contradiction(ctx.conversation_id)
        personality_brief = ctx.agent.name if ctx.agent else "AI"
        reply = await generate_contradiction_reply(
            user_message=user_message,
            conflict=pending,
            analysis=analysis,
            personality_brief=personality_brief,
        )
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
    except Exception as e:
        logger.warning(f"Contradiction resolution failed: {e}")
        await clear_pending_contradiction(ctx.conversation_id)


async def resolve_pending_deletion(
    user_message: str,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """spec §5 step 3 + Phase 5: 若有待确认删除/改期, 根据用户回答执行或放弃."""
    pending = await load_pending_action(ctx.conversation_id)
    if not pending:
        return
    candidates = pending.get("candidates") or []
    action = pending.get("action") or "delete"
    new_time = pending.get("new_time")
    try:
        if is_deletion_confirmed(user_message):
            agent_name = ctx.agent.name if ctx.agent else "伙伴"
            preview = "\n".join(
                f"- {c.get('content', c.get('summary', ''))[:60]}"
                for c in candidates[:5]
            ) or "(无)"

            if action == "reschedule" and new_time:
                updated = await apply_reschedule(
                    ctx.user_id, candidates, new_time, agent_id=ctx.agent_id,
                )
                await clear_pending_deletion(ctx.conversation_id)
                reply = (
                    f"好嘞, 已经把以下 {updated} 件事挪到 {new_time} 啦~\n{preview}"
                    if updated
                    else "诶, 改期没成功, 你再说一遍?"
                )
            else:
                deleted = await execute_confirmed_deletion(ctx.user_id, candidates)
                await clear_pending_deletion(ctx.conversation_id)
                reply = (
                    await deletion_done_reply(
                        message=user_message,
                        personality_brief=agent_name,
                        deleted_memories=preview,
                    )
                    or await generate_deletion_reply(agent_name, "之前提到的", deleted)
                )
        else:
            await clear_pending_deletion(ctx.conversation_id)
            reply = (
                "好的，那就不改了，继续聊吧~" if action == "reschedule"
                else "好的，那就不删了，继续聊吧~"
            )
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
    except Exception as e:
        logger.warning(f"Deletion/reschedule confirmation failed: {e}")
        await clear_pending_deletion(ctx.conversation_id)
