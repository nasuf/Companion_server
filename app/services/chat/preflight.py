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
from typing import Any, Awaitable, Callable

from app.services.chat.intent_replies import deletion_done_reply
from app.services.memory.interaction.contradiction import (
    analyze_contradiction_response,
    apply_contradiction_resolution,
    clear_pending_contradiction,
    load_pending_contradiction,
)
from app.services.memory.interaction.deletion import (
    clear_pending_deletion,
    execute_confirmed_deletion,
    generate_deletion_reply,
    is_deletion_confirmed,
    load_pending_deletion,
)

logger = logging.getLogger(__name__)


@dataclass
class PreflightCtx:
    """前置阶段共享上下文。`stopped=True` 表示 orchestrator 必须立即返回。"""

    conversation_id: str
    agent_id: str | None
    user_id: str
    agent: Any
    trace_ctx: Any
    trace_id: str | None
    end_trace_fn: Callable[..., None]
    short_circuit_fn: Callable[..., Awaitable[list[dict]]]
    stopped: bool = False


async def resolve_pending_contradiction(
    user_message: str,
    ctx: PreflightCtx,
    chat_invoke_fn: Callable[[str], Awaitable[str]],
) -> AsyncGenerator[dict, None]:
    """spec §4 step 3-5：若有待解决矛盾，分析用户回答 → 应用解析 → 回复。"""
    pending = await load_pending_contradiction(ctx.conversation_id)
    if not pending:
        return
    try:
        analysis = await analyze_contradiction_response(user_message, pending)
        await apply_contradiction_resolution(pending, analysis)
        await clear_pending_contradiction(ctx.conversation_id)
        change = analysis.get("change_type", "新增")
        reason = analysis.get("reason", "")
        agent_name = ctx.agent.name if ctx.agent else "AI"
        try:
            reply = await chat_invoke_fn(
                f"你是{agent_name}。用户刚才解释了一个信息变化：{reason}（类型：{change}）。"
                f"请用1-2句自然的话回应，表示你记住了新的情况，然后把话题拉回正轨。"
                f"不要提及'记忆''修改''系统'等词。"
            )
        except Exception:
            reply = "好的，我记住了~"
        for evt in await ctx.short_circuit_fn(reply, ctx.conversation_id, ctx.agent_id, ctx.user_id):
            yield evt
        ctx.end_trace_fn(ctx.trace_ctx, ctx.trace_id, ctx.conversation_id)
        ctx.stopped = True
    except Exception as e:
        logger.warning(f"Contradiction resolution failed: {e}")
        await clear_pending_contradiction(ctx.conversation_id)


async def resolve_pending_deletion(
    user_message: str,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """spec §5 step 3：若有待确认删除，根据用户回答执行删除或放弃。"""
    pending_del = await load_pending_deletion(ctx.conversation_id)
    if not pending_del:
        return
    try:
        if is_deletion_confirmed(user_message):
            deleted = await execute_confirmed_deletion(ctx.user_id, pending_del)
            await clear_pending_deletion(ctx.conversation_id)
            agent_name = ctx.agent.name if ctx.agent else "伙伴"
            deleted_preview = "\n".join(
                f"- {c.get('content', c.get('summary', ''))[:60]}"
                for c in pending_del[:5]
            ) or "(无)"
            reply = (
                await deletion_done_reply(
                    message=user_message,
                    personality_brief=agent_name,
                    deleted_memories=deleted_preview,
                )
                or await generate_deletion_reply(agent_name, "之前提到的", deleted)
            )
        else:
            await clear_pending_deletion(ctx.conversation_id)
            reply = "好的，那就不删了，继续聊吧~"
        for evt in await ctx.short_circuit_fn(reply, ctx.conversation_id, ctx.agent_id, ctx.user_id):
            yield evt
        ctx.end_trace_fn(ctx.trace_ctx, ctx.trace_id, ctx.conversation_id)
        ctx.stopped = True
    except Exception as e:
        logger.warning(f"Deletion confirmation failed: {e}")
        await clear_pending_deletion(ctx.conversation_id)
