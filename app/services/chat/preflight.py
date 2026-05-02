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
from app.services.schedule_domain.time_parser import (
    parse_loose_offset,
    parse_with_statement_time,
)
from app.services.schedule_domain.time_service import _now_corrected

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
    """spec §5 step 3 + Phase 5: 若有待确认删除/改期/补全提醒时间, 根据用户回答执行或放弃.

    pending.action ∈ {delete, reschedule, set_reminder}:
    - delete / reschedule: 旧路径, 用户确认/放弃删除或改期
    - set_reminder: Round-3 工程扩展, 第一轮 RECORD_REQUEST 时间没说清存的 pending,
      第二轮根据用户回答 (给具体时间 / 取消 / 答非所问) 分发到 _handle_pending_set_reminder
    """
    pending = await load_pending_action(ctx.conversation_id)
    if not pending:
        return
    if pending.get("action") == "set_reminder":
        async for evt in _handle_pending_set_reminder(user_message, pending, ctx):
            yield evt
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


# ═══════════════════════════════════════════════════════════════════
# Round-3 工程扩展: RECORD_REQUEST 时间没说清, 第二轮反问后的 4 分支处理
# ═══════════════════════════════════════════════════════════════════


async def _handle_pending_set_reminder(
    user_message: str,
    pending: dict,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """RECORD_REQUEST 反问后第二轮: 用户回答时间 / 取消 / 答非所问.

    4 分支:
    1. 取消语义 (_is_cancel_reminder 命中) → 清 pending + 友好放弃 + stop
    2. parse 出 future 时间 → 用 pending.summary + 新时间落库 + 确认 + stop
    3. parse 出非 future 时间 (含过去时刻 / 模糊词只解出当前) → 清 pending +
       提示"先不记" + stop. 防无限反问 loop.
    4. 完全没时间表达 (用户在聊别的) → 清 pending + 不阻塞主流程 (不 stop,
       让 user_message 走正常意图识别 + 回复)
    """
    from app.services.chat.intent_handlers import _is_cancel_reminder
    from app.services.reminder.scheduling import create_user_reminder
    from app.services.workspace.workspaces import get_active_workspace

    summary = pending.get("summary") or "(未指定事项)"

    # 分支 1: 取消语义
    if _is_cancel_reminder(user_message):
        await clear_pending_deletion(ctx.conversation_id)
        reply = "嗯嗯, 那不记了, 想起来再说~"
        logger.info(f"[SET-REMINDER-PENDING] user cancelled: {user_message[:30]!r}")
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    # 解析时间. parse_with_statement_time + parse_loose_offset 兜底.
    parse_now = _now_corrected()
    parsed = parse_with_statement_time(user_message, now=parse_now)
    future_events = [e for e in parsed.event_times if e.is_future]
    occur_time = None
    if future_events:
        occur_time = sorted(e.start for e in future_events)[0]
    else:
        loose = parse_loose_offset(user_message, parse_now)
        if loose is not None:
            occur_time = loose

    # 分支 4: 完全没时间表达 → 用户答非所问. 清 pending + 不阻塞主流程.
    # 判别: 任何 event_times (含 past/present) 都没抽到 + parse_loose_offset 也 None
    has_any_time_expr = bool(parsed.event_times) or occur_time is not None
    if not has_any_time_expr:
        await clear_pending_deletion(ctx.conversation_id)
        logger.info(
            f"[SET-REMINDER-PENDING] user message has no time expression "
            f"({user_message[:30]!r}); cleared pending, not blocking main flow"
        )
        # 不 stop, 不发回复 — 让 user_message 走正常 orchestrator
        return

    # 分支 3: 有时间表达但不是 future (用户给的还是模糊 / 过去时刻 / 已过期)
    # 防无限反问 loop, 直接清 pending + 提示让用户重新说.
    if occur_time is None:
        await clear_pending_deletion(ctx.conversation_id)
        reply = "嗯, 这个时间不太明确, 先不记了, 你想清楚再跟我说哈~"
        logger.info(
            f"[SET-REMINDER-PENDING] time expression but no future occur_time "
            f"({user_message[:30]!r}); declining to re-ask"
        )
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    # 分支 2: 解析成功 → 落库 + 确认
    if not ctx.user_id or not ctx.agent_id:
        # 没 agent — 异常 case (preflight 应该总有), 清 pending + 通用回复
        await clear_pending_deletion(ctx.conversation_id)
        reply = "诶, 这边记的时候出了点小问题, 你再跟我说一遍吧~"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    workspace = await get_active_workspace(
        user_id=ctx.user_id, agent_id=ctx.agent_id,
    )
    workspace_id = workspace.id if workspace else None

    memory_id = await create_user_reminder(
        user_id=ctx.user_id,
        agent_id=ctx.agent_id,
        workspace_id=workspace_id,
        summary=f"我让 AI 提醒: {summary[:120]}",
        occur_time=occur_time,
        statement_time=parse_now,
        recurrence="once",  # 反问场景默认一次性 — 用户没说重复关键词
    )

    await clear_pending_deletion(ctx.conversation_id)

    if memory_id:
        # when_text 形如 "8 月 1 日 10:00"
        from app.services.schedule_domain.time_service import _TZ
        local = occur_time.astimezone(_TZ)
        when_text = local.strftime("%m月%d日 %H:%M")
        reply = f"好嘞, {when_text}叫你, 记好啦~"
        logger.info(
            f"[SET-REMINDER-PENDING] scheduled at {occur_time.isoformat()} "
            f"summary={summary[:30]!r}"
        )
    else:
        reply = "诶, 这边记的时候出了点小问题, 你再跟我说一遍吧~"
        logger.warning(
            f"[SET-REMINDER-PENDING] create_user_reminder returned None for "
            f"summary={summary[:30]!r}"
        )

    ctx.last_short_circuit_reply = reply
    for evt in await ctx.short_circuit_fn(
        reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
        trace_id=ctx.tracer.safe_trace_id,
    ):
        yield evt
    ctx.tracer.close()
    ctx.stopped = True
