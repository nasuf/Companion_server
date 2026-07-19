"""交互前置校验：pending 跨消息状态 → 边界分流（spec §2.6 / §4 / §5）。

把 orchestrator 进入主流程前的两个独立判定块抽出：

- `resolve_pending_contradiction`：spec §4 step 3-5，处理上一轮的矛盾追问回答
- `resolve_pending_deletion`：spec §5 step 3，处理上一轮的删除确认/取消

两者都是 AsyncGenerator：命中就 yield reply/done 事件并通过 `ctx.stopped=True`
告诉 orchestrator 终止本次流程；否则不产出任何事件，orchestrator 继续下一阶段。
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chat.tracing import ChatTracer

from app.observability.events import EVT_PREFLIGHT_FAILED, EVT_PREFLIGHT_RESOLVED
from app.services.chat.intent_replies import deletion_done_reply, record_confirm_reply
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
    save_pending_action,
)
from app.services.rules.chat_keywords import (
    CANCEL_CHOICE_ALL_KEYWORDS,
    CANCEL_CONFIRM_KEYWORDS,
    CANCEL_DENY_KEYWORDS,
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
    tracer: "ChatTracer"
    short_circuit_fn: Callable[..., Awaitable[list[dict]]]
    stopped: bool = False
    # 短路 reply 文本回写: orchestrator finally 兜底 fire post_process 用
    last_short_circuit_reply: str | None = None


async def discard_pending_states_for_crisis(conversation_id: str) -> None:
    """危机回合显式丢弃跨消息 pending 状态（矛盾追问 / 删除确认）。

    危机对话使追问上下文失效：用户脱离危机后的第一条消息绝不该被误解析成
    "对矛盾追问的回答"或"删除确认"。丢弃是安全的——矛盾在后续相关消息中
    会被重新检测，删除可由用户重新发起。清理失败不阻塞危机回复主路径。
    """
    try:
        await clear_pending_contradiction(conversation_id)
        await clear_pending_deletion(conversation_id)
    except Exception as e:
        logger.warning(
            f"crisis pending-state cleanup failed conv={conversation_id}: {e}",
            extra={"event": EVT_PREFLIGHT_FAILED, "stage": "crisis_discard"},
        )


# ═══════════════════════════════════════════════════════════════════
# Phase 0.2: Universal undo preflight — 统一 cancel_reminder + delete undo.
# 用户说"撤回/恢复"时优先检查 undo state, 不依赖任何 intent 路径
# ═══════════════════════════════════════════════════════════════════


async def resolve_recent_undo(
    user_message: str,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """Check if user wants to undo recent cancel-reminder OR delete.

    优先级: 在 contradiction/deletion pending 之前 check (用户说"撤回" 应该
    跳过任何 pending 状态). 1h 内可恢复, 都未命中则 no-op fall-through.
    """
    from app.services.chat.intent_handlers import (
        _is_undo_cancel, _undo_recent_cancel,
    )
    from app.services.memory.interaction.deletion import (
        load_delete_undo, clear_delete_undo, restore_deleted_memories,
    )

    if not _is_undo_cancel(user_message):
        return

    # 检查 delete undo state — 优先 (用户最可能记得"刚才删的")
    delete_undo = await load_delete_undo(ctx.conversation_id)
    if delete_undo:
        snapshots = delete_undo.get("snapshots") or []
        n = await restore_deleted_memories(snapshots)
        if n > 0:
            await clear_delete_undo(ctx.conversation_id)
            reply = f"嗯, 已经把刚才删除的 {n} 条记忆恢复啦~"
        else:
            reply = "诶, 帮你恢复的时候出了点问题, 你再确认一下?"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        logger.info(
            f"[PREFLIGHT-UNDO] restored {n} deleted memories",
            extra={"event": EVT_PREFLIGHT_RESOLVED, "kind": "undo_delete",
                   "n_restored": n},
        )
        return

    # 否则检查 cancel reminder undo state
    n = await _undo_recent_cancel(conversation_id=ctx.conversation_id)
    if n > 0:
        from app.services.reminder.scheduling import notify_reminder_changed
        await notify_reminder_changed(ctx.conversation_id, kind="restored")
        reply = (
            f"嗯, 已经把刚才取消的 {n} 个提醒都恢复啦~ "
            "如果还想取消, 跟我说一声哈."
        )
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        logger.info(
            f"[PREFLIGHT-UNDO] restored {n} cancelled reminders",
            extra={"event": EVT_PREFLIGHT_RESOLVED, "kind": "undo_cancel",
                   "n_restored": n},
        )
        return

    # 都没有 undo state — 友好告知 (避免用户困惑)
    reply = "嗯, 没有可撤回的操作哦 (1 小时内的取消/删除才能撤回)~"
    ctx.last_short_circuit_reply = reply
    for evt in await ctx.short_circuit_fn(
        reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
        trace_id=ctx.tracer.safe_trace_id,
    ):
        yield evt
    ctx.tracer.close()
    ctx.stopped = True


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
        logger.info(
            f"[PREFLIGHT] contradiction resolved change_type={analysis.get('change_type')}",
            extra={
                "event": EVT_PREFLIGHT_RESOLVED,
                "kind": "contradiction",
                "change_type": analysis.get("change_type"),
                "reply_text_len": len(reply),
            },
        )
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
    except Exception as e:
        logger.warning(
            f"Contradiction resolution failed: {e}",
            extra={"event": EVT_PREFLIGHT_FAILED, "kind": "contradiction",
                   "error_type": type(e).__name__},
        )
        await clear_pending_contradiction(ctx.conversation_id)


async def resolve_pending_deletion(
    user_message: str,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """spec §5 step 3 + Phase 5: 若有待确认删除/改期/补全提醒时间, 根据用户回答执行或放弃.

    pending.action ∈ {delete, reschedule, set_reminder, update_reminder_content}:
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
    if pending.get("action") == "cancel_reminder":
        async for evt in _handle_pending_cancel_reminder(user_message, pending, ctx):
            yield evt
        return
    if pending.get("action") == "update_reminder_content":
        async for evt in _handle_pending_update_reminder_content(user_message, pending, ctx):
            yield evt
        return
    candidates = pending.get("candidates") or []
    action = pending.get("action") or "delete"
    new_time = pending.get("new_time")
    try:
        # Phase 0.2: 多候选删除场景必须用数字选择, 不接受模糊"嗯"一刀切.
        # reschedule 仍走老逻辑 (改期是把所有 candidate 同时挪到 new_time, 语义
        # 是"批量改时间"不是"挑选删哪条", 一刀切是合理的).
        is_multi_delete = action == "delete" and len(candidates) > 1
        chosen_indices: list[int] | None = None

        if is_multi_delete:
            chosen_indices = _parse_user_choice(user_message, len(candidates))
            msg_clean = user_message.strip().lower()

            # 否定 → 取消
            if msg_clean in CANCEL_DENY_KEYWORDS or any(
                kw in user_message for kw in ("不是", "保留", "别动", "算了")
            ):
                await clear_pending_deletion(ctx.conversation_id)
                reply = "好的，那就不删了，继续聊吧~"
                ctx.last_short_circuit_reply = reply
                for evt in await ctx.short_circuit_fn(
                    reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
                    trace_id=ctx.tracer.safe_trace_id,
                ):
                    yield evt
                ctx.tracer.close()
                ctx.stopped = True
                return

            # 模糊 confirm ("嗯/好") 不接受 → 二次反问要求编号
            if chosen_indices is None and is_deletion_confirmed(user_message):
                preview_list = "\n".join(
                    f"{i + 1}) {c.get('content', c.get('summary', ''))[:60]}"
                    for i, c in enumerate(candidates[:5])
                )
                reply = (
                    f"诶, 我需要你说清楚删哪一条 (避免误删):\n{preview_list}\n"
                    "回数字 (如 '1' 或 '1和3'), '全部', 或 '算了'~"
                )
                # 不清 pending — 让用户继续在该 pending 状态回答
                ctx.last_short_circuit_reply = reply
                for evt in await ctx.short_circuit_fn(
                    reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
                    trace_id=ctx.tracer.safe_trace_id,
                ):
                    yield evt
                ctx.tracer.close()
                ctx.stopped = True
                return

            # 没解析出 indices 也不是 deny/confirm → 答非所问, 清 pending fall-through
            if chosen_indices is None:
                await clear_pending_deletion(ctx.conversation_id)
                logger.info(
                    f"[PREFLIGHT] deletion: no choice from {user_message[:30]!r}; "
                    "clearing pending, falling through to main reply"
                )
                return  # 不 stop, 让正常 reply

        # 单候选 / reschedule / 多候选+已选号: 按 confirmed 路径
        if chosen_indices is not None or is_deletion_confirmed(user_message):
            agent_name = ctx.agent.name if ctx.agent else "伙伴"
            # 用户实际选的 candidates (单候选 / reschedule = 全部, 多候选 = 编号)
            target_candidates = (
                [candidates[i] for i in chosen_indices]
                if chosen_indices is not None
                else candidates
            )
            preview = "\n".join(
                f"- {c.get('content', c.get('summary', ''))[:60]}"
                for c in target_candidates[:5]
            ) or "(无)"

            if action == "reschedule" and new_time:
                updated = await apply_reschedule(
                    ctx.user_id, target_candidates, new_time, agent_id=ctx.agent_id,
                )
                await clear_pending_deletion(ctx.conversation_id)
                reply = (
                    f"好嘞, 已经把以下 {updated} 件事挪到 {new_time} 啦~\n{preview}"
                    if updated
                    else "诶, 改期没成功, 你再说一遍?"
                )
                logger.info(
                    f"[PREFLIGHT] reschedule confirmed n_updated={updated}",
                    extra={"event": EVT_PREFLIGHT_RESOLVED, "kind": "reschedule",
                           "n_updated": updated, "new_time": new_time},
                )
            else:
                # Phase 0.2: 传 conversation_id 让 execute_confirmed_deletion 存
                # snapshot 到 Redis (1h 内可 undo).
                deleted = await execute_confirmed_deletion(
                    ctx.user_id, target_candidates,
                    conversation_id=ctx.conversation_id,
                )
                await clear_pending_deletion(ctx.conversation_id)
                # reply 加 undo 提示, 让用户知道有撤回机会
                undo_hint = " (1 小时内说'撤回刚才的删除' 还能恢复)"
                base_reply = (
                    await deletion_done_reply(
                        message=user_message,
                        personality_brief=agent_name,
                        deleted_memories=preview,
                    )
                    or await generate_deletion_reply(agent_name, "之前提到的", deleted)
                )
                reply = base_reply + undo_hint if deleted > 0 else base_reply
                logger.info(
                    f"[PREFLIGHT] deletion confirmed n_deleted={deleted}",
                    extra={"event": EVT_PREFLIGHT_RESOLVED, "kind": "deletion",
                           "n_deleted": deleted, "n_candidates": len(target_candidates)},
                )
        else:
            await clear_pending_deletion(ctx.conversation_id)
            reply = (
                "好的，那就不改了，继续聊吧~" if action == "reschedule"
                else "好的，那就不删了，继续聊吧~"
            )
            logger.info(
                f"[PREFLIGHT] {action} cancelled by user",
                extra={"event": EVT_PREFLIGHT_RESOLVED, "kind": action,
                       "outcome": "cancelled"},
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
# Phase 0.1: cancel_reminder 第二轮 — 用户回答"对/不是/数字/全部"
# ═══════════════════════════════════════════════════════════════════


def _parse_user_choice(message: str, n_candidates: int) -> list[int] | None:
    """从用户回复解析数字选择: '1' / '1和3' / '1,2' / '全部' / 'all'.

    返回 0-indexed 的下标列表, 或 None (没解析出数字).
    """
    import re
    msg = message.strip().lower()
    if msg in CANCEL_CHOICE_ALL_KEYWORDS:
        return list(range(n_candidates))
    nums = re.findall(r"\d+", msg)
    if not nums:
        return None
    indices = []
    for s in nums:
        try:
            idx = int(s) - 1  # 1-indexed → 0-indexed
            if 0 <= idx < n_candidates:
                indices.append(idx)
        except ValueError:
            continue
    return indices if indices else None


async def _handle_pending_cancel_reminder(
    user_message: str,
    pending: dict,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """cancel_reminder 第二轮: 用户回复"对"/"不是"/"数字"/"全部"/"算了"/answer-elsewhere.

    分支:
    - DENY ("不是/算了") → 清 pending + 友好放弃
    - 单候选 + CONFIRM ("对/嗯") → 撤 + undo 提示
    - 多候选 + 数字 ("1" / "1和3" / "全部") → 撤指定 + undo 提示
    - 多候选 + CONFIRM 但无数字 → 二次反问 (避免一刀切全删)
    - 答非所问 (没匹配上面任何) → 清 pending + 不阻塞主流程 (走正常 reply)
    """
    from app.services.chat.intent_handlers import _cancel_active_reminders
    from app.services.memory.interaction.deletion import clear_pending_deletion
    from app.services.reminder.scheduling import notify_reminder_changed

    candidates = pending.get("candidates") or []
    if not candidates:
        await clear_pending_deletion(ctx.conversation_id)
        return

    msg_clean = user_message.strip().lower()

    # 否定语义最高优先级 — 防误删
    if msg_clean in CANCEL_DENY_KEYWORDS or any(
        kw in user_message for kw in ("不是", "保留", "别动", "算了")
    ):
        await clear_pending_deletion(ctx.conversation_id)
        reply = "好嘞, 那就保留着, 该响的时候我喊你~"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        logger.info(
            f"[CANCEL-REMINDER-PENDING] user denied ({user_message[:30]!r})",
            extra={"event": EVT_PREFLIGHT_RESOLVED, "kind": "cancel_reminder",
                   "outcome": "denied"},
        )
        return

    # 数字选择 (多候选场景)
    chosen_indices = _parse_user_choice(user_message, len(candidates))

    # 单候选 + CONFIRM (无需选择).
    # 长回复 (>6 字) 含 "对/撤/好" 容易假阳 (e.g. "对方有事别提了" 含"对"),
    # 限定短回复 (≤6 字) 才走 loose match. 即使误判, 1h 内可 undo, 安全网兜底.
    if len(candidates) == 1 and (
        msg_clean in CANCEL_CONFIRM_KEYWORDS
        or (len(msg_clean) <= 6 and any(kw in user_message for kw in ("对", "撤", "好")))
    ):
        chosen_indices = [0]

    # 多候选 + CONFIRM 但用户没指定数字 → 二次反问 (不一刀切)
    if len(candidates) > 1 and chosen_indices is None and msg_clean in CANCEL_CONFIRM_KEYWORDS:
        reply = (
            "嗯, 你想撤哪个呀? 回数字 (如 '1', '1和3'), "
            "或者'全部', 或者'算了'~"
        )
        # 不清 pending — 让用户继续在该 pending 状态回答
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    # 答非所问 → 放弃 pending, 让主流程接管 (用户在聊别的)
    if chosen_indices is None:
        await clear_pending_deletion(ctx.conversation_id)
        logger.info(
            f"[CANCEL-REMINDER-PENDING] no choice parsed from {user_message[:30]!r}; "
            "clearing pending, falling through to main reply"
        )
        return  # 不 stop, 让 user_message 走正常 orchestrator

    # 执行 cancel
    chosen_triggers = [candidates[i]["trigger_id"] for i in chosen_indices]
    n = await _cancel_active_reminders(
        user_id=ctx.user_id,
        agent_id=ctx.agent_id,
        trigger_ids=chosen_triggers,
        user_message=user_message,
        conversation_id=ctx.conversation_id,
    )
    await clear_pending_deletion(ctx.conversation_id)

    if n > 0:
        await notify_reminder_changed(ctx.conversation_id, kind="cancelled")
        if n == 1:
            chosen_summary = candidates[chosen_indices[0]]["summary"]
            chosen_when = candidates[chosen_indices[0]]["when_text"]
            reply = (
                f"好嘞, 已经把'{chosen_summary}'({chosen_when})撤掉啦~ "
                "1 小时内说'撤回'还能恢复."
            )
        else:
            reply = f"好嘞, 已经把 {n} 个提醒都撤掉啦~ 1 小时内说'撤回'能全部恢复."
    else:
        reply = "诶, 帮你撤的时候出了点小问题, 你再说一遍?"

    logger.info(
        f"[CANCEL-REMINDER-PENDING] cancelled {n} of {len(candidates)} "
        f"({user_message[:30]!r})",
        extra={"event": EVT_PREFLIGHT_RESOLVED, "kind": "cancel_reminder",
               "outcome": "executed", "n_cancelled": n},
    )
    ctx.last_short_circuit_reply = reply
    for evt in await ctx.short_circuit_fn(
        reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
        trace_id=ctx.tracer.safe_trace_id,
    ):
        yield evt
    ctx.tracer.close()
    ctx.stopped = True


async def _handle_pending_update_reminder_content(
    user_message: str,
    pending: dict,
    ctx: PreflightCtx,
) -> AsyncGenerator[dict, None]:
    """修改提醒内容第二轮: 多 active reminder 时接住用户的数字选择。"""
    from prisma import Json
    from app.db import db
    from app.services.chat.intent_handlers import extract_reminder_content_update
    from app.services.memory.storage import repo as memory_repo
    from app.services.reminder.scheduling import notify_reminder_changed

    candidates = pending.get("candidates") or []
    if not candidates:
        await clear_pending_deletion(ctx.conversation_id)
        return

    msg_clean = user_message.strip().lower()
    if msg_clean in CANCEL_DENY_KEYWORDS or any(
        kw in user_message for kw in ("不是", "保留", "别动", "算了", "不改")
    ):
        await clear_pending_deletion(ctx.conversation_id)
        reply = "好的，那就不改了。"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    explicit_content = extract_reminder_content_update(user_message)
    chosen_indices = _parse_user_choice(user_message, len(candidates))
    if chosen_indices is None and len(candidates) == 1 and explicit_content:
        chosen_indices = [0]
    if chosen_indices is None and len(candidates) == 1 and not explicit_content:
        chosen_indices = [0]
    if not chosen_indices:
        preview_list = "\n".join(
            f"{i + 1}) {c.get('summary', '')[:60]}"
            for i, c in enumerate(candidates[:5])
        )
        reply = f"我需要你说清楚改哪一个:\n{preview_list}\n回数字就行。"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return
    if len(chosen_indices) != 1:
        reply = "提醒内容一次先改一个，回一个数字就行。"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    choice_only = bool(re.fullmatch(r"\s*(?:第?\s*)?\d+\s*", user_message.strip()))
    content = explicit_content or str(pending.get("summary") or "").strip()
    if not content and len(candidates) == 1 and not choice_only:
        content = user_message.strip()
    if not content:
        selected = candidates[chosen_indices[0]]
        await save_pending_action(
            ctx.conversation_id,
            action="update_reminder_content",
            candidates=[selected],
            summary="",
        )
        reply = "好，提醒内容想改成哪一句?"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    selected = candidates[chosen_indices[0]]
    summary = f"我让 AI 提醒: {content[:120]}"
    action_data = dict(selected.get("action_data") or {})
    action_data["summary"] = summary
    await db.timetrigger.update(
        where={"id": selected["trigger_id"]},
        data={"actionData": Json(action_data)},
    )
    memory_id = selected.get("memory_id")
    if memory_id:
        try:
            await memory_repo.update(
                memory_id,
                source=selected.get("memory_side") or "user",
                content=summary,
                summary=summary,
            )
        except Exception as e:
            logger.warning(f"[REMINDER-CONTENT-PENDING] memory update failed: {e}")
    await notify_reminder_changed(ctx.conversation_id, kind="updated")
    await clear_pending_deletion(ctx.conversation_id)

    reply = f"好，已把第 {chosen_indices[0] + 1} 个提醒内容改成「{content[:60]}」。"
    ctx.last_short_circuit_reply = reply
    for evt in await ctx.short_circuit_fn(
        reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
        trace_id=ctx.tracer.safe_trace_id,
    ):
        yield evt
    ctx.tracer.close()
    ctx.stopped = True


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
    from app.services.chat.intent_handlers import (
        classify_cancel_intent,
        classify_record_request_action,
        extract_reminder_content_update,
    )
    from app.services.reminder.scheduling import create_user_reminder
    from app.services.workspace.workspaces import get_active_workspace

    summary = pending.get("summary") or "(未指定事项)"

    # 用户在反问时间之后补的是“提醒内容/文案”，这不是取消，也不应清掉 pending。
    # 保留原待办状态，更新 summary 后继续要时间。
    if classify_record_request_action(user_message) == "reminder_content":
        content = extract_reminder_content_update(user_message)
        if content:
            from app.services.memory.interaction.deletion import save_pending_action
            await save_pending_action(
                ctx.conversation_id,
                action="set_reminder",
                summary=content[:200],
            )
            summary = content
        reply = "好，提醒内容我按这句来。具体什么时候提醒你?"
        ctx.last_short_circuit_reply = reply
        for evt in await ctx.short_circuit_fn(
            reply, ctx.conversation_id, ctx.agent_id, ctx.user_id,
            trace_id=ctx.tracer.safe_trace_id,
        ):
            yield evt
        ctx.tracer.close()
        ctx.stopped = True
        return

    # 分支 1: 取消语义 — pending set_reminder 还没建 trigger, 用 high+low 都接受
    # (避免用户回 "算了" 时被当成"答非所问")
    if classify_cancel_intent(user_message) != "none":
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
        # 智能 when_text — 同天显示相对/绝对, 不再死板"05月02日 22:50"
        from app.services.reminder.scheduling import format_when_text
        when_text = format_when_text(occur_time, now=parse_now)
        # 走 LLM 确认回复 (跟第一轮一致, 不要硬编码模板). LLM 失败 fallback 模板.
        agent_name = ctx.agent.name if ctx.agent else "AI"
        reply = await record_confirm_reply(
            summary=summary[:120],
            when_text=when_text,
            is_recurring=False,
            personality_brief=agent_name,
        ) or f"好嘞, {when_text}叫你, 记好啦~"
        logger.info(
            f"[SET-REMINDER-PENDING] scheduled at {occur_time.isoformat()} "
            f"when={when_text!r} summary={summary[:30]!r}"
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
