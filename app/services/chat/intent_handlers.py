"""Spec §3.4 各意图的短路处理器。

从 `orchestrator.stream_chat_response` 中抽出 7 个意图分支：每个 handler
只关心自己的输入/参考信息 + 生成 reply，尾部统一交给 `finalize_short_circuit`。

handler 作为 async generator 产出 SSE 事件，orchestrator 只需 `async for ...: yield`。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chat.tracing import LangSmithTracer

from app.services.chat.intent_replies import (
    apology_reply,
    current_state_reply,
    deletion_confirm_reply,
    end_reply,
    record_confirm_reply,
    schedule_query_reply,
)
from app.services.chat.multi_intent import finalize_short_circuit
from app.services.memory.interaction.deletion import (
    detect_deletion_intent,
    find_matching_memories,
    generate_deletion_confirmation_prompt,
    save_pending_action,
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
from app.services.schedule_domain.time_service import resolve_implicit_time

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
    # 短路 handler 经 ctx.finalize(reply) 把回复文本回写到这里, 让 orchestrator
    # finally 兜底 fire post_process 时拿到正确的 full_response (否则 short-circuit
    # 路径直接 return, post_process 永不跑, 记忆/PAD/trait 全丢失).
    last_short_circuit_reply: str | None = None
    # spec §3.3 step 2 假设多意图能字面拆分 (示例"我好难过，不想聊了" → 2 个独立
    # 子句). 但生产场景的口语融合句 (e.g. "算了别提醒了, 我吃过了" 整体语义 =
    # 取消) 子片段单独看意图相反, 强行 sub-intent 处理会反向回复. handler
    # 在主意图消化整句时 (典型: RECORD_REQUEST 取消分支) 设此 flag = True,
    # finalize 跳过 sub-intent 递归. 详见 CLAUDE.md §6 偏离表.
    consumed_full_message: bool = False

    async def finalize(self, reply: str) -> AsyncGenerator[dict, None]:
        self.last_short_circuit_reply = reply
        # consumed_full_message=True 时清空 sub fragments, 让 finalize_short_circuit
        # 跳过 process_sub_intents 递归. 比加新参数到 finalize 签名干净.
        sub_fragments = (
            {} if self.consumed_full_message else self.pending_sub_fragments
        )
        async for evt in finalize_short_circuit(
            reply,
            conversation_id=self.conversation_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
            agent=self.agent,
            reply_context=self.reply_context,
            tracer=self.tracer,
            save_replies_fn=self.save_replies_fn,
            pending_sub_fragments=sub_fragments,
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
    """spec §5 删除 + Phase 5 改期合一: detect_deletion_intent 同时识别两类,
    intent ∈ {delete, reschedule} 决定 pending shape 与 confirmation 文案."""
    try:
        deletion_result = await detect_deletion_intent(user_message)
        description = (deletion_result or {}).get("target_description")
        if not description:
            return False, None

        candidates = await find_matching_memories(ctx.user_id, description)
        agent_name = ctx.agent.name if ctx.agent else "伙伴"
        if not candidates:
            return True, ctx.finalize("嗯...我好像没有关于这个的记忆呢。")

        intent = (deletion_result or {}).get("intent") or "delete"
        new_time_raw = (deletion_result or {}).get("new_time")
        candidate_preview = "\n".join(
            f"{i + 1}. {c.get('content', c.get('summary', ''))[:60]}"
            for i, c in enumerate(candidates[:5])
        )

        if intent == "reschedule" and new_time_raw:
            await save_pending_action(
                ctx.conversation_id,
                action="reschedule",
                candidates=candidates,
                new_time=new_time_raw,
            )
            # 简短自然: "你想把 X / Y 挪到 {time} 对吗?". 不再调 LLM (Phase 5
            # 走最小扩展, prompt 加 reschedule 选项不值得为 confirm 文案再加 prompt).
            reply = (
                f"你想把这些挪到 {new_time_raw} 对吗?\n{candidate_preview}\n"
                "回我「对/好」我就改, 回「算了」就保持."
            )
        else:
            await save_pending_deletion(ctx.conversation_id, candidates)
            reply = (
                await deletion_confirm_reply(
                    message=user_message,
                    personality_brief=agent_name,
                    candidate_memories=candidate_preview,
                )
                or await generate_deletion_confirmation_prompt(agent_name, candidates)
            )
        return True, ctx.finalize(reply)
    except Exception as e:
        logger.warning(f"Hot-path deletion/reschedule failed, falling through: {e}")
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
    # spec §3.2 隐性时间解析: 走时间中枢 helper, 复用 caller 已加载的 ai_status
    _, current_activity = await resolve_implicit_time(ctx.agent_id or "", ai_status)
    try:
        response = await current_state_reply(
            message=user_message,
            user_emotion=user_emotion,
            personality_brief=_agent_name(ctx.agent),
            user_portrait=str(portrait) if portrait else "(未知)",
            current_activity=current_activity,
            ai_schedule=schedule_context or "(未知)",
        )
        if not response:
            return False, None
        return True, ctx.finalize(response)
    except Exception as e:
        logger.warning(f"Current state short-circuit failed, falling through: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════
# 工程扩展 §3.4 + Part 5 §4.2: 记录请求 (RECORD_REQUEST)
# ═══════════════════════════════════════════════════════════════════


def _format_when_text(occur_dt) -> str:
    """把 datetime 渲染成人话, 给 confirm prompt 用."""
    from app.services.schedule_domain.time_service import _TZ
    local = occur_dt.astimezone(_TZ)
    return local.strftime("%m月%d日 %H:%M")


# 周期性提醒关键词识别. 用户口语 "每天提醒我吃药" / "每月 1 号交房租" 走
# RECORD_REQUEST 短路时 _direct_create_reminder 必须识别这些, 否则硬编码
# recurrence="once" 让周期信息丢失, 提醒只响一次. 关键词级判断, 不调 LLM.
_RECURRENCE_KEYWORDS: tuple[tuple[str, str], ...] = (
    # 顺序重要 — "每年" 必须在 "每" 单字之前匹配
    ("每年", "yearly"), ("每月", "monthly"), ("每周", "weekly"),
    ("每星期", "weekly"), ("每天", "daily"), ("每日", "daily"),
    ("每晚", "daily"), ("每早", "daily"), ("每个月", "monthly"),
    ("每一年", "yearly"), ("每一月", "monthly"), ("每一周", "weekly"),
    ("每一天", "daily"),
)


def _detect_recurrence(message: str) -> str:
    """从用户消息识别 recurrence (once/daily/weekly/monthly/yearly).
    `_RECURRENCE_KEYWORDS` 命中即返对应周期, 否则 "once".
    """
    msg = message.strip()
    for kw, rec in _RECURRENCE_KEYWORDS:
        if kw in msg:
            return rec
    return "once"


async def _persist_one_reminder(
    *,
    user_id: str,
    workspace_id: str | None,
    summary: str,
    occur_time: datetime,
    statement_time: datetime,
    recurrence: str,
) -> str | None:
    """落库一条 reminder memory + 建/更新对应 timetrigger. 返回 memory_id (失败 None).

    内部封装 dedup 复用语义: store_memory 命中 dedup 返 None → find_duplicate_id
    拿 existing id + update occurTime → 用 existing id 继续建 trigger.
    `upsert_reminder_trigger` 自身幂等 (existing trigger update 而非 silent skip).
    """
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.storage.embedding import generate_embedding
    from app.services.memory.storage.persistence import find_duplicate_id, store_memory

    try:
        memory_id = await store_memory(
            user_id=user_id,
            content=summary,
            summary=summary,
            level=3,
            importance=0.45,  # 落 L3 (pipeline clamp 也是 [0.4, 0.49])
            memory_type="life",
            main_category="生活",
            sub_category="提醒",
            occur_time=occur_time,
            statement_time=statement_time,
            workspace_id=workspace_id,
            source="user",
            recurrence=recurrence,
        )
    except Exception as e:
        logger.warning(f"[RECORD-REQ] store_memory failed: {e}")
        return None

    if not memory_id:
        # dedup 命中 → 复用 existing memory_id, 更新 occurTime 到新时刻 (重设语义)
        try:
            embedding = await generate_embedding(summary)
            memory_id = await find_duplicate_id(
                user_id, summary, embedding, workspace_id=workspace_id,
            )
            if memory_id:
                await memory_repo.update(
                    memory_id, source="user",
                    occurTime=occur_time, statementTime=statement_time,
                )
                logger.info(
                    f"[RECORD-REQ] reusing deduped memory={memory_id[:8]} "
                    f"updated occurTime={occur_time} recurrence={recurrence}"
                )
        except Exception as e:
            logger.warning(f"[RECORD-REQ] dedup fallback failed: {e}")
            return None

    if not memory_id:
        logger.warning(
            "[RECORD-REQ] both store_memory and dedup lookup failed; no reminder"
        )
        return None
    return memory_id


async def _direct_create_reminder(
    *, user_message: str, ctx: ShortCircuitCtx,
) -> str | None:
    """同步用 time_parser 抽 future event_time → 直接 store_memory + 建 timetrigger.

    返回人话 when_text 给 confirm reply 用; 没抽到时间则返 None (调用方 fallback
    到模糊确认 "我帮你记上了" 不带时间).

    支持:
    - 单 occur_time ("一分钟后提醒X")
    - 多 occur_time ("8 点吃药, 9 点开会") — 各建 1 条 trigger, when_text 合并为
      "08:00 / 09:00" 格式
    - 周期性 ("每天提醒X" / "每月 1 号Y") — 关键词识别 recurrence, 通过 trigger
      handler 周期续期
    - 缺"后"字口语 ("一分钟提醒X") — _record_request_loose_offset fallback

    架构: 跨模块依赖 (memory + workspace + reminder/scheduling) 通过本文件 inline
    import 隔离, 真正 reminder 落地逻辑都收口在 `services/reminder/scheduling.py`.
    """
    import asyncio
    from datetime import datetime
    from app.services.reminder.scheduling import upsert_reminder_trigger
    from app.services.schedule_domain.time_parser import (
        parse_loose_offset, parse_with_statement_time,
    )
    from app.services.schedule_domain.time_service import _now_corrected
    from app.services.workspace.workspaces import get_active_workspace

    if not ctx.user_id:
        return None

    # parse 必须用"用户消息接收时刻"而不是 handler 跑到这一行的时刻 — 链路上前面
    # 的 LLM 调用累计 ~25s, "两分钟后" 实际算成 "处理完 + 2分钟" 比期望晚.
    received_at: datetime | None = None
    if ctx.reply_context:
        raw = ctx.reply_context.get("received_at")
        if raw:
            try:
                received_at = datetime.fromisoformat(str(raw))
            except (TypeError, ValueError):
                received_at = None
    parse_now = received_at or _now_corrected()
    statement_time = parse_now

    recurrence = _detect_recurrence(user_message)

    parsed = parse_with_statement_time(user_message, now=parse_now)
    future_events = [e for e in parsed.event_times if e.is_future]
    occur_times: list[datetime] = []
    if future_events:
        # 多 occur_time 全保留 ("8 点吃药, 9 点开会"), 按时间升序
        occur_times = sorted(e.start for e in future_events)
    else:
        loose = parse_loose_offset(user_message, parse_now)
        if loose is not None:
            occur_times = [loose]

    if not occur_times:
        logger.info(
            f"[RECORD-REQ] no future event time parsed from {user_message[:60]!r}; "
            "falling back to fuzzy confirmation"
        )
        return None

    # workspace 仅查一次; .id 直接拿避免 resolve_workspace_id 二次 find_first.
    workspace = await get_active_workspace(
        user_id=ctx.user_id, agent_id=ctx.agent_id,
    )
    if not workspace:
        logger.warning(
            f"[RECORD-REQ] no active workspace for user={ctx.user_id[:8]}; "
            "reminder will NOT fire"
        )
        return None
    agent_id = (
        getattr(workspace, "agentId", None)
        or getattr(workspace, "ai_agent_id", None)
    )
    if not agent_id:
        logger.warning(
            "[RECORD-REQ] workspace has no agentId; reminder will NOT fire"
        )
        return None
    workspace_id = workspace.id

    # 每个 occur_time 各建一条 (memory + trigger). 多条时并发建 (gather), N=1 也走
    # 同一路径无副作用. summary 第一人称 ("我让 AI 提醒") 让用户在记忆面板看到时
    # 不错位; (N/M) 后缀防 dedup 误合并.
    async def _schedule_one(idx: int, ot: datetime) -> datetime | None:
        suffix = f" ({idx + 1}/{len(occur_times)})" if len(occur_times) > 1 else ""
        summary = f"我让 AI 提醒: {user_message[:120]}{suffix}"
        memory_id = await _persist_one_reminder(
            user_id=ctx.user_id,
            workspace_id=workspace_id,
            summary=summary,
            occur_time=ot,
            statement_time=statement_time,
            recurrence=recurrence,
        )
        if not memory_id:
            return None
        await upsert_reminder_trigger(
            user_id=ctx.user_id,
            agent_id=agent_id,
            memory_id=memory_id,
            summary=summary,
            trigger_time=ot,
            recurrence=recurrence,
            side="user",
        )
        return ot

    results = await asyncio.gather(
        *(_schedule_one(i, ot) for i, ot in enumerate(occur_times)),
        return_exceptions=False,
    )
    scheduled = [r for r in results if r is not None]

    if not scheduled:
        return None

    logger.info(
        f"[RECORD-REQ] {len(scheduled)} reminder(s) scheduled "
        f"recurrence={recurrence} times={[t.isoformat() for t in scheduled]}"
    )
    if len(scheduled) == 1:
        return _format_when_text(scheduled[0])
    return " / ".join(_format_when_text(t) for t in sorted(scheduled))


# 取消提醒的口语关键词. 用户说"算了别提醒了"/"不用提醒了"/"取消那个提醒" 时
# 走取消分支, 直接 deactivate 该用户最近 active reminder + 简短确认.
# 不依赖 LLM (这种短句关键词足够明确), 跟 RECORD_REQUEST 主路径互斥.
#
# 直接命中: 完整短语
_CANCEL_DIRECT_KEYWORDS = (
    "算了", "别提醒", "不用提醒", "不用了",
    "不记了", "别记了", "我吃过了", "我做过了", "已经做了",
    "已经吃了", "已经喝了",
)
# 共现命中: 任一负向词 + "提醒"/"记" 同时出现 (覆盖"取消那个提醒"/"删掉那个提醒"等)
_CANCEL_NEG_TOKENS = ("取消", "删掉", "删除", "撤销", "不要")


def _is_cancel_reminder(message: str) -> bool:
    """口语 + 关键词级别的取消语义判定 (不调 LLM, ~0ms).

    仅 RECORD_REQUEST intent 已确认时调用; 不会跟正常聊天里的"算了"误匹配.
    """
    msg = message.strip()
    if any(kw in msg for kw in _CANCEL_DIRECT_KEYWORDS):
        return True
    # 共现规则: "取消X提醒" / "删掉提醒" 等 (X 可以是"那个"/"这个"/空)
    if any(neg in msg for neg in _CANCEL_NEG_TOKENS) and ("提醒" in msg or "记" in msg):
        return True
    return False


async def _cancel_active_reminders(
    *, user_id: str, agent_id: str | None = None,
) -> int:
    """deactivate 该 (user, agent) 的所有 active reminder timetrigger. 返回处理条数.

    Round-2 review bug fix: 必须按 agent_id 过滤. 不传则跨 agent 误删 (用户在
    agent A 上说"算了别提醒"会把 agent B 的所有 active reminder 一起 deactivate).
    handler 拿 ctx.agent_id 传入.

    用户说"算了别提醒了"时调用. 不用 LLM 判别要取消哪一个 — 用户当前对话
    的语境通常只关心最近一次刚设的, 一刀切 deactivate 所有 active 是最朴素
    可靠的选择.
    """
    from app.services.reminder.scheduling import deactivate_reminder_triggers
    return await deactivate_reminder_triggers(
        user_id=user_id, agent_id=agent_id,
    )


async def handle_record_request(
    user_message: str,
    ctx: ShortCircuitCtx,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    """RECORD_REQUEST 短路: 用户请求 AI 记一件事 / 设提醒 / 取消提醒.

    分支:
    - 取消语义 (含"算了/别提醒/我吃过了" 等) → deactivate user 所有 active
      reminder + 短确认. 跳过 LLM, 体感: "好嘞, 那就不提醒了~"
    - 设置语义 → _direct_create_reminder (规则引擎 parse 时间 + 同步建 trigger)

    背景: 生产观察到 LLM 把"算了别提醒了, 我吃过了"硬塞到 RECORD_REQUEST + 拆出
    "计划查询" sub-intent → AI 回了一堆离题的话. 加取消分支让取消语义有正确
    路径处理.
    """
    try:
        # 取消语义优先 — 即使 LLM intent 输出 "记录请求", 关键词命中也走取消.
        if _is_cancel_reminder(user_message) and ctx.user_id:
            # 必须传 agent_id, 否则跨 agent 误删 (round-2 review bug fix)
            n = await _cancel_active_reminders(
                user_id=ctx.user_id, agent_id=ctx.agent_id,
            )
            if n > 0:
                reply = "好嘞, 那就不提醒了, 已经帮你撤掉啦~"
            else:
                # 没 active reminder 可取消 — 用户可能误说或上下文断了
                reply = "嗯嗯, 我没在帮你记什么提醒哦, 不用担心~"
            logger.info(
                f"[RECORD-CANCEL] user said cancel ({user_message[:30]!r}); "
                f"deactivated {n} active reminder(s)"
            )
            # 取消是整句消化的语义 — 不让 sub-intent 处理用户残句 ("别提醒了"
            # 不该再被当成"计划查询"询问 AI 日程, 反过来给离题回复).
            ctx.consumed_full_message = True
            return True, ctx.finalize(reply)

        # 设置/新增提醒走原逻辑
        when_text = await _direct_create_reminder(
            user_message=user_message, ctx=ctx,
        )
        reply = await record_confirm_reply(
            summary=user_message[:120],
            when_text=when_text or "随时",
            is_recurring=False,
            personality_brief=_agent_name(ctx.agent),
        )
        if not reply:
            # LLM 生成失败 — 用模板兜底而非 fall-through, 否则用户会看到泛泛回复
            # 而 timetrigger 已经建好导致体感错位 ("AI 在闲聊但 1 分钟后突然提醒").
            reply = (
                f"好嘞, {when_text}叫你, 记好啦~" if when_text
                else "好嘞, 我帮你记上了~"
            )
        return True, ctx.finalize(reply)
    except Exception as e:
        logger.warning(f"Record request short-circuit failed, falling through: {e}")
        return False, None
