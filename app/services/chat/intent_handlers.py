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

    async def finalize(self, reply: str) -> AsyncGenerator[dict, None]:
        self.last_short_circuit_reply = reply
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


# RECORD_REQUEST 专用宽松正则: 允许"X 分钟/小时/天/周"省略"后"字.
# 严格 parser (`time_parser._REL_OFFSET_PAT`) 要求"前|后"是为了防止
# "我等了一分钟" 这种非时间表达被误匹配; 但 RECORD_REQUEST intent 已确认
# 用户在设提醒, 大概率 "X 分钟" = "X 分钟后".
import re as _re
_LOOSE_OFFSET_PAT = _re.compile(
    r"([一二三四五六七八九十百两\d]{1,4})\s*(秒|分钟|小时|天|周)"
)
_CN_NUM = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_cn_or_arabic(s: str) -> int | None:
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    # 简单"十X"/"X十"/"X十Y" — 不超过 99
    if "十" in s:
        parts = s.split("十")
        if len(parts) == 2:
            tens = _CN_NUM.get(parts[0], 1) if parts[0] else 1
            ones = _CN_NUM.get(parts[1], 0) if parts[1] else 0
            return tens * 10 + ones
    return None


def _record_request_loose_offset(message: str, now: datetime) -> datetime | None:
    """RECORD_REQUEST 上下文里宽松提取 "X 分钟/小时/天/周" 默认当作 "+ 方向".
    返回 future datetime 或 None."""
    from datetime import timedelta
    m = _LOOSE_OFFSET_PAT.search(message)
    if not m:
        return None
    amount = _parse_cn_or_arabic(m.group(1))
    if amount is None or amount <= 0:
        return None
    unit = m.group(2)
    if unit == "秒":
        delta = timedelta(seconds=amount)
    elif unit == "分钟":
        delta = timedelta(minutes=amount)
    elif unit == "小时":
        delta = timedelta(hours=amount)
    elif unit == "天":
        delta = timedelta(days=amount)
    elif unit == "周":
        delta = timedelta(weeks=amount)
    else:
        return None
    return now + delta


async def _direct_create_reminder(
    *, user_message: str, ctx: ShortCircuitCtx,
) -> str | None:
    """同步用 time_parser 抽 future event_time → 直接 store_memory + 建 timetrigger.

    返回人话 when_text 给 confirm reply 用; 没抽到时间则返 None (调用方 fallback
    到模糊确认 "我帮你记上了" 不带时间).

    架构选择 (生产 bug 触发的根因修复): 之前依赖后台 `_bg_memory_pipeline`
    抽取 → 但 pre-filter LLM 偶尔把"提醒我X好吗"这种问句判为"不记", 导致
    用户的提醒永远不入库 → trigger 不创建 → 用户没收到提醒. 改为 handler
    内同步用规则引擎 parse, 100% 可靠. 后台 pipeline 仍跑作冗余 (dedup
    会防 memory 重复, pipeline 侧的 `_create_reminder_timetrigger` 会被
    idempotency 闸跳过).
    """
    from datetime import datetime
    from app.services.memory.recording.pipeline import _create_reminder_timetrigger
    from app.services.memory.storage.persistence import store_memory
    from app.services.schedule_domain.time_parser import parse_with_statement_time
    from app.services.schedule_domain.time_service import _now_corrected
    from app.services.workspace.workspaces import resolve_workspace_id

    if not ctx.user_id:
        return None

    # 关键: parse 必须用"用户消息接收时刻"而不是 handler 跑到这一行的时刻.
    # 之前用 _now_corrected() → 处理链路上前面的 LLM 调用累计 ~25s, 导致
    # "两分钟后" 实际算成 "处理完 + 2分钟", 提醒比用户期望晚 25s.
    # ctx.reply_context["received_at"] 是 ws 收到用户消息的时间戳 (ISO 字符串).
    received_at: datetime | None = None
    if ctx.reply_context:
        raw = ctx.reply_context.get("received_at")
        if raw:
            try:
                received_at = datetime.fromisoformat(str(raw))
            except (TypeError, ValueError):
                received_at = None
    parse_now = received_at or _now_corrected()

    parsed = parse_with_statement_time(user_message, now=parse_now)
    future_events = [e for e in parsed.event_times if e.is_future]

    # Fallback: 用户口语经常省"后"字 ("一分钟提醒我喝水好吗"). 全局 parser
    # 严格要"X分钟后"格式以防误匹配, 但 RECORD_REQUEST intent 已确认是用户
    # 设提醒, 这里宽松匹配 "X(分钟|小时|天|周)" 默认当 "+方向", 优先级低于
    # 严格匹配 (没找到 future event 才走 fallback).
    if not future_events:
        offset = _record_request_loose_offset(user_message, parse_now)
        if offset is not None:
            occur_time = offset
        else:
            logger.info(
                f"[RECORD-REQ] no future event time parsed from {user_message[:60]!r}; "
                "falling back to fuzzy confirmation (background pipeline still runs)"
            )
            return None
    else:
        # 取置信度最高的 future event_time 作为提醒时刻
        chosen = max(future_events, key=lambda e: e.confidence)
        occur_time = chosen.start
    # Summary 用第一人称 "我..." 而非 spec §2.1 extraction 的 "用户..." 模板:
    # RECORD_REQUEST 是用户主动让 AI 帮他记一件事, 记忆是用户的指令记录, 第一
    # 人称体感更自然 (用户在自己的记忆面板看到 "我让 AI 提醒..." 比 "用户请求..."
    # 顺); spec §2.1 那个 "用户..." 前缀约束是给 LLM extraction 的输出格式指令,
    # 不是 schema 强制. 前端按 source 已能区分谁的记忆, 不需要前缀帮 disambig.
    summary_for_memory = f"我让 AI 提醒: {user_message[:120]}"

    workspace_id = await resolve_workspace_id(user_id=ctx.user_id)
    # statement_time 也用 received_at, 跟 occur_time 的时间基准一致.
    statement_time = parse_now
    try:
        memory_id = await store_memory(
            user_id=ctx.user_id,
            content=summary_for_memory,
            summary=summary_for_memory,
            level=3,
            importance=0.45,  # in [0.4, 0.49] 区间, 落 L3 (跟 pipeline clamp 对齐)
            memory_type="life",
            main_category="生活",
            sub_category="提醒",
            occur_time=occur_time,
            statement_time=statement_time,
            workspace_id=workspace_id,
            source="user",
            recurrence="once",
        )
    except Exception as e:
        logger.warning(f"[RECORD-REQ] store_memory failed: {e}")
        return None

    # store_memory 在 dedup 命中 (相似度 > 0.9) 时返回 None. 这种情况下
    # 必须**复用** existing memory id 继续建 trigger — 之前的 bug 是直接
    # skip, 但 existing memory 的旧 trigger 早已 fired 完 (isActive=False),
    # 用户这次重新请求的 "1分钟后提醒" 不会触发, 体感像没设上.
    if not memory_id:
        from app.services.memory.storage.embedding import generate_embedding
        from app.services.memory.storage.persistence import find_duplicate_id
        from app.services.memory.storage import repo as memory_repo
        try:
            embedding = await generate_embedding(summary_for_memory)
            memory_id = await find_duplicate_id(
                ctx.user_id, summary_for_memory, embedding,
                workspace_id=workspace_id,
            )
            if memory_id:
                # 用户重新设置了提醒 → 把 occur_time 更新到这次的新值,
                # 让新 trigger 按新时刻 fire. 旧 trigger 已 inactive 不影响.
                await memory_repo.update(
                    memory_id, source="user",
                    occurTime=occur_time, statementTime=statement_time,
                )
                logger.info(
                    f"[RECORD-REQ] reusing deduped memory={memory_id[:8]}, "
                    f"updated occurTime={occur_time}"
                )
        except Exception as e:
            logger.warning(f"[RECORD-REQ] dedup fallback failed: {e}")
            return None

    if not memory_id:
        logger.warning(
            "[RECORD-REQ] both store_memory and dedup-id lookup failed; "
            "no reminder scheduled"
        )
        return None

    await _create_reminder_timetrigger(
        user_id=ctx.user_id,
        memory_id=memory_id,
        summary=summary_for_memory,
        occur_time=occur_time,
        recurrence="once",
        side="user",
    )
    logger.info(
        f"[RECORD-REQ] reminder scheduled memory={memory_id[:8]} at={occur_time}"
    )
    return _format_when_text(occur_time)


async def handle_record_request(
    user_message: str,
    ctx: ShortCircuitCtx,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    """RECORD_REQUEST 短路: 用户请求 AI 记一件事 / 设提醒.

    流程:
    1. 规则引擎 parse user_message 找 future event_time
    2. 找到 → 同步直接 store memory + 建 timetrigger (跳过后台 pipeline 不可靠路径)
    3. 生成 confirmation reply (人话 when_text 注入)
    4. 没找到时间 → 仍生成模糊确认 "我帮你记上了", 后台 pipeline 兜底 (best-effort)

    背景: 生产环境观察到背景 pipeline 的 pre-filter LLM 把"提醒我X好吗?"判为
    "不记" → 用户的提醒被丢. 改为 handler 直接创建保证可靠性. 后台 pipeline
    仍跑作冗余 (memory dedup + 已存在 trigger 的 idempotency 防重).
    """
    try:
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
