"""Spec §3.4 各意图的短路处理器。

从 `orchestrator.stream_chat_response` 中抽出 7 个意图分支：每个 handler
只关心自己的输入/参考信息 + 生成 reply，尾部统一交给 `finalize_short_circuit`。

handler 作为 async generator 产出 SSE 事件，orchestrator 只需 `async for ...: yield`。
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chat.tracing import LangSmithTracer

from app.db import db
from app.observability.events import EVT_INTENT_SHORT_CIRCUIT
from app.services.chat.intent_replies import (
    apology_reply,
    crisis_followup_reply,
    crisis_reply,
    current_state_reply,
    deletion_confirm_reply,
    end_reply,
    record_ask_time,
    record_confirm_reply,
    schedule_query_reply,
)
from app.services.chat.intent_dispatcher import (
    infer_schedule_query_type,
    is_explicit_current_state_query,
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
    get_cached_schedule,
    handle_schedule_adjustment,
    resolve_schedule_query_scope,
    update_schedule_slot,
)
from app.services.schedule_domain.time_service import get_current_time, resolve_implicit_time
from app.services.prompting.store import get_prompt_text_or_default
from app.services.prompting.utils import EMPTY_RECENT_CONTEXT
from app.services.rules.chat_keywords import (
    CANCEL_NEG_TOKENS,
    HIGH_CONFIDENCE_CANCEL_KEYWORDS,
    LOW_CONFIDENCE_CANCEL_KEYWORDS,
    MEMORY_FACT_RECALL_CUES,
    RECURRENCE_KEYWORDS,
    RECORD_MEMORY_CUES,
    REMINDER_ACTION_CUES,
    REMINDER_CONTENT_CUES,
    SELF_NOTE_CUES,
    TIME_OR_EVENT_CUES,
    UNDO_CANCEL_KEYWORDS,
)

logger = logging.getLogger(__name__)

_SHORT_CIRCUIT_REPEAT_COOLDOWN = timedelta(seconds=90)
_SHORT_CIRCUIT_REPEAT_CONTEXT_MAX_AGE = timedelta(minutes=15)
_SHORT_CIRCUIT_REPEAT_RECENT_TAKE = 12
_SCHEDULE_REPEAT_CONTEXT_HINT = (
    "【重复追问处理】用户刚才已经问过相同时间段的忙闲或安排。"
    "这次只自然承接当前这句，简短回答即可；不要复述完整日程，"
    "不要说“安排没变”“具体哪一段”“拎出来”等工具化话术。"
)
_CURRENT_STATE_REPEAT_CONTEXT_HINT = (
    "【重复追问处理】用户刚才已经问过你当前状态。"
    "这次只自然承接当前这句，简短回答即可；不要重复完整活动细节，"
    "不要使用固定模板。"
)
_CURRENT_STATE_METADATA_KEY = "current_state_context"
_SHORT_CIRCUIT_REPEAT_METADATA_KEY = "short_circuit_repeat"


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
    # 最近几轮对话格式化文本 (format_recent_context 的输出). 所有走 *_reply prompt
    # 的 handler 都需要它注入 {context}: 缺了就只能看到当前消息 + AI 当前作息,
    # LLM 会从作息里"借"内容编出跟当下活动巧合相关的答案 (生产 bug: 用户问
    # "你看到什么段子" → AI 编了一个跟自己当前划船活动巧合的段子). 所有 handler
    # 共享 orchestrator 已计算好的同一份 recent_context, 不再各自取数.
    recent_context: str = EMPTY_RECENT_CONTEXT
    # 短路 handler 经 ctx.finalize(reply) 把回复文本回写到这里, 让 orchestrator
    # finally 兜底 fire post_process 时拿到正确的 full_response (否则 short-circuit
    # 路径直接 return, post_process 永不跑, 记忆/用户情绪/trait 全丢失).
    last_short_circuit_reply: str | None = None
    # spec §3.3 step 2 假设多意图能字面拆分 (示例"我好难过，不想聊了" → 2 个独立
    # 子句). 但生产场景的口语融合句 (e.g. "算了别提醒了, 我吃过了" 整体语义 =
    # 取消) 子片段单独看意图相反, 强行 sub-intent 处理会反向回复. handler
    # 在主意图消化整句时 (典型: RECORD_REQUEST 取消分支) 设此 flag = True,
    # finalize 跳过 sub-intent 递归. 详见 CLAUDE.md §6 偏离表.
    consumed_full_message: bool = False
    last_short_circuit_kind: str | None = None
    response_diagnostics: dict[str, Any] | None = None
    covered_until_user_ts: datetime | None = None
    achievement_turn_final: bool = True

    async def finalize(
        self,
        reply: str,
        *,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """`kind` 是该 handler 的 intent 名 (e.g. "apology_promise", "deletion_delete"),
        作为 EVT_INTENT_SHORT_CIRCUIT 的查询维度. 必填以防止 handler 漏标."""
        self.last_short_circuit_reply = reply
        self.last_short_circuit_kind = kind
        logger.info(
            f"[INTENT-SC] kind={kind} reply_len={len(reply)}",
            extra={
                "event": EVT_INTENT_SHORT_CIRCUIT,
                "intent_kind": kind,
                "reply_text_len": len(reply),
                "consumed_full_message": self.consumed_full_message,
                "sub_intent_mode": self.sub_intent_mode,
            },
        )
        # consumed_full_message=True 时清空 sub fragments, 让 finalize_short_circuit
        # 跳过 process_sub_intents 递归. 比加新参数到 finalize 签名干净.
        sub_fragments = (
            {} if self.consumed_full_message else self.pending_sub_fragments
        )
        extra_metadata: dict[str, Any] | None = dict(metadata or {})
        if self.covered_until_user_ts is not None:
            extra_metadata.setdefault(
                "covered_until_user_ts",
                self.covered_until_user_ts.isoformat(),
            )
        if self.response_diagnostics is not None:
            self.response_diagnostics.update({
                "reply_path": "short_circuit",
                "short_circuit_kind": kind,
                "main_prompt_built": False,
            })
            extra_metadata["response_diagnostics"] = self.response_diagnostics
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
            extra_metadata=extra_metadata,
            achievement_turn_final=self.achievement_turn_final,
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
        context=ctx.recent_context,
        personality_brief=_agent_name(ctx.agent),
    )
    if not farewell:
        # 结构性兜底指令: 停用时退回代码默认 (终结链路必须产出道别, 不能断).
        fallback_instruction = await get_prompt_text_or_default(
            "intent.conversation_end_fallback_instruction"
        )
        farewell = await fallback_fn(
            ctx.agent, user_message, str(fallback_instruction),
        )
    async for evt in ctx.finalize(farewell, kind="conversation_end"):
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
            context=ctx.recent_context,
            personality_brief=_agent_name(ctx.agent),
            new_patience=new_patience,
        ) or "好啦，我不生气了~"
        return True, ctx.finalize(reply, kind="apology_promise")
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
        deletion_result = await detect_deletion_intent(
            user_message,
            recent_context=ctx.recent_context,
        )
        description = (deletion_result or {}).get("target_description")
        if not description:
            return False, None

        candidates = await find_matching_memories(ctx.user_id, description)
        agent_name = ctx.agent.name if ctx.agent else "伙伴"
        if not candidates:
            return True, ctx.finalize("嗯...我好像没有关于这个的记忆呢。", kind="deletion_no_match")

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
                    context=ctx.recent_context,
                    personality_brief=agent_name,
                    candidate_memories=candidate_preview,
                )
                or await generate_deletion_confirmation_prompt(agent_name, candidates)
            )
        # deletion_delete or deletion_reschedule
        return True, ctx.finalize(reply, kind=f"deletion_{intent}")
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
        return True, ctx.finalize(response, kind="schedule_adjust")
    except Exception as e:
        logger.warning(f"Schedule adjustment failed, falling through: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════
# §3.4.1 计划查询
# ═══════════════════════════════════════════════════════════════════


def _normalize_repeat_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    data = getattr(value, "data", None)
    return data if isinstance(data, dict) else {}


def _as_aware_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _hashed_repeat_key(*parts: Any) -> str:
    normalized = "\n".join(
        _normalize_repeat_key(part) for part in parts if part is not None
    )
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _repeat_metadata(kind: str, repeat_key: str, **extra: Any) -> dict[str, Any] | None:
    if not repeat_key:
        return None
    context = {
        "kind": kind,
        "key": repeat_key,
        **{k: v for k, v in extra.items() if v is not None},
    }
    return {_SHORT_CIRCUIT_REPEAT_METADATA_KEY: context}


def _with_repeat_context_hint(context: str | None, hint: str) -> str:
    base = (context or "").strip()
    if not base or base == EMPTY_RECENT_CONTEXT:
        return hint
    return f"{base}\n{hint}"


def _recent_repeat_context_matches(metadata: dict[str, Any], kind: str, repeat_key: str) -> bool:
    context = metadata.get(_SHORT_CIRCUIT_REPEAT_METADATA_KEY)
    if isinstance(context, dict):
        return context.get("kind") == kind and context.get("key") == repeat_key

    # Backward-compatible read for current-state metadata written before the
    # generic repeat context existed in this branch.
    if kind == "current_state":
        legacy = metadata.get(_CURRENT_STATE_METADATA_KEY)
        if isinstance(legacy, dict):
            return legacy.get("activity_key") == repeat_key
    return False


def _is_same_short_circuit_topic(kind: str, text: str) -> bool:
    if kind == "current_state":
        return is_explicit_current_state_query(text)
    if kind == "schedule_query":
        return infer_schedule_query_type(text, require_query_cue=False) is not None
    return False


def _has_short_circuit_topic_break(kind: str, user_messages: list[str]) -> bool:
    return any(
        content and not _is_same_short_circuit_topic(kind, content)
        for content in user_messages
    )


async def _has_recent_short_circuit_repeat(
    conversation_id: str,
    *,
    kind: str,
    repeat_key: str,
    current_message: str | None = None,
) -> bool:
    if not conversation_id or not kind or not repeat_key:
        return False
    try:
        rows = await db.message.find_many(
            where={"conversationId": conversation_id},
            order={"createdAt": "desc"},
            take=_SHORT_CIRCUIT_REPEAT_RECENT_TAKE,
        )
    except Exception as e:
        logger.warning(f"Short-circuit repeat lookup failed: {e}")
        return False

    now = datetime.now(timezone.utc)
    newer_user_messages = (
        [_normalize_repeat_key(current_message)] if current_message else []
    )
    for row in rows:
        content = _normalize_repeat_key(getattr(row, "content", ""))
        role = getattr(row, "role", "")
        if role == "user":
            if content and content not in newer_user_messages:
                newer_user_messages.append(content)
            continue
        if role != "assistant":
            continue
        created_at = _as_aware_utc(getattr(row, "createdAt", None))
        metadata = _metadata_dict(getattr(row, "metadata", None))
        if not _recent_repeat_context_matches(metadata, kind, repeat_key):
            continue
        if _has_short_circuit_topic_break(kind, newer_user_messages):
            return False
        age = now - created_at if created_at else None
        if age is None or age <= _SHORT_CIRCUIT_REPEAT_COOLDOWN:
            return True
        if age <= _SHORT_CIRCUIT_REPEAT_CONTEXT_MAX_AGE:
            return True
    return False

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
    scope = resolve_schedule_query_scope(
        user_message,
        now=get_current_time().now,
        require_query_cue=False,
    )
    resolved_query_type = scope.query_type if scope else query_type
    target_schedule = schedule
    target_status = ai_status if resolved_query_type == "current" else None
    date_label = (
        scope.date_label if scope
        else ("今天" if resolved_query_type in {"current", "routine"} else "相关日期")
    )
    if resolved_query_type == "date":
        target_schedule = []
        if ctx.agent_id and scope and scope.target_date is not None:
            cached = await get_cached_schedule(ctx.agent_id, scope.target_date)
            if cached:
                target_schedule = cached
    if target_schedule:
        schedule_context = format_full_schedule_for_query(
            target_schedule, resolved_query_type, target_status, date_label=date_label,
        )
    else:
        # 结构性兜底上下文: 停用时退回代码默认 (计划查询必须有上下文说明).
        missing_tpl = await get_prompt_text_or_default("intent.schedule_missing_context")
        schedule_context = missing_tpl.format(date_label=date_label)
    repeat_key = _hashed_repeat_key(resolved_query_type, date_label, schedule_context)
    repeat_metadata = _repeat_metadata(
        "schedule_query",
        repeat_key,
        query_type=resolved_query_type,
        date_label=date_label,
    )
    is_repeat_query = await _has_recent_short_circuit_repeat(
        ctx.conversation_id,
        kind="schedule_query",
        repeat_key=repeat_key,
        current_message=user_message,
    )
    try:
        reply_context = (
            _with_repeat_context_hint(ctx.recent_context, _SCHEDULE_REPEAT_CONTEXT_HINT)
            if is_repeat_query else ctx.recent_context
        )
        response = await schedule_query_reply(
            message=user_message,
            context=reply_context,
            user_emotion=user_emotion,
            personality_brief=_agent_name(ctx.agent),
            user_portrait=str(portrait) if portrait else "",
            current_activity=(
                format_schedule_context(ai_status)
                if resolved_query_type == "current" and ai_status else ""
            ),
            ai_schedule=schedule_context or "(未知)",
        )
        if not response:
            return False, None, schedule_context
        return (
            True,
            ctx.finalize(response, kind="schedule_query", metadata=repeat_metadata),
            schedule_context,
        )
    except Exception as e:
        logger.warning(f"Schedule query short-circuit failed, falling through: {e}")
        return False, None, schedule_context


# ═══════════════════════════════════════════════════════════════════
# §3.4.3 询问当前状态
# ═══════════════════════════════════════════════════════════════════


def _normalize_current_activity(activity: Any) -> str:
    """Stable key for "same current state" checks.

    This intentionally keys on resolved activity, not the user's text, so
    variants like "在忙吗" / "干嘛呢" / "你现在做什么" share the same cooldown.
    """
    return re.sub(r"\s+", " ", str(activity or "")).strip()


def _current_state_metadata(activity_key: str) -> dict[str, Any]:
    metadata = _repeat_metadata("current_state", activity_key) or {}
    metadata.update({
        _CURRENT_STATE_METADATA_KEY: {
            "intent": "current_state",
            "activity_key": activity_key,
        }
    })
    return metadata

async def handle_current_state(
    user_message: str,
    ctx: ShortCircuitCtx,
    *,
    ai_status: dict | None,
    schedule_context: str | None,
    portrait: Any,
    user_emotion: dict | None,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    if not is_explicit_current_state_query(user_message):
        return False, None

    # spec §3.2 隐性时间解析: 走时间中枢 helper, 复用 caller 已加载的 ai_status
    _, current_activity = await resolve_implicit_time(ctx.agent_id or "", ai_status)
    activity_key = _normalize_current_activity(current_activity)
    metadata = _current_state_metadata(activity_key) if activity_key else None
    is_repeat_query = await _has_recent_short_circuit_repeat(
        ctx.conversation_id,
        kind="current_state",
        repeat_key=activity_key,
        current_message=user_message,
    )
    try:
        reply_context = (
            _with_repeat_context_hint(ctx.recent_context, _CURRENT_STATE_REPEAT_CONTEXT_HINT)
            if is_repeat_query else ctx.recent_context
        )
        response = await current_state_reply(
            message=user_message,
            context=reply_context,
            user_emotion=user_emotion,
            personality_brief=_agent_name(ctx.agent),
            user_portrait=str(portrait) if portrait else "(未知)",
            current_activity=current_activity,
            ai_schedule="",
        )
        if not response:
            return False, None
        return True, ctx.finalize(response, kind="current_state", metadata=metadata)
    except Exception as e:
        logger.warning(f"Current state short-circuit failed, falling through: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════
# 工程扩展 (P0 危机安全网): 危机求助 (CRISIS)
# ═══════════════════════════════════════════════════════════════════


# 静态兜底: handle_crisis 调 LLM 失败 (dashscope/Ollama 全挂) 时, 用这条
# 写好的回复保住用户安全 — 总比"crisis 漏到主路径 → AI 编兔子假耳朵故事"
# 好得多. 措辞遵循三步原则 (接住 → 想了解), 没有空话/客套.
_CRISIS_STATIC_FALLBACK = (
    "听到你这么说我心里很难受。我在这儿陪着你。"
    "可以告诉我现在是什么让你这么难吗？我想多听一听。"
)


def _format_user_memory_for_crisis(
    classified_memories: list,
    *,
    include_factual: bool = False,
) -> str:
    """从已检索 classified_memories 中筛出"用户侧 + 跟情绪/求助/边界相关"的条目,
    给 crisis prompt 的 {user_memory} 占位符. 主线: 让 LLM 知道这是不是 Ta 第一次
    这样表达 — 已知历史 (e.g. L1 `用户表达过强烈负面情绪`) 帮助 LLM 用"我记得你
    上次也..."的连贯语气, 而不是干巴巴的通用模板.

    crisis follow-up 里用户可能转而问普通记忆事实, 或主动转移到别的话题
    来稳定情绪; include_factual=True 时保留当前话题相关记忆, 防止 aftercare
    只看到安全记忆后断掉正常朋友式承接.

    筛选不调 LLM (热路径首位), 只用关键字粗筛. 没命中相关条目时返"(无)" —
    就当 Ta 第一次说, prompt 自己有兜底措辞.
    """
    def _source(memory: Any) -> str:
        if isinstance(memory, dict):
            return str(memory.get("source") or "user")
        return str(getattr(memory, "source", "user") or "user")

    def _text(memory: Any) -> str:
        if isinstance(memory, dict):
            return str(memory.get("text") or memory.get("summary") or memory.get("content") or "")
        return str(getattr(memory, "text", "") or "")

    def _rank_reasons(memory: Any) -> list[str]:
        if isinstance(memory, dict):
            return [str(reason) for reason in (memory.get("rank_reasons") or [])]
        return [str(reason) for reason in (getattr(memory, "rank_reasons", None) or [])]

    user_memories = [m for m in (classified_memories or []) if _source(m) != "ai" and _text(m)]
    user_lines = [_text(m) for m in user_memories]
    if not user_lines:
        return "(无)"
    # 跟情绪/求助/求救相关的关键词. 跟 CRISIS_KEYWORDS 重叠是 OK 的
    # (那些是触发判定, 这些是召回筛选), 同时加宽到泛情绪词避免漏召回.
    relevance_kw = (
        "情绪", "难过", "委屈", "崩溃", "压力", "焦虑", "抑郁",
        "哭", "孤独", "撑不住", "求救", "求助",
        "自伤", "自残", "轻生", "想死", "活不下去", "活着没",
        "跳楼", "跳河", "自杀",
    )

    if include_factual:
        seen: set[str] = set()

        def _take(items: list[str], limit: int) -> list[str]:
            result: list[str] = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                result.append(item)
                if len(result) >= limit:
                    break
            return result

        def _has_reason(memory: Any, prefix: str) -> bool:
            return any(reason.startswith(prefix) for reason in _rank_reasons(memory))

        named_relation = [
            _text(m) for m in user_memories
            if _has_reason(m, "保护槽:关系命名")
        ]
        literal = [
            _text(m) for m in user_memories
            if _has_reason(m, "保护槽:字面命中")
        ]
        topical = [
            _text(m) for m in user_memories
            if _has_reason(m, "保护槽:当前话题")
        ]
        safety = [
            _text(m) for m in user_memories
            if _has_reason(m, "保护槽:危机安全背景")
            or any(kw in _text(m) for kw in relevance_kw)
        ]
        other = [t for t in user_lines]

        sections: list[str] = []

        def _append_section(label: str, items: list[str]) -> None:
            if items:
                sections.append(label)
                sections.extend(f"- {item}" for item in items)

        _append_section("【回答当前关系 / 名字问题优先参考】", _take(named_relation, 2))
        _append_section("【回答当前问题可参考】", _take(literal, 2))
        _append_section("【当前话题相关记忆】", _take(topical, 4))
        _append_section("【安全 / 情绪背景】", _take(safety, 3))
        _append_section("【其他已选记忆】", _take(other, 2))
        return "\n".join(sections) if sections else "(无)"

    hits = [t for t in user_lines if any(kw in t for kw in relevance_kw)]
    if not hits:
        return "(无)"
    max_items = 7 if include_factual else 5
    return "\n".join(f"- {t}" for t in hits[:max_items])  # 避免 token 爆


def _crisis_followup_safety_check_instruction(mode: str) -> str:
    """Instruction string rendered into the crisis follow-up prompt."""
    if mode == "soft":
        return (
            "本轮需要做一次轻量安全复核。先自然回应用户当前话题, "
            "再用一句不审问的方式确认 Ta 现在是否安全、是否还有伤害自己的冲动。"
        )
    if mode == "annoyed":
        return (
            "本轮需要做一次安全复核, 但用户可能已经嫌问题烦。"
            "先承认这些确认会烦, 简短说明只是因为刚才的风险需要确认安全, "
            "然后只请 Ta 用一句话确认现在是否安全。"
        )
    return "本轮不主动复核安全状态; 除非用户主动提到风险, 不要追问安全。"


def _extract_memory_preference_topics(message: str) -> list[str]:
    """Extract concrete preference topics from recall-style questions.

    Crisis follow-up can keep guarding safety while the user moves to an
    ordinary memory question. For preference recall, absence is meaningful when
    the user asks "我喜欢的 X"; if the retrieved user-memory block does not
    contain X, the reply should say that directly instead of letting the LLM
    invent a remembered fact.
    """
    msg = (message or "").strip()
    if not msg or not any(cue in msg for cue in MEMORY_FACT_RECALL_CUES):
        return []

    match = re.search(r"喜欢的\s*([^吗嘛么呀啊呢了，。！？?\s、]{1,20})", msg)
    if match:
        topic = match.group(1).strip()
        if topic:
            return [topic]
    return []


def _crisis_followup_memory_absence_reply(
    message: str,
    user_memory: str,
    safety_check_mode: str,
) -> str | None:
    topics = _extract_memory_preference_topics(message)
    if not topics:
        return None

    memory_text = (user_memory or "").strip()
    primary_topic = topics[0]
    if memory_text and memory_text != "(无)" and primary_topic in memory_text:
        return None

    if "喜欢" in message and not primary_topic.startswith("喜欢"):
        fact_label = f"你喜欢的{primary_topic}"
    else:
        fact_label = primary_topic

    reply = f"我这里没有看到你跟我说过{fact_label}，不能乱猜。你愿意的话再告诉我一次。"
    if safety_check_mode == "soft":
        reply += " 也跟我确认一下，你现在是安全的吗？"
    elif safety_check_mode == "annoyed":
        reply += " 我知道一直确认会烦，但刚才那种风险我还是得问一句：你现在安全吗？"
    return reply


async def handle_crisis(
    user_message: str,
    ctx: ShortCircuitCtx,
    *,
    classified_memories: list,
    portrait: Any,
) -> AsyncGenerator[dict, None]:
    """P0 危机安全网 short-circuit handler.

    设计:
    - 用专属 intent.crisis_reply prompt 单独调 LLM, **完全切掉**主 system_prompt
      14 段干扰 (delay/ai_state/topic/long history). 实测 trace 已证: 主路径
      crisis_active flag 注入 hint section 不够, 其他段落 (delay_context /
      ai_state_constraint / 历史 AI 旁白) 仍把 LLM 拉去"虽然我在忙乌龟但..."
      句式. handler 隔离才是根治.
    - {user_memory} 占位符传入用户跟情绪/求助相关的已知 L1/L2 — 让 LLM 知道
      Ta 历史背景, 输出"这不是 Ta 第一次"或"我记得你上次..."的连贯回复.
    - LLM 失败 (两级降级全挂) 走 _CRISIS_STATIC_FALLBACK — 比静默漏到主路径
      让 AI 编故事好得多.
    - emoji/sticker 跳过 — handler 直接通过 ctx.finalize 走 finalize_short_circuit
      路径, 不走 emit_replies decoration. 这是天然就跳的, 不需要额外 flag.
    - reply_count = 1 — finalize_short_circuit 默认单条, 跟其他 short-circuit 一致.
    """
    user_memory_block = _format_user_memory_for_crisis(classified_memories)
    try:
        response = await crisis_reply(
            message=user_message,
            context=ctx.recent_context,
            personality_brief=_agent_name(ctx.agent),
            user_portrait=str(portrait) if portrait else "(未知)",
            user_memory=user_memory_block,
        )
        if not response:
            response = _CRISIS_STATIC_FALLBACK
            logger.warning("[CRISIS] LLM returned empty, using static fallback")
    except Exception as e:
        logger.warning(f"Crisis handler LLM failed, using static fallback: {e}")
        response = _CRISIS_STATIC_FALLBACK

    # crisis 整句已被本 handler 消化, 强制 sub fragments 跳过 (防 multi-intent
    # 多个 sub 把"危机 + 别的意图"拆出后 sub 再跑离题回复).
    ctx.consumed_full_message = True
    async for evt in ctx.finalize(response, kind="crisis"):
        yield evt


async def handle_crisis_followup(
    user_message: str,
    ctx: ShortCircuitCtx,
    *,
    classified_memories: list,
    portrait: Any,
    safety_check_mode: str = "none",
) -> AsyncGenerator[dict, None]:
    """Crisis aftercare for follow-up messages that no longer repeat keywords.

    Example: user says "我想死"; AI responds safely; user then asks "你开心吗".
    That question is literally about the AI's current feeling, but the active
    conversational state is still unresolved crisis. This handler prevents the
    current-state branch from describing AI activities and drifting away.
    """
    user_memory_block = _format_user_memory_for_crisis(
        classified_memories,
        include_factual=True,
    )
    safety_check_instruction = _crisis_followup_safety_check_instruction(
        safety_check_mode,
    )
    memory_absence_reply = _crisis_followup_memory_absence_reply(
        user_message,
        user_memory_block,
        safety_check_mode,
    )
    if memory_absence_reply:
        ctx.consumed_full_message = True
        async for evt in ctx.finalize(memory_absence_reply, kind="crisis_followup"):
            yield evt
        return

    try:
        response = await crisis_followup_reply(
            message=user_message,
            context=ctx.recent_context,
            personality_brief=_agent_name(ctx.agent),
            user_portrait=str(portrait) if portrait else "(未知)",
            user_memory=user_memory_block,
            safety_check_instruction=safety_check_instruction,
        )
        if not response:
            response = _CRISIS_STATIC_FALLBACK
            logger.warning("[CRISIS-FOLLOWUP] LLM returned empty, using static fallback")
    except Exception as e:
        logger.warning(f"Crisis followup handler LLM failed, using static fallback: {e}")
        response = _CRISIS_STATIC_FALLBACK

    ctx.consumed_full_message = True
    async for evt in ctx.finalize(response, kind="crisis_followup"):
        yield evt


# ═══════════════════════════════════════════════════════════════════
# 工程扩展 §3.4 + Part 5 §4.2: 记录请求 (RECORD_REQUEST)
# ═══════════════════════════════════════════════════════════════════


def _format_when_text(occur_dt) -> str:
    """渲染时间给 confirm prompt 用. 真实现在 reminder/scheduling.format_when_text
    (智能相对/绝对切换, 不再死板"05月02日 22:50叫你")."""
    from app.services.reminder.scheduling import format_when_text
    return format_when_text(occur_dt)


def classify_record_request_action(message: str) -> str:
    """把 RECORD_REQUEST 再分成提醒管理 / 普通记忆 / 用户自记笔记。

    intent 识别只负责把“记 / 提醒 / 改期 / 取消”类消息粗路由进来。这里再用
    保守的关键词和结构规则决定是否真的进入 reminder 写库，避免“记一下我的
    偏好”被错误反问时间。

    返回:
    - "reminder": 创建/改期/取消/周期提醒等事项管理
    - "memory_note": 用户明确让 AI 记住一个事实/偏好/原则
    - "self_note": 用户在整理自己要写下来的句子，不是让 AI 设提醒
    - "reminder_content": 修改已有提醒的文案/内容
    - "none": 不应由 RECORD_REQUEST 短路消费
    """
    msg = message.strip()
    if not msg:
        return "none"

    if (
        any(cue in msg for cue in ("你记得", "还记得", "记不记得"))
        and any(q in msg for q in ("吗", "么", "？", "?"))
    ):
        return "none"

    if any(cue in msg for cue in REMINDER_CONTENT_CUES):
        return "reminder_content"
    if any(cue in msg for cue in REMINDER_ACTION_CUES):
        return "reminder"
    if any(cue in msg for cue in SELF_NOTE_CUES):
        return "self_note"

    has_record_cue = any(cue in msg for cue in RECORD_MEMORY_CUES)
    if has_record_cue:
        # “记得帮我盯着报告” 已被 REMINDER_ACTION_CUES 捕获; 这里剩下的是
        # “记住我…” “记一下：…” 这类长期记忆请求。
        if (
            "我" in msg
            or "：" in msg
            or ":" in msg
            or msg.startswith(("记一下", "记住", "帮我记", "替我记"))
        ):
            return "memory_note"
        return "none"

    # 无显式“提醒我”但有明确未来/周期事项的历史行为仍保留为 reminder。
    if any(cue in msg for cue in TIME_OR_EVENT_CUES):
        return "reminder"
    return "none"


def extract_record_memory_text(message: str) -> str:
    """从“记一下：X / 你可以记住X”里抽要记住的主体文本。"""
    msg = message.strip()
    for sep in ("：", ":"):
        if sep in msg:
            return msg.split(sep, 1)[1].strip() or msg
    prefixes = (
        "你可以记住", "可以记住", "帮我记一下", "替我记一下", "帮我记",
        "替我记", "记一下", "记住", "记下来", "记着",
    )
    for prefix in prefixes:
        if msg.startswith(prefix):
            return msg[len(prefix):].strip(" ，,。") or msg
    return msg


def _detect_recurrence(message: str) -> str:
    """从用户消息识别 recurrence (once/daily/weekly/monthly/yearly).
    `RECURRENCE_KEYWORDS` 命中即返对应周期, 否则 "once".
    """
    msg = message.strip()
    for kw, rec in RECURRENCE_KEYWORDS:
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
) -> tuple[str, str | None]:
    """同步用 time_parser 抽 future event_time → 直接 store_memory + 建 timetrigger.

    返回 (status, when_text):
    - ("scheduled", "8月1日 10:00") — 落库成功, when_text 给 confirm reply
    - ("asked", None) — 时间没说清, 已 save_pending_action(action="set_reminder"),
       调用方应 record_ask_time 反问. 用户下条消息走 preflight.resolve_pending_set_reminder
    - ("failed", None) — workspace/agent 缺失等错误, 调用方走兜底回复

    支持:
    - 单 occur_time ("一分钟后提醒X")
    - 多 occur_time ("8 点吃药, 9 点开会") — 各建 1 条 trigger, when_text 合并为
      "08:00 / 09:00" 格式
    - 周期性 ("每天提醒X" / "每月 1 号Y") — 关键词识别 recurrence, 通过 trigger
      handler 周期续期
    - 缺"后"字口语 ("一分钟提醒X") — parse_loose_offset fallback
    - 完全模糊时间 ("待会"/"过会"/"下周") — pending + 反问 (Round-3 工程扩展)

    架构: 跨模块依赖 (memory + workspace + reminder/scheduling + deletion) 通过
    本文件 inline import 隔离.
    """
    import asyncio
    from datetime import datetime
    from app.services.memory.interaction.deletion import save_pending_action
    from app.services.reminder.scheduling import upsert_reminder_trigger
    from app.services.schedule_domain.time_parser import (
        parse_loose_offset, parse_with_statement_time,
    )
    from app.services.schedule_domain.time_service import _now_corrected
    from app.services.workspace.workspaces import get_active_workspace

    if not ctx.user_id:
        return ("failed", None)

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
        # Round-3 工程扩展: 时间没说清 → save pending + 反问. 之前 fallback 假装
        # 记下 ("好嘞, 待会叫你") 但实际没建 trigger, 是 silent correctness bug.
        # 现在反问让用户给具体时间, 第二轮走 preflight.resolve_pending_set_reminder.
        try:
            await save_pending_action(
                ctx.conversation_id,
                action="set_reminder",
                summary=user_message[:200],
            )
            logger.info(
                f"[RECORD-REQ] no time parsed from {user_message[:60]!r}; "
                f"saved pending set_reminder, will ask user for specific time"
            )
            return ("asked", None)
        except Exception as e:
            logger.warning(f"[RECORD-REQ] save_pending_action failed: {e}; falling back")
            return ("failed", None)

    # workspace 仅查一次; .id 直接拿避免 resolve_workspace_id 二次 find_first.
    workspace = await get_active_workspace(
        user_id=ctx.user_id, agent_id=ctx.agent_id,
    )
    if not workspace:
        logger.warning(
            f"[RECORD-REQ] no active workspace for user={ctx.user_id[:8]}; "
            "reminder will NOT fire"
        )
        return ("failed", None)
    agent_id = (
        getattr(workspace, "agentId", None)
        or getattr(workspace, "ai_agent_id", None)
    )
    if not agent_id:
        logger.warning(
            "[RECORD-REQ] workspace has no agentId; reminder will NOT fire"
        )
        return ("failed", None)
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
        return ("failed", None)

    logger.info(
        f"[RECORD-REQ] {len(scheduled)} reminder(s) scheduled "
        f"recurrence={recurrence} times={[t.isoformat() for t in scheduled]}"
    )
    # 通知 inspector 提醒 tab 实时刷新
    from app.services.reminder.scheduling import notify_reminder_changed
    await notify_reminder_changed(ctx.conversation_id, kind="created")

    when_text = (
        _format_when_text(scheduled[0])
        if len(scheduled) == 1
        else " / ".join(_format_when_text(t) for t in sorted(scheduled))
    )
    return ("scheduled", when_text)


# 取消提醒的口语关键词分级 (Phase 0.1 重构):
#
# 历史 bug: 单一 _CANCEL_DIRECT_KEYWORDS 含 "我吃过了/已经吃了" 等口语短语,
# 用户在普通聊天说 "我吃过午饭了" 命中 → 一刀切 deactivate 全部 active reminder.
# 例如 user 设了 "明天 8 点开会提醒", 之后聊"我吃过午饭了" → 开会提醒被秒删.
#
# 修复策略: 把关键词拆 high/low confidence + 引入 confirmation 中间态:
#
#   HIGH (含"提醒/记"语义, 直接动作可接受):
#     1 reminder  → 立即撤 + 撤销窗口 (1h 内说"撤回"恢复)
#     2+ reminders → 列出让用户选 (不假设 user 想全撤)
#     0 reminders  → 友好告知 "我没在帮你记什么提醒哦"
#
#   LOW (歧义口语, 必须确认):
#     1 reminder + 内容相关 → ask confirmation
#     其他 LOW 命中           → 完全忽略 (走正常 reply 流程, 防止打扰)
#
# 关键洞察: 含"提醒/记" 字的关键词 = 用户主动 mention reminder = 高置信度;
# 不含的口语 ("算了/我吃过了") 在普通对话中假阳率高 = 必须 confirm.

def classify_cancel_intent(message: str) -> str:
    """口语取消语义分级判定 (不调 LLM, ~0ms).

    返回:
    - "high": 用户明确表达取消 (含"提醒/记" 关键词)
    - "low":  歧义口语, 可能是取消也可能是日常 (含"算了/我吃过了" 等)
    - "none": 没有任何取消信号

    仅 RECORD_REQUEST intent 已确认时调用; 不会跟正常聊天里的"算了"误匹配.
    """
    msg = message.strip()

    # 高置信度: 明确 mention "提醒/记"
    if any(kw in msg for kw in HIGH_CONFIDENCE_CANCEL_KEYWORDS):
        return "high"
    # 共现规则也算 HIGH
    if any(neg in msg for neg in CANCEL_NEG_TOKENS) and ("提醒" in msg or "记" in msg):
        return "high"
    # 低置信度: 仅口语命中
    if any(kw in msg for kw in LOW_CONFIDENCE_CANCEL_KEYWORDS):
        return "low"
    return "none"


# 向后兼容 alias — preflight._handle_pending_set_reminder 仍 import 这个名字.
# 仅判 "明确取消" 用; pending set_reminder 反问场景下用户回复歧义不会误删 trigger
# (因为 pending 还没建 trigger). 故只看 high.
def _is_cancel_reminder(message: str) -> bool:
    return classify_cancel_intent(message) == "high"


def _is_undo_cancel(message: str) -> bool:
    """识别 "撤回刚才的取消" 语义."""
    msg = message.strip()
    return any(kw in msg for kw in UNDO_CANCEL_KEYWORDS)


def _topic_overlap(user_message: str, reminder_summary: str) -> bool:
    """LOW confidence 命中时, 用户消息跟某 reminder 内容相关吗?

    简单 substring 匹配 (中文不分词, 用 ≥2 字 substring 求交集). 避免 LLM 调用.

    例:
      msg="我吃过药了" vs summary="提醒我吃药" → 共现 "吃药" → True
      msg="我吃过午饭了" vs summary="提醒我吃药" → 仅 "吃" 1 字不算 → False
    """
    msg_clean = user_message.strip()
    sum_clean = reminder_summary.strip()
    if not msg_clean or not sum_clean:
        return False

    # 提取消息中所有 ≥2 字 substring
    msg_bigrams = {msg_clean[i:i + 2] for i in range(len(msg_clean) - 1)}
    # 滤掉极常见无信息 bigram
    stop_bigrams = {"我的", "你的", "了的", "的我", "的你", "你了", "我了"}
    msg_bigrams -= stop_bigrams

    for bg in msg_bigrams:
        if bg in sum_clean:
            return True
    return False


async def _list_active_reminders_with_meta(
    *, user_id: str, agent_id: str | None,
) -> list[dict]:
    """查 (user, agent) 的所有 active reminder, 返回带 trigger_id/summary/when_text/memory_id 的 list.

    返给 ask confirmation 用. 按 triggerTime 升序 (最早响的在前).
    """
    from app.services.reminder.scheduling import (
        find_active_reminder_triggers, format_when_text,
    )
    triggers = await find_active_reminder_triggers(
        user_id=user_id, agent_id=agent_id,
    )
    items = []
    for t in triggers:
        action = t.actionData or {}
        items.append({
            "trigger_id": t.id,
            "summary": (action.get("summary") or "")[:80],
            "when_text": format_when_text(t.triggerTime),
            "trigger_time": t.triggerTime.isoformat() if t.triggerTime else None,
            "memory_id": action.get("memory_id"),
            "recurrence": action.get("recurrence", "once"),
            "memory_side": action.get("memory_side", "user"),
            "action_data": dict(action),  # 完整保留, undo 时复用
        })
    items.sort(key=lambda x: x.get("trigger_time") or "")
    return items


async def _cancel_active_reminders(
    *, user_id: str, agent_id: str | None = None,
    trigger_ids: list[str] | None = None,
    user_message: str = "",
    conversation_id: str | None = None,
) -> int:
    """deactivate 指定 trigger (或全部) reminder + 写 audit log + 存 undo state.

    参数:
    - agent_id: 必传 (用户路径), 不传跨 agent 误删
    - trigger_ids: 指定 ID 列表; None 表示全部 active
    - user_message: 触发取消的用户原话 (audit log 用)
    - conversation_id: 用于存 undo state (1h 内可恢复)

    返回 deactivate 条数. 同步写 changelog 和 Redis undo state.
    """
    from app.services.reminder.scheduling import (
        deactivate_reminder_triggers, save_cancel_undo,
        find_active_reminder_triggers,
    )
    from app.services.memory.storage.persistence import log_memory_changelog

    # 取消前先抓出待 deactivate 的 trigger 完整信息 (audit + undo 用)
    all_active = await find_active_reminder_triggers(
        user_id=user_id, agent_id=agent_id,
    )
    if trigger_ids is not None:
        target = [t for t in all_active if t.id in set(trigger_ids)]
    else:
        target = all_active

    if not target:
        return 0

    # 真删 (其实是 isActive=False soft delete)
    if trigger_ids is not None:
        # 只删指定的: 用 update_many with id in [...]
        try:
            result = await db.timetrigger.update_many(
                where={"id": {"in": [t.id for t in target]}, "isActive": True},
                data={"isActive": False},
            )
            n = int(result) if result is not None else len(target)
        except Exception as e:
            logger.warning(f"[REMINDER-CANCEL] selective deactivate failed: {e}")
            return 0
    else:
        # 全部: 复用现有 helper
        n = await deactivate_reminder_triggers(
            user_id=user_id, agent_id=agent_id,
        )

    # audit log: 每条 trigger 一条 changelog
    for t in target:
        memory_id = (t.actionData or {}).get("memory_id")
        if not memory_id:
            continue
        try:
            audit_value = (
                f"trigger={t.id} cancelled by user; "
                f"original_msg={user_message[:60]!r}"
            )
            await log_memory_changelog(
                user_id, memory_id, "reminder_cancelled_by_user",
                new_value=audit_value,
            )
        except Exception:
            pass  # changelog 失败不阻塞主流程

    # undo state: 1 小时内可恢复
    if conversation_id and target:
        try:
            await save_cancel_undo(
                conversation_id=conversation_id,
                triggers=[
                    {
                        "trigger_id": t.id,
                        "trigger_time": (
                            t.triggerTime.isoformat() if t.triggerTime else None
                        ),
                        "action_type": t.actionType,
                        "action_data": dict(t.actionData or {}),
                        "ai_agent_id": t.aiAgentId,
                    }
                    for t in target
                ],
            )
        except Exception as e:
            logger.warning(f"[REMINDER-CANCEL] save undo state failed: {e}")

    logger.info(
        f"[REMINDER-CANCEL] deactivated {n} trigger(s) for user={user_id[:8]} "
        f"agent={(agent_id or 'all')[:8]}; user_msg={user_message[:30]!r}; "
        f"undo window 1h"
    )
    return n


async def _undo_recent_cancel(*, conversation_id: str) -> int:
    """撤回最近 1h 内的 cancel: 重新激活 trigger + 清 undo state.

    返回恢复条数. 0 表示没 undo state 可恢复.
    """
    from app.services.reminder.scheduling import (
        load_cancel_undo, clear_cancel_undo, reactivate_reminder_triggers,
    )
    undo = await load_cancel_undo(conversation_id)
    if not undo:
        return 0
    triggers = undo.get("triggers") or []
    if not triggers:
        return 0
    n = await reactivate_reminder_triggers(triggers)
    if n > 0:
        await clear_cancel_undo(conversation_id)
    logger.info(
        f"[REMINDER-UNDO] reactivated {n}/{len(triggers)} trigger(s) "
        f"for conversation={conversation_id[:8]}"
    )
    return n


def _format_cancel_candidate_list(items: list[dict]) -> str:
    """渲染 reminder 列表给用户选 (1-indexed)."""
    return "\n".join(
        f"{i + 1}) {item['summary']} ({item['when_text']})"
        for i, item in enumerate(items)
    )


def extract_reminder_content_update(message: str) -> str | None:
    """抽取“提醒内容/文案改成 X”的新内容。"""
    msg = message.strip()
    for sep in ("：", ":"):
        if sep in msg:
            tail = msg.split(sep, 1)[1].strip()
            if tail:
                return tail
    markers = (
        "提醒内容就写", "提醒内容写成", "提醒内容改成",
        "提醒文案写成", "提醒文案改成", "内容就写", "内容写成",
        "文案写成", "改成",
    )
    for marker in markers:
        if marker in msg:
            tail = msg.split(marker, 1)[1].strip(" ，,。")
            if tail:
                return tail
    return None


async def _update_reminder_content(
    *,
    ctx: ShortCircuitCtx,
    user_message: str,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    """修改已有 active reminder 的展示/触发文案。

    只在目标唯一时直接改; 多个 active reminder 时反问，避免猜错用户要改哪一个。
    """
    from prisma import Json
    from app.services.memory.storage import repo as memory_repo
    from app.services.memory.interaction.deletion import save_pending_action
    from app.services.reminder.scheduling import notify_reminder_changed

    if not ctx.agent_id:
        return False, None

    content = extract_reminder_content_update(user_message)
    items = await _list_active_reminders_with_meta(
        user_id=ctx.user_id, agent_id=ctx.agent_id,
    )
    if not items:
        return False, None
    if len(items) > 1:
        await save_pending_action(
            ctx.conversation_id,
            action="update_reminder_content",
            candidates=items,
            summary=content or "",
        )
        candidate_list = _format_cancel_candidate_list(items)
        reply = (
            "你想改哪个提醒的内容? 我现在有这些:\n"
            f"{candidate_list}\n"
            "回数字就行。"
        )
        ctx.consumed_full_message = True
        return True, ctx.finalize(reply, kind="record_request_content_ask_multi")
    if not content:
        await save_pending_action(
            ctx.conversation_id,
            action="update_reminder_content",
            candidates=[items[0]],
            summary="",
        )
        reply = "可以，提醒内容想改成哪一句?"
        ctx.consumed_full_message = True
        return True, ctx.finalize(reply, kind="record_request_content_ask_text")

    item = items[0]
    summary = f"我让 AI 提醒: {content[:120]}"
    action_data = dict(item.get("action_data") or {})
    action_data["summary"] = summary
    try:
        await db.timetrigger.update(
            where={"id": item["trigger_id"]},
            data={"actionData": Json(action_data)},
        )
        memory_id = item.get("memory_id")
        if memory_id:
            try:
                await memory_repo.update(
                    memory_id, source=item.get("memory_side") or "user",
                    content=summary, summary=summary,
                )
            except Exception as e:
                logger.warning(f"[REMINDER-CONTENT] memory update failed: {e}")
        await notify_reminder_changed(ctx.conversation_id, kind="updated")
    except Exception as e:
        logger.warning(f"[REMINDER-CONTENT] update failed: {e}")
        return True, ctx.finalize(
            "这边改提醒内容时出了点问题，你再说一遍?",
            kind="record_request_content_failed",
        )

    ctx.consumed_full_message = True
    return True, ctx.finalize(
        f"好，提醒内容改成「{content[:60]}」了。",
        kind="record_request_content_updated",
    )


async def _handle_cancel_intent(
    *,
    user_message: str,
    cancel_level: str,
    ctx: ShortCircuitCtx,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    """取消语义分级处理.

    返回 (handled, gen):
    - handled=True: 已生成 reply (短路完成或 ask confirmation)
    - handled=False: 不该走取消流程 (LOW + 无相关 reminder), 让外层 fall-through

    Cancel level:
    - high: 含 "提醒/记" 明确语义
    - low:  仅口语命中 ("算了/我吃过了"), 容易假阳

    Decision matrix:
                     0 reminder    1 reminder    2+ reminders
    HIGH 关键词       友好告知       直接撤+undo    列出让用户选
    LOW 关键词        ❌不打扰        内容相关才ask  ❌不打扰
    """
    # save_pending_action 已在模块顶部 import (跟 deletion handler 共享 pending Redis key)
    from app.services.reminder.scheduling import notify_reminder_changed

    items = await _list_active_reminders_with_meta(
        user_id=ctx.user_id, agent_id=ctx.agent_id,
    )

    # === LOW + 无 reminder OR 多 reminder → 不打扰用户 ===
    if cancel_level == "low" and len(items) != 1:
        logger.info(
            f"[RECORD-CANCEL] LOW confidence, {len(items)} reminders, "
            f"NOT entering cancel flow ({user_message[:30]!r})"
        )
        return False, None

    # === LOW + 1 reminder, 但内容不相关 → 不打扰 ===
    if cancel_level == "low" and len(items) == 1:
        if not _topic_overlap(user_message, items[0]["summary"]):
            logger.info(
                f"[RECORD-CANCEL] LOW confidence + 1 reminder but topic mismatch, "
                f"NOT entering cancel flow ({user_message[:30]!r} vs "
                f"{items[0]['summary'][:30]!r})"
            )
            return False, None

    # === 0 reminder (HIGH only, LOW already returned above) ===
    if not items:
        # HIGH + 0 reminder: 友好告知
        ctx.consumed_full_message = True
        return True, ctx.finalize(
            "嗯嗯, 我没在帮你记什么提醒哦, 不用担心~",
            kind="record_request_cancel_no_active",
        )

    # === HIGH + 1 reminder → 直接撤 + undo 提示 (用户体感最自然) ===
    if cancel_level == "high" and len(items) == 1:
        only = items[0]
        n = await _cancel_active_reminders(
            user_id=ctx.user_id, agent_id=ctx.agent_id,
            trigger_ids=[only["trigger_id"]],
            user_message=user_message,
            conversation_id=ctx.conversation_id,
        )
        if n > 0:
            await notify_reminder_changed(ctx.conversation_id, kind="cancelled")
            reply = (
                f"好嘞, 已经把'{only['summary']}'({only['when_text']})撤掉啦~ "
                f"如果反悔了, 1 小时内跟我说'撤回'就能恢复."
            )
        else:
            reply = "嗯, 帮你撤的时候出了点小问题, 你再说一遍?"
        ctx.consumed_full_message = True
        return True, ctx.finalize(reply, kind="record_request_cancel_single")

    # === LOW + 1 reminder + 相关 → ask confirmation (低置信度必须确认) ===
    if cancel_level == "low" and len(items) == 1:
        only = items[0]
        await save_pending_action(
            ctx.conversation_id,
            action="cancel_reminder",
            candidates=[only],
            summary=user_message[:200],  # 原话存 audit 用
        )
        reply = (
            f"诶, 你是想让我取消'{only['summary']}'({only['when_text']})吗? "
            "回'对'就撤掉, 回'不是'就保留~"
        )
        ctx.consumed_full_message = True
        return True, ctx.finalize(reply, kind="record_request_cancel_ask_low")

    # === HIGH + 2+ reminders → 列出让用户选 ===
    candidate_list = _format_cancel_candidate_list(items)
    await save_pending_action(
        ctx.conversation_id,
        action="cancel_reminder",
        candidates=items,
        summary=user_message[:200],
    )
    reply = (
        "嗯, 你想取消哪个? 我现在帮你记着这些:\n"
        f"{candidate_list}\n"
        "回数字 (比如 '1' 或 '1和3'), 或者'全部', 或者'算了'~"
    )
    ctx.consumed_full_message = True
    return True, ctx.finalize(reply, kind="record_request_cancel_ask_multi")


async def handle_record_request(
    user_message: str,
    ctx: ShortCircuitCtx,
) -> tuple[bool, AsyncGenerator[dict, None] | None]:
    """RECORD_REQUEST 短路: 用户请求 AI 记一件事 / 设提醒 / 取消提醒 / 撤回取消.

    分支 (Phase 0.1 重构):
    - 撤回语义 ("撤回/恢复刚才的取消") → 1h 内可恢复; 复活 trigger
    - 取消语义 → 分级处理:
        HIGH 关键词 + 1 reminder  → 直接撤 + 撤销窗口提示
        HIGH 关键词 + 2+ reminder → 列出让用户选 (save pending)
        HIGH 关键词 + 0 reminder  → 友好告知
        LOW  关键词 + 1 内容相关  → ask confirmation (save pending)
        其他 LOW 命中             → 走正常 reply (不打扰)
    - 设置语义 (有时间) → 落库 + confirm reply ("好嘞, 8 点叫你~")
    - 设置语义 (无时间) → save_pending + 反问 ("下周哪天呀?")

    历史背景: 单一 _is_cancel_reminder 关键词命中即一刀切 deactivate 全部
    active reminder, 用户在普通对话说"我吃过午饭了"也会误删开会提醒. 重构
    为分级 + confirmation + undo, 防止 silent 数据丢失.
    """
    try:
        # 撤回语义已统一到 preflight.resolve_recent_undo (在主流程之前运行,
        # 同时处理 cancel_reminder 和 delete undo). 不在这里重复处理.

        # 取消语义分级判定
        cancel_level = classify_cancel_intent(user_message)
        if cancel_level != "none" and ctx.user_id:
            handled, gen = await _handle_cancel_intent(
                user_message=user_message,
                cancel_level=cancel_level,
                ctx=ctx,
            )
            if handled:
                return True, gen
            # 未 handle (LOW + 内容无关 / 0 reminder + LOW) → fall-through 到设置流程

        action = classify_record_request_action(user_message)
        if action == "self_note" or action == "none":
            return False, None
        if action == "memory_note":
            content = extract_record_memory_text(user_message)
            ctx.consumed_full_message = True
            return True, ctx.finalize(
                f"好，我记住这点: {content[:60]}",
                kind="record_request_memory_note",
            )
        if action == "reminder_content":
            handled, gen = await _update_reminder_content(
                ctx=ctx,
                user_message=user_message,
            )
            if handled:
                return True, gen
            return False, None

        # 设置/新增提醒走 _direct_create_reminder. 三态返回:
        status, when_text = await _direct_create_reminder(
            user_message=user_message, ctx=ctx,
        )

        if status == "asked":
            # 时间没说清 — 反问让用户补全. 第二轮由 preflight 接管.
            reply = await record_ask_time(
                user_message=user_message,
                personality_brief=_agent_name(ctx.agent),
            )
            # 反问后整句已被 set_reminder pending 吸收 — sub-intent 不再处理残句
            ctx.consumed_full_message = True
            return True, ctx.finalize(reply, kind="record_request_ask_time")

        # status == "scheduled" or "failed"
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
        # record_request_scheduled / _failed
        return True, ctx.finalize(reply, kind=f"record_request_{status}")
    except Exception as e:
        logger.warning(f"Record request short-circuit failed, falling through: {e}")
        return False, None
