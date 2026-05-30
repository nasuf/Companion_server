"""Chat service — optimized for low latency.

Hot path (user-facing, ~2s):
  save msg → parallel(vector retrieval + load cached context) → prompt → stream LLM

Background (fire-and-forget, after response):
  user emotion metadata + memory pipeline + trait/patience updates
"""

import asyncio
import json
import logging
import random
import re
from datetime import UTC, datetime
from time import perf_counter
from collections.abc import AsyncGenerator
from typing import Any

from prisma import Json

from app.db import db
from app.observability.events import (
    EVT_INTENT_SPLIT,
    EVT_MEMORY_CONTRADICTION,
    EVT_REPLY_EMITTED,
    EVT_REPLY_EMOTION,
)
from app.services.llm.models import get_chat_model, convert_messages
from app.services.chat.prompt_builder import build_system_prompt, build_chat_messages
from app.services.prompting.store import (
    get_prompt_text,
    reset_prompt_runtime_context,
    set_prompt_runtime_context,
)
from app.services.prompting.trace_components import (
    prompt_hash,
    record_prompt_render,
    reset_prompt_render_trace,
    snapshot_prompt_render_traces,
    start_prompt_render_trace,
)
from app.services.prompts.system_prompts import (
    MAX_PER_REPLY, MAX_REPLY_COUNT, MAX_TOTAL_CHARS,
)
from app.services.schedule_domain.timing import (
    calculate_reply_delay,
    explain_delay_reason,
)
from app.services.memory.interaction.contradiction import (
    detect_l1_contradiction, generate_contradiction_inquiry,
    save_pending_contradiction,
)  # analyze/apply/load/clear 已由 preflight.resolve_pending_contradiction 接管
from app.services.memory.interaction.retrieval_feedback import (
    resolve_retrieval_feedback_correction,
)
from app.services.memory.retrieval.access_log import log_memory_access
from app.services.topic import push_topic, format_topic_context
from app.services.schedule_domain.time_service import build_time_context
from app.services.schedule_domain.time_parser import parse_time_expressions, has_explicit_time
from app.services.schedule_domain.schedule import (
    format_schedule_context,
    get_cached_schedule,
    get_current_status,
)
from app.services.interaction.boundary import (
    PATIENCE_MAX,
    get_patience_prompt_instruction,
)
from app.services.relationship.intimacy import get_relationship_stage
from app.services.relationship.emotion import is_negative_emotion
from app.services.chat.intent_dispatcher import (
    detect_current_state_fast_path,
    detect_intent_unified,
    infer_schedule_query_type,
    is_explicit_current_state_query,
    IntentType,
    IntentResult,
    LABEL_TO_INTENT,
)
from app.services.chat.multi_intent import (
    finalize_short_circuit as _finalize_short_circuit,
    process_sub_intents as _process_sub_intents,
    short_circuit_reply as _short_circuit_reply_impl,
)
from app.services.chat.intent_handlers import (
    ShortCircuitCtx,
    handle_apology_promise,
    handle_conversation_end,
    handle_crisis,
    handle_crisis_followup,
    handle_current_state,
    handle_deletion,
    handle_record_request,
    handle_schedule_adjust,
    handle_schedule_query,
)
from app.services.chat.crisis_guard_phase import run_crisis_guard
from app.services.chat.intent_replies import (
    delay_explanation_reply as _delay_explanation_reply,
    memory_weak_reply as _memory_weak_reply,
    memory_medium_reply as _memory_medium_reply,
    memory_strong_reply as _memory_strong_reply,
    memory_l3_reply as _memory_l3_reply,
    l3_trigger_analyze as _l3_trigger_analyze,
    # split_reply_to_n_sentences 已删除 — 主 LLM 直接按 || 输出
    ai_reply_emotion as _ai_reply_emotion,
)
from app.services.chat.reply_post_process import emit_replies as _emit_replies
from app.services.chat.reply_generate import generate_reply as _generate_reply
from app.services.chat.preflight import (
    PreflightCtx,
    resolve_pending_contradiction,
    resolve_pending_deletion,
    resolve_recent_undo,
)
from app.services.chat.boundary_phase import BoundaryPhaseCtx, run_boundary
from app.services.chat.data_fetch_phase import (
    FetchedContext,
    fetch_parallel_context,
    format_recent_context,
    maybe_awaken_l3,
)
from app.services.chat.post_process import (
    save_replies as _save_replies,
    run_post_process as _background_post_process,
    _bg_memory_pipeline,
)
from app.services.chat.tracing import LangSmithTracer
from app.services.mbti import get_mbti
from app.services.interaction.reply_context import actual_delay_seconds, save_last_reply_timestamp
from app.services.proactive.state import start_or_restart_proactive_session
from app.services.rules.chat_keywords import (
    DISTRESS_KEYWORDS,
    RELATIONAL_COMPLAINT_KEYWORDS,
)
from app.services.runtime.tasks import fire_background as _fire_background

logger = logging.getLogger(__name__)


async def _short_circuit_reply(
    reply: str,
    conversation_id: str,
    agent_id: str | None,
    user_id: str,
    *,
    sub_intent_mode: bool = False,
    reply_index_offset: int = 0,
    include_done: bool = True,
    extra_metadata: dict | None = None,
    trace_id: str | None = None,
) -> list[dict]:
    """Orchestrator-side adapter that injects `_save_replies`."""
    return await _short_circuit_reply_impl(
        reply, conversation_id, agent_id, user_id, _save_replies,
        sub_intent_mode=sub_intent_mode,
        reply_index_offset=reply_index_offset,
        include_done=include_done,
        extra_metadata=extra_metadata,
        trace_id=trace_id,
    )


def _downgrade_non_explicit_current_state(
    detected_intent: IntentResult,
    user_message: str,
    response_diagnostics: dict[str, Any],
) -> IntentResult:
    """Keep CURRENT_STATE only for explicit AI state questions.

    LLM intent can over-label memory/identity recall questions as current-state.
    If that label survives, the reply path cannot use memory tier prompts even
    after retrieval. Demote those misses to normal chat so memory relevance can
    drive the final reply.
    """
    if detected_intent.intent != IntentType.CURRENT_STATE:
        return detected_intent
    if is_explicit_current_state_query(user_message):
        return detected_intent

    metadata = dict(detected_intent.metadata or {})
    metadata["downgraded_from"] = IntentType.CURRENT_STATE.value
    metadata["downgrade_reason"] = "not_explicit_current_state"
    response_diagnostics["intent_downgrade_reason"] = "not_explicit_current_state"
    return IntentResult(
        intent=IntentType.NONE,
        confidence=detected_intent.confidence,
        metadata=metadata,
    )


def _route_current_schedule_query_to_current_state(
    detected_intent: IntentResult,
    user_message: str,
    response_diagnostics: dict[str, Any],
) -> IntentResult:
    """Treat present-tense availability questions as current-state chat.

    The schedule-query path is for concrete schedule/routine/date lookups. For
    "现在忙吗/你现在有空吗", using that path exposes the full schedule table to
    the reply prompt and makes the assistant sound like it is reading a diary.
    """
    if detected_intent.intent != IntentType.SCHEDULE_QUERY:
        return detected_intent
    if (detected_intent.metadata or {}).get("query_type") != "current":
        return detected_intent
    if not is_explicit_current_state_query(user_message):
        return detected_intent

    metadata = dict(detected_intent.metadata or {})
    metadata["rerouted_from"] = IntentType.SCHEDULE_QUERY.value
    metadata["reroute_reason"] = "current_availability_as_current_state"
    response_diagnostics["intent_reroute_reason"] = "current_availability_as_current_state"
    return IntentResult(
        intent=IntentType.CURRENT_STATE,
        confidence=detected_intent.confidence,
        metadata=metadata,
    )


def _downgrade_non_explicit_current_schedule_query(
    detected_intent: IntentResult,
    user_message: str,
    response_diagnostics: dict[str, Any],
) -> IntentResult:
    """Keep current schedule-query only when the user actually asks.

    The unified classifier can over-label social openers such as "好几天没找你聊了"
    as "计划查询" with query_type=current. If that survives, a plain greeting
    takes the schedule short-circuit path and may spawn extra sub-intents.
    """
    if detected_intent.intent != IntentType.SCHEDULE_QUERY:
        return detected_intent
    if (detected_intent.metadata or {}).get("query_type") != "current":
        return detected_intent
    if infer_schedule_query_type(user_message, require_query_cue=True):
        return detected_intent
    if is_explicit_current_state_query(user_message):
        return detected_intent

    metadata = dict(detected_intent.metadata or {})
    metadata["downgraded_from"] = IntentType.SCHEDULE_QUERY.value
    metadata["downgrade_reason"] = "not_explicit_current_schedule_query"
    metadata.pop("fragments", None)
    response_diagnostics["intent_downgrade_reason"] = "not_explicit_current_schedule_query"
    return IntentResult(
        intent=IntentType.NONE,
        confidence=detected_intent.confidence,
        metadata=metadata,
    )


def _filter_non_explicit_sub_fragments(
    fragments: dict[str, str],
    response_diagnostics: dict[str, Any],
) -> dict[str, str]:
    """Drop over-eager current-state/schedule sub-fragments from casual text."""
    filtered: dict[str, str] = {}
    dropped: list[str] = []
    for label, text in fragments.items():
        fragment = str(text).strip()
        if not fragment:
            continue
        if label == "询问当前状态" and not is_explicit_current_state_query(fragment):
            dropped.append(label)
            continue
        if (
            label == "计划查询"
            and not infer_schedule_query_type(fragment, require_query_cue=True)
            and not is_explicit_current_state_query(fragment)
        ):
            dropped.append(label)
            continue
        filtered[label] = fragment
    if dropped:
        response_diagnostics["intent_sub_fragments_dropped"] = dropped
    return filtered


async def _intent_llm_reply(
    agent,
    user_message: str,
    instruction: str,
) -> str:
    """Generate a short LLM reply for a special intent (farewell, reconciliation, etc.)."""
    prompt = await build_system_prompt(agent=agent, reply_count=1, reply_total=60)
    base_prompt_hash = prompt_hash(prompt)
    base_components: list[dict[str, Any]] = []
    for trace in reversed(snapshot_prompt_render_traces()):
        if trace.get("prompt_hash") == base_prompt_hash:
            raw_components = trace.get("components")
            if isinstance(raw_components, list):
                base_components = [dict(item) for item in raw_components if isinstance(item, dict)]
            break
    appendix_tpl = await get_prompt_text("chat.special_instruction_appendix")
    appendix_start = len(prompt)
    appendix = appendix_tpl.format(instruction=instruction)
    prompt += appendix
    record_prompt_render(
        prompt,
        prompt_key="chat.system_base",
        components=[
            *base_components,
            {
                "prompt_key": "chat.special_instruction_appendix",
                "start": appendix_start,
                "end": appendix_start + len(appendix),
                "editable": True,
            },
        ],
        source="chat.special_instruction",
    )
    model = get_chat_model()
    result = await model.ainvoke(convert_messages([
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ]))
    return result.content.strip().split("||")[0][:60]


def _previous_assistant_message(recent_messages: list[Any], current_user_message_id: str | None) -> Any | None:
    """Find the assistant turn immediately before the current user turn."""
    current_index = len(recent_messages)
    if current_user_message_id:
        for idx, message in enumerate(recent_messages):
            if getattr(message, "id", None) == current_user_message_id:
                current_index = idx
                break
    for message in reversed(recent_messages[:current_index]):
        if getattr(message, "role", None) == "assistant":
            return message
    return None


def _current_turn_message_ids(
    reply_context: dict | None,
    current_user_message_id: str | None,
) -> set[str]:
    ids: set[str] = set()
    raw_ids = (reply_context or {}).get("turn_message_ids")
    if isinstance(raw_ids, list):
        ids.update(item for item in raw_ids if isinstance(item, str) and item)
    if current_user_message_id:
        ids.add(current_user_message_id)
    return ids


def _ensure_current_user_message(
    messages: list[dict],
    *,
    user_message: str,
    user_message_id: str | None,
    reply_context: dict | None,
) -> list[dict]:
    """Ensure the prompt context contains the current user turn.

    Normal chat saves the user message before fetching history, but delayed /
    aggregated delivery can hit visibility or payload races. If the current
    turn is absent, the main LLM sees the previous user turn as latest and can
    regenerate the prior reply. Add a synthetic current turn as a final guard.
    """
    if user_message_id:
        if any(m.get("id") == user_message_id for m in messages):
            return messages
    elif messages:
        last = messages[-1]
        if last.get("role") == "user" and last.get("content") == user_message:
            return messages

    received_at = None
    if user_message_id and reply_context:
        received_at = reply_context.get("latest_received_at") or reply_context.get("received_at")
    if user_message_id and (not isinstance(received_at, str) or not received_at):
        received_at = datetime.now(UTC).isoformat()

    return [
        *messages,
        {
            "id": user_message_id,
            "role": "user",
            "content": user_message,
            "createdAt": received_at,
            "synthetic_current": True,
        },
    ]


def _parse_message_created_at(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _max_user_created_at(messages: list[dict]) -> datetime | None:
    times = [
        ts for ts in (
            _parse_message_created_at(m.get("createdAt"))
            for m in messages
            if m.get("role") == "user"
        )
        if ts is not None
    ]
    return max(times, default=None)


async def _record_memory_retrieval_feedback(
    *,
    assistant_message_id: str,
    assistant_reply: str,
    user_message: str,
    user_message_id: str | None,
) -> None:
    """Write a diagnostic signal when the next user turn may correct memory use."""
    try:
        from app.services.memory.retrieval.trace import build_memory_retrieval_feedback

        previous = await db.message.find_unique(where={"id": assistant_message_id})
        if not previous:
            return
        metadata = previous.metadata if isinstance(previous.metadata, dict) else {}
        feedback = build_memory_retrieval_feedback(
            user_message=user_message,
            previous_assistant_reply=getattr(previous, "content", None) or assistant_reply,
            previous_metadata=metadata,
        )
        if not feedback:
            return

        feedback["assistant_message_id"] = assistant_message_id
        feedback["user_message_id"] = user_message_id
        existing = metadata.get("memory_retrieval_feedback")
        if (
            isinstance(existing, dict)
            and existing.get("user_message_id") == user_message_id
        ):
            return
        if isinstance(existing, dict):
            feedback = {**existing, **feedback}

        await db.message.update(
            where={"id": assistant_message_id},
            data={"metadata": Json({**metadata, "memory_retrieval_feedback": feedback})},
        )
    except Exception:
        logger.exception("[MEM-FEEDBACK] failed to record memory retrieval feedback")


# --- Multi-reply split & validate (PRD §3.2.1/§3.2.2) ---

_SENTENCE_END = re.compile(r'[。！？…～~!?]+')

# 切分分隔符: 优先 || (我们要求 LLM 输出的); 兼容 LLM 自作主张用空行分段的情况
# (\n\n+) — 不切就会被前端 pre-wrap 渲染成同一气泡内带空行.
_REPLY_SPLIT_RE = re.compile(r'\|\||\n{2,}')

# 单条回复内部残留的换行 (单个 \n 或单个 \r) 收敛为空格,
# 避免句子内 "你好\n吗" 这种意外换行被 pre-wrap 渲染成断行.
_INTRA_REPLY_WS_RE = re.compile(r'[\r\n]+')
_LONG_REPLY_SOFT_BREAK_CHARS = "。！？…～~!?，,；;、"
_NON_TERMINAL_REPLY_END_RE = re.compile(r"[，,、；;：:]+$")


def _clean_reply_part(text: str) -> str:
    """单条回复内部规范化: 去首尾空白 + 单个换行折叠成空格."""
    return _INTRA_REPLY_WS_RE.sub(" ", text).strip()


def _strip_non_terminal_reply_end(text: str) -> str:
    """Remove dangling boundary punctuation from a split message bubble."""
    return _NON_TERMINAL_REPLY_END_RE.sub("", text.rstrip()).rstrip()


def _polish_split_boundaries(parts: list[str]) -> list[str]:
    """Clean punctuation that only exists because a reply was split."""
    if len(parts) <= 1:
        return parts
    polished = [
        _strip_non_terminal_reply_end(part) if idx < len(parts) - 1 else part
        for idx, part in enumerate(parts)
    ]
    return [part for part in polished if part]


def truncate_at_sentence(text: str, max_len: int) -> str:
    """截断至max_len内最后一个句子边界。"""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    match = None
    for m in _SENTENCE_END.finditer(truncated):
        match = m
    if match and match.end() > max_len // 2:
        return truncated[:match.end()]
    return truncated


def _find_soft_reply_cut(text: str, max_len: int) -> int:
    """Find a punctuation cut point for long single-bubble fallbacks."""
    window = text[:max_len]
    min_cut = max(8, max_len // 3)
    best = -1
    for idx, ch in enumerate(window):
        if ch in _LONG_REPLY_SOFT_BREAK_CHARS and idx + 1 >= min_cut:
            best = idx + 1
    return best


def _split_long_reply_part(text: str, max_len: int, max_count: int) -> list[str]:
    """Split an overlong no-delimiter reply at punctuation instead of mid-clause."""
    remaining = text.strip()
    parts: list[str] = []
    while remaining and len(parts) < max_count:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break
        cut = _find_soft_reply_cut(remaining, max_len)
        if cut <= 0:
            parts.append(truncate_at_sentence(remaining, max_len))
            break
        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()
    return parts


def split_and_validate_replies(
    raw: str,
    max_count: int = MAX_REPLY_COUNT,
    max_per_reply: int = MAX_PER_REPLY,
    max_total: int = MAX_TOTAL_CHARS,
) -> list[str]:
    """按 || 或空行分割 LLM 输出, 校验条数/单条长度/总长度.

    LLM 偶尔不按 prompt 用 ||, 改用空行分段 — 不切的话前端 pre-wrap 会把
    \\n\\n 渲染成单气泡里的空行, 视觉跟正常多条回复混淆. 这里把空行也当
    分隔符. 单条内的孤立 \\n 由 _clean_reply_part 折叠成空格.
    """
    has_explicit_split = bool(_REPLY_SPLIT_RE.search(raw))
    parts = [_clean_reply_part(p) for p in _REPLY_SPLIT_RE.split(raw)]
    parts = [p for p in parts if p]
    if not parts:
        return [_clean_reply_part(raw) or "..."]
    if not has_explicit_split and len(parts) == 1 and len(parts[0]) > max_per_reply:
        parts = _split_long_reply_part(parts[0], max_per_reply, max_count)
    else:
        parts = parts[:max_count]
        parts = [truncate_at_sentence(p, max_per_reply) for p in parts]
    result: list[str] = []
    total = 0
    for p in parts:
        if total + len(p) > max_total:
            remaining = max_total - total
            if remaining > 5:
                result.append(truncate_at_sentence(p, remaining))
            break
        result.append(p)
        total += len(p)
    result = _polish_split_boundaries(result)
    return result or [parts[0][:max_per_reply]]


def detect_relational_context(message: str, user_emotion: dict | None) -> str | None:
    """Detect relationship repair / distress cues that need more human handling."""
    text = message.strip()
    if any(keyword in text for keyword in RELATIONAL_COMPLAINT_KEYWORDS):
        return (
            "用户这句更像是在确认你有没有在意Ta，或者在表达被忽略感。"
            "先短促地接住关系情绪，比如安抚、解释半句、表明你不是故意的；"
            "不要一上来就长解释，也不要立刻抛万能反问。"
        )

    negative_emotion = is_negative_emotion(user_emotion)
    if any(keyword in text for keyword in DISTRESS_KEYWORDS) or negative_emotion:
        return (
            "用户这句带明显低落或烦闷情绪。"
            "先回应当下感受，语气真一点、短一点；"
            "不要套模板安慰，不要一下子给很多建议，追问也只问最贴当前情绪的一句。"
        )
    return None


# 最近 6 条 ≈ 3 轮对话, 覆盖 "好" / "嗯" / "对" 类短应答对前一两轮
# AI 问话的 context 依赖; 再大会挤占 intent prompt 的 token 预算.
_INTENT_CONTEXT_WINDOW = 6


async def _fetch_intent_context(
    conversation_id: str,
    *,
    exclude_id: str | None = None,
    exclude_ids: set[str] | None = None,
    exclude_content: str | None = None,
) -> str:
    """拉最近 N 条消息拼成意图识别 prompt 的上下文段落。

    spec §3.3 step 1 要求识别 "用户消息及上下文". 常见场景:
    AI 问 "要我再陪你一会儿吗?" + 用户回 "好" — 必须结合 AI 上一句
    才能判定用户 "好" 是 作息调整 意图.

    当前消息已经作为 {user_message} 传给 prompt, 不应再出现在 context 里.
    优先用 `exclude_id` (已落库场景) 精确排除; 若消息尚未入库或只有内容,
    回退到 `exclude_content` 字符串匹配 (仅第一条命中者).
    """
    try:
        rows = await db.message.find_many(
            where={"conversationId": conversation_id},
            order={"createdAt": "desc"},
            take=_INTENT_CONTEXT_WINDOW + 1,
        )
    except Exception as e:
        # 静默降级对意图识别质量影响较大, 生产环境需能 grep 到
        logger.warning(f"intent context fetch failed: {e}")
        return ""

    excluded_ids = set(exclude_ids or set())
    if exclude_id:
        excluded_ids.add(exclude_id)

    # Prisma desc 排序下, 首条即最新; 当前消息通常在这里.
    # 回退按内容匹配时只过滤第一条 (即最新那条) 命中的用户消息,
    # 避免用户连发两条相同短消息 ("好" / "好") 把上一轮的 "好" 也丢掉.
    lines: list[str] = []
    content_fallback_consumed = False
    for row in rows:
        content = (getattr(row, "content", "") or "").strip()
        if not content:
            continue
        role = "AI" if getattr(row, "role", "") == "assistant" else "用户"
        if getattr(row, "id", None) in excluded_ids:
            continue
        if (
            not exclude_id
            and exclude_content
            and not content_fallback_consumed
            and role == "用户"
            and content == exclude_content
        ):
            content_fallback_consumed = True
            continue
        lines.append(f"{role}: {content}")

    # desc 顺序 → 反转为时间顺序, 只保留最近 N 条
    lines.reverse()
    lines = lines[-_INTENT_CONTEXT_WINDOW:]
    return "\n".join(lines)


async def stream_chat_response(
    conversation_id: str,
    user_message: str,
    agent,
    user_id: str,
    reply_context: dict | None = None,
    *,
    save_user_message: bool = True,
    user_message_id: str | None = None,
    delivered_from_queue: bool = False,
    sub_intent_mode: bool = False,
    forced_intent: IntentType | None = None,
    reply_index_offset: int = 0,
    parent_patience: int | None = None,
    parent_trace_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """spec §3.3 step 3：多意图拆分后递归调用本函数处理每个子片段。

    sub_intent_mode=True 的子调用：跳过用户消息 DB 写入、边界/pending 检查、
    延迟解释、done 事件、save_last_reply_timestamp 与后台任务；由父调用统一完成。
    forced_intent 指定片段意图不再识别；reply_index_offset 让回复 index 顺延；
    parent_patience 复用父调用的耐心值，避免每个子片段再读一次 Redis。
    子调用共享 reply_context 沿用首条消息的 due_at（spec §6 延迟批处理）。
    """
    pending_sub_fragments: dict[str, str] = {}
    skip_time_memory_lookup = bool(
        (reply_context or {}).get("skip_time_memory_lookup")
    )

    # 碎片聚合/延迟队列在入队时已落库；sub_intent_mode 共享父调用的原始消息
    if save_user_message and not sub_intent_mode:
        saved_msg = await db.message.create(
            data={
                "conversation": {"connect": {"id": conversation_id}},
                "role": "user",
                "content": user_message,
            }
        )
        user_message_id = saved_msg.id

    agent_id = getattr(agent, "id", None)
    conversation = await db.conversation.find_unique(where={"id": conversation_id})
    workspace_id = getattr(conversation, "workspaceId", None)

    # 把当前 agent 绑到 ContextVar, 后续 get_chat_model() 等据此应用 per-agent
    # override (无 override 时回落 system / env). 整条流式期间 ContextVar 有效,
    # 异步任务 (post_process / memory pipeline) 通过 fire_background_fn copy_context
    # 自己拿快照, 不受 finally reset 影响.
    # 预置 None token 让 finally reset 在 bind 抛异常时也能 no-op.
    from app.services.runtime_config import bind_agent_context, reset_current_agent
    _agent_ctx_token = None
    _agent_ctx_token = await bind_agent_context(agent_id)
    _prompt_ctx_token = set_prompt_runtime_context(agent_id=agent_id, user_id=user_id)

    # --- LLM usage 累加 session ---
    # 父调用启 session, 所有 phase 内 LLM wrapper 自动 record 进来; 出口 finally 写一行
    # llm_usage. sub_intent_mode 共享父 session, 不开新的 (避免子片段重复计费).
    from app.services.llm import usage_tracker
    usage_token = usage_tracker.start_session() if not sub_intent_mode else None

    # --- LangSmith trace ---
    # 主调用 (sub_intent_mode=False): 开新 chat_request span 作为 root.
    # sub_intent 调用: attach 到 parent trace_id, 不开新 span — sub 内的 LLM
    # 调用通过 LangSmith contextvars 自动 attach 到 parent ctx 形成嵌套树,
    # sub 产生的消息 metadata.trace_id 跟 parent 一致, 用户点 trace 跳到 root
    # 视图能看到完整树 (而非只看 sub 子树).
    if sub_intent_mode and parent_trace_id:
        tracer = LangSmithTracer(user_message, conversation_id).attach_to_parent(parent_trace_id)
    else:
        tracer = LangSmithTracer(user_message, conversation_id).enter()
    retrieval_trace_token = None
    prompt_trace_token = None
    if not sub_intent_mode:
        from app.services.memory.retrieval.trace import start_retrieval_trace
        retrieval_trace_token = start_retrieval_trace()
        prompt_trace_token = start_prompt_render_trace()

    # spec §2.1/§2.2 全消息走 post_process. 短路路径直接 return 跳过主路径末尾的
    # _fire_background(post_process) → 必须 finally 兜底, 否则 7 个短路意图 (终结/计划查询/
    # 询问当前状态/作息调整/道歉/删除/L3) 跟 2 个 preflight (矛盾追问/删除确认) 命中时
    # 全部丢失记忆抽取 + 用户情绪 metadata + trait + 正向恢复后台任务. boundary 路径例外:
    # CLAUDE.md §3.3 design — apology 自己 fire memory pipeline, blocked 故意跳;
    # finally 通过 `boundary_ctx.stopped` 直接判别, 无需独立 flag.
    post_process_fired = False
    boundary_ctx: BoundaryPhaseCtx | None = None
    preflight_ctx: PreflightCtx | None = None
    sc_ctx: ShortCircuitCtx | None = None
    prompt_user_emotion: dict | None = None
    messages_dicts: list[dict] = []

    try:
        # 必须早于 boundary phase: spec §2.6 步骤 4 攻击目标识别需要最近几轮做上下文,
        # 单看孤立含糊代词 ("不然呢") 时 LLM 无法判定 "你" 指 AI.
        recent_messages = await db.message.find_many(
            where={"conversationId": conversation_id},
            order={"createdAt": "desc"},
            take=30,
        )
        recent_messages.reverse()
        messages_dicts = [
            {
                "id": getattr(m, "id", None),  # 给 format_recent_context 排除当前消息用
                "role": m.role,
                "content": m.content,
                "createdAt": m.createdAt.isoformat() if getattr(m, "createdAt", None) else None,
            }
            for m in recent_messages
        ]
        messages_dicts = _ensure_current_user_message(
            messages_dicts,
            user_message=user_message,
            user_message_id=user_message_id,
            reply_context=reply_context,
        )
        current_turn_ids = _current_turn_message_ids(reply_context, user_message_id)
        covered_until_user_ts = _max_user_created_at(messages_dicts)
        if not sub_intent_mode:
            previous_assistant = _previous_assistant_message(
                recent_messages, user_message_id,
            )
            if previous_assistant is not None:
                _fire_background(_record_memory_retrieval_feedback(
                    assistant_message_id=previous_assistant.id,
                    assistant_reply=previous_assistant.content,
                    user_message=user_message,
                    user_message_id=user_message_id,
                ))

        # ── 危机守护 (P0) ──────────────────────────────────────────────
        # 统一入口: 直接危机、含蓄危机、危机照护延续、release、边界跳过、
        # 以及非攻击危机照护下的耐心恢复都由 crisis_guard_phase 决策。
        # Orchestrator 只消费结构化结果，避免安全规则散落在多个分支。
        crisis_decision = await run_crisis_guard(
            conversation_id=conversation_id,
            user_id=user_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            user_message=user_message,
            sub_intent_mode=sub_intent_mode,
            messages_dicts=messages_dicts,
            user_message_id=user_message_id,
        )
        crisis_force_intent = crisis_decision.crisis_force_intent
        crisis_followup_active = crisis_decision.crisis_followup_active
        crisis_care_turn = crisis_decision.crisis_care_turn
        recent_crisis_context = crisis_decision.recent_crisis_context
        crisis_followup_check_mode = crisis_decision.crisis_followup_check_mode

        # spec §2.6 边界系统全流程（含步骤 2-6 + 步骤 6 中/低耐心短路）
        cached_patience = (
            crisis_decision.cached_patience
            if crisis_decision.cached_patience is not None
            else (parent_patience if parent_patience is not None else PATIENCE_MAX)
        )
        recent_context_text = format_recent_context(
            messages_dicts,
            exclude_message_id=user_message_id,
            exclude_message_ids=current_turn_ids,
        )
        if not crisis_decision.skip_boundary:
            # boundary_ctx.recent_context 给 short-circuit handler 的 prompt {context} 用,
            # 排除当前 user_message_id 防 LLM 看到该消息两遍 ({message} + {context} 重复,
            # 实测 trace 2026-05-07 16:57 已确认).
            boundary_ctx = BoundaryPhaseCtx(
                conversation_id=conversation_id,
                agent_id=agent_id,
                user_id=user_id,
                agent=agent,
                user_message=user_message,
                sub_intent_mode=sub_intent_mode,
                parent_patience=parent_patience,
                tracer=tracer,
                short_circuit_fn=_short_circuit_reply,
                fire_background_fn=_fire_background,
                bg_memory_pipeline_fn=_bg_memory_pipeline,
                recent_context=recent_context_text,
            )
            async for evt in run_boundary(boundary_ctx):
                yield evt
            if boundary_ctx.stopped:
                # boundary phase 自己决定是否进 memory pipeline (apology fires _bg_memory_pipeline,
                # blocked 故意跳 per CLAUDE.md §3.3) — finally 通过 boundary_ctx.stopped 跳过兜底.
                return
            cached_patience = boundary_ctx.cached_patience

        # Pending 跨消息状态：矛盾追问 / 删除确认。用户的回答不会带意图关键词，
        # 必须在意图识别前先匹配 Redis 里的待处理状态。sub_intent_mode 下跳过。
        # crisis 路径也跳过: 用户求救信号优先级高于一切, pending 留着等用户安全后处理.
        if not sub_intent_mode and not crisis_care_turn:
            preflight_ctx = PreflightCtx(
                conversation_id=conversation_id,
                agent_id=agent_id,
                user_id=user_id,
                agent=agent,
                tracer=tracer,
                short_circuit_fn=_short_circuit_reply,
            )

            # Phase 0.2: 用户说"撤回/恢复" 时优先 short-circuit, 跳过任何 pending
            # 状态. 1h 内的 cancel_reminder + delete 都可恢复, 都未命中则告知.
            async for evt in resolve_recent_undo(user_message, preflight_ctx):
                yield evt
            if preflight_ctx.stopped:
                return

            async for evt in resolve_pending_contradiction(user_message, preflight_ctx):
                yield evt
            if preflight_ctx.stopped:
                return

            async for evt in resolve_pending_deletion(user_message, preflight_ctx):
                yield evt
            if preflight_ctx.stopped:
                return

            async for evt in resolve_retrieval_feedback_correction(
                user_message=user_message,
                previous_assistant=previous_assistant,
                ctx=preflight_ctx,
                workspace_id=workspace_id,
            ):
                yield evt
            if preflight_ctx.stopped:
                return

        current_state_fast_path = (
            forced_intent is None
            and not sub_intent_mode
            and not crisis_care_turn
            and detect_current_state_fast_path(user_message)
        )
        response_diagnostics: dict[str, Any] = {
            "version": 1,
            "reply_path": None,
            "memory_relevance": None,
            "main_prompt_built": False,
            "main_prompt_build_ms": None,
            "memory_retrieval_skipped_reason": None,
            "empty_prompt_sections_removed_count": None,
            "intent_fast_path": (
                "current_state_phrase" if current_state_fast_path else None
            ),
            "crisis_guard_status": crisis_decision.status,
            "crisis_guard_reason": crisis_decision.reason,
            "crisis_semantic_checked": crisis_decision.semantic_checked,
            "crisis_semantic_detected": crisis_decision.semantic_detected,
            "crisis_boundary_attack_present": (
                crisis_decision.boundary_attack_present
                if crisis_decision.crisis_care_turn else None
            ),
        }

        # P0-2: auto-intent 路径下提前 kick off fetch 与 intent 并行 (节省 600-1500ms).
        # 短路意图 (CONVERSATION_END/APOLOGY_PROMISE/DELETION) 命中时 cancel fetch_task
        # 避免浪费已 in-flight 的 LLM. 非短路场景 await fetch 后单独跑 L3 awakening
        # (因 L3 需要 intent + relevance 双信号).
        # sub_intent_mode (forced_intent != None) 同步取得 intent 无需并行, fetch_task=None
        # 走原同步路径在下方 fetch_parallel_context.
        #
        # crisis 路径 (P0): 跳过 fetch_parallel_context 的并行子任务 (relevance LLM /
        # user emotion LLM / topic_intimacy / time_memories / schedule / 等),
        # 只起安全专用记忆召回 + portrait 两个 (handle_crisis 仅需这俩). 实测 trace
        # 2026-05-07 16:57: crisis 路径走完整 fetch 浪费 4s 无关 LLM (relevance 1.2s
        # + 用户情绪 1.4s) — 求救场景下用户体感冷漠. 牺牲: message metadata 缺 emotion 字段
        # (post_process 的 _bg_user_emotion 会 gracefully 跳过 None).
        fetch_task: asyncio.Task | None = None
        crisis_memory_task: asyncio.Task | None = None
        crisis_portrait_task: asyncio.Task | None = None
        early_parsed_times: list = []
        if crisis_force_intent or crisis_followup_active:
            # 轻量 fetch — 仅 handle_crisis 必需的两项 (lazy import 防循环依赖).
            # crisis 首轮走专用安全召回; follow-up 用双通道召回: 安全背景 +
            # 当前话题记忆, 避免用户尝试转移话题时只看到危机记忆.
            from app.services.memory.retrieval.safety import (
                retrieve_crisis_followup_memories,
                retrieve_crisis_memories,
            )
            from app.services.portrait import get_latest_portrait
            if crisis_followup_active:
                crisis_memory_task = asyncio.create_task(
                    retrieve_crisis_followup_memories(
                        user_message,
                        user_id,
                        recent_context=recent_crisis_context,
                        workspace_id=workspace_id,
                    )
                )
            else:
                crisis_memory_task = asyncio.create_task(
                    retrieve_crisis_memories(
                        user_message, user_id, workspace_id=workspace_id,
                    )
                )
            if agent_id:
                crisis_portrait_task = asyncio.create_task(
                    get_latest_portrait(user_id, agent_id)
                )
        elif forced_intent is None and not current_state_fast_path:
            early_parsed_times = (
                parse_time_expressions(user_message)
                if not skip_time_memory_lookup and has_explicit_time(user_message) else []
            )
            fetch_task = asyncio.create_task(fetch_parallel_context(
                user_id=user_id, agent_id=agent_id, workspace_id=workspace_id,
                user_message=user_message,
                messages_dicts=messages_dicts, parsed_times=early_parsed_times,
                # detected_intent / l3_trigger_classify_fn 都不传 → fetch 跳过 L3,
                # 由 orchestrator 在 intent 出来后单独跑.
            ))

        # --- 统一意图识别：spec §3.3 step 1 严格实现 ---
        # 每条用户消息都调小模型做意图分类, 并把最近对话历史作为上下文注入.
        # 不再区分消息长度 — 短消息如 "好" / "嗯" 只有结合 AI 上一句
        # ("要我再陪你一会儿吗?") 才能识别出 "作息调整" 意图.
        # crisis force: 关键字命中直接 force CRISIS, 不调 intent LLM (省成本 +
        # 保证不被误归 — 实证 LLM 把"我想跳楼"误归"询问当前状态").
        if forced_intent is not None:
            # sub_intent_mode 的子片段, 意图由父调用指定, 不再识别
            detected_intent = IntentResult(intent=forced_intent, confidence=1.0)
        elif crisis_force_intent:
            detected_intent = IntentResult(intent=IntentType.CRISIS, confidence=1.0)
        elif crisis_followup_active:
            detected_intent = IntentResult(
                intent=IntentType.CRISIS,
                confidence=1.0,
                metadata=crisis_decision.intent_metadata,
            )
        elif current_state_fast_path:
            detected_intent = IntentResult(
                intent=IntentType.CURRENT_STATE,
                confidence=1.0,
                metadata={"fast_path": "current_state_phrase"},
            )
        else:
            context_text = await _fetch_intent_context(
                conversation_id,
                exclude_id=user_message_id,
                exclude_ids=current_turn_ids,
                exclude_content=user_message if not user_message_id else None,
            )
            detected_intent = await detect_intent_unified(user_message, context=context_text)
            if detected_intent.intent != IntentType.NONE:
                logger.info(
                    f"[INTENT-LLM] '{user_message[:30]}' → {detected_intent.intent.value} "
                    f"(labels={detected_intent.metadata.get('llm_labels')})"
                )
            detected_intent = _downgrade_non_explicit_current_schedule_query(
                detected_intent,
                user_message,
                response_diagnostics,
            )

        async def _cancel_fetch_task() -> None:
            """短路 / 异常时调用: cancel + 等待 propagate, 避免 orphan task warning."""
            if fetch_task is None or fetch_task.done():
                return
            fetch_task.cancel()
            try:
                await fetch_task
            except (asyncio.CancelledError, Exception):
                pass
        # crisis 路径下 detected_intent 是手动构造的 CRISIS, metadata={} 没 fragments,
        # 这里 if 分支自然不进; 但显式 `not crisis_force_intent` 防御 future 改动加进 metadata.
        # (crisis 优先级最高, 不该被多意图拆分稀释)
        if forced_intent is None and not crisis_care_turn:
            # spec §3.3 step 3: 多意图 → 待处理子片段列表（主意图片段替换 user_message，其它稍后递归处理）
            fragments = detected_intent.metadata.get("fragments") if detected_intent.metadata else None
            if fragments and len(fragments) > 1:
                fragments = _filter_non_explicit_sub_fragments(
                    fragments,
                    response_diagnostics,
                )
                if len(fragments) <= 1:
                    fragments = None
            if fragments and len(fragments) > 1:
                primary_label = next(
                    (lb for lb, it in LABEL_TO_INTENT.items()
                     if it == detected_intent.intent and lb in fragments),
                    None,
                )
                if primary_label and fragments.get(primary_label):
                    user_message = str(fragments[primary_label]).strip() or user_message
                # spec §3.3 step 3: 子意图按延迟批处理依次进入对应分支. 不再过滤
                # "日常交流" — 之前为防"日常交流" sub 跟其他意图主题撞车 (commit
                # 3d0417d) 引入的 hack 已不需要, 因为根因 (prompt_builder 把
                # schedule_context 注入 §4 主回复, 让 §4 输出 AI 当前活动跟 §3.4.3
                # 撞车) 已修. 现在 §4 不再带 AI 状态主题, 多意图自然分流.
                pending_sub_fragments = {
                    lb: str(txt).strip()
                    for lb, txt in fragments.items()
                    if lb != primary_label and str(txt).strip()
                }
                if pending_sub_fragments:
                    logger.info(
                        f"[INTENT-MULTI] primary={detected_intent.intent.value} "
                        f"sub={list(pending_sub_fragments.keys())}",
                        extra={
                            "event": EVT_INTENT_SPLIT,
                            "intent_primary": detected_intent.intent.name,
                            "sub_intents": list(pending_sub_fragments.keys()),
                            "n_sub": len(pending_sub_fragments),
                        },
                    )

        # 统一短路上下文：6 个意图 handler 共用
        sc_ctx = ShortCircuitCtx(
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_id=user_id,
            agent=agent,
            reply_context=reply_context,
            tracer=tracer,
            save_replies_fn=_save_replies,
            pending_sub_fragments=pending_sub_fragments,
            sub_intent_mode=sub_intent_mode,
            reply_index_offset=reply_index_offset,
            cached_patience=cached_patience,
            # 复用 boundary/preflight 之前算好的同一份 recent_context, 避免二次格式化.
            # handler 把它传进 *_reply prompt 的 {context} 占位符 — 否则短路路径
            # LLM 看不到对话历史, 只能从 AI 当前作息编内容 (生产 bug 2026-05-05:
            # 用户问"你看到什么段子" → AI 编了一个跟自己当下划船活动巧合的段子,
            # 因为 prompt 里 {context} 是 "(无)").
            recent_context=recent_context_text,
            response_diagnostics=response_diagnostics,
            covered_until_user_ts=covered_until_user_ts,
        )
        response_diagnostics["crisis_followup_check_mode"] = (
            crisis_followup_check_mode if crisis_followup_active else None
        )

        # §3.4.6 终结意图
        if detected_intent.intent == IntentType.CONVERSATION_END:
            await _cancel_fetch_task()
            async for evt in handle_conversation_end(user_message, sc_ctx, _intent_llm_reply):
                yield evt
            if workspace_id and agent_id and not sub_intent_mode:
                _fire_background(start_or_restart_proactive_session(
                    workspace_id=workspace_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    reason="farewell",
                ))
            return

        # §3.4.4 道歉承诺热路径
        if detected_intent.intent == IntentType.APOLOGY_PROMISE:
            handled, events = await handle_apology_promise(user_message, sc_ctx)
            if handled and events is not None:
                await _cancel_fetch_task()
                async for evt in events:
                    yield evt
                return

        # §5 step 1-2 删除意图：找候选 → 请求确认
        elif detected_intent.intent == IntentType.DELETION:
            handled, events = await handle_deletion(user_message, sc_ctx)
            if handled and events is not None:
                await _cancel_fetch_task()
                async for evt in events:
                    yield evt
                return

        # NOTE: SCHEDULE_ADJUST/SCHEDULE_QUERY/CURRENT_STATE 在 parallel data fetch 之后处理

        # ── P0 危机安全网 短路 (排在主路径 fetch_parallel_context 之前) ─────────
        # 关键: 必须在 await fetch_parallel_context 之前 dispatch, 否则等完整
        # fetch (含 relevance/user emotion 等无关 LLM) 才到 handle_crisis.
        # 实测 trace 2026-05-07 16:57: 走完整 fetch 总延迟 18s 中 4s 是浪费 LLM.
        # 跳过完整 fetch, 只 await 我们提前 kick off 的 crisis_memory_task +
        # crisis_portrait_task (handle_crisis 仅需这俩).
        if detected_intent.intent == IntentType.CRISIS:
            await _cancel_fetch_task()
            if crisis_memory_task is None:
                from app.services.memory.retrieval.safety import (
                    retrieve_crisis_followup_memories,
                    retrieve_crisis_memories,
                )
                if detected_intent.metadata.get("followup"):
                    crisis_memory_task = asyncio.create_task(
                        retrieve_crisis_followup_memories(
                            user_message,
                            user_id,
                            recent_context=recent_crisis_context,
                            workspace_id=workspace_id,
                        )
                    )
                else:
                    crisis_memory_task = asyncio.create_task(
                        retrieve_crisis_memories(
                            user_message, user_id, workspace_id=workspace_id,
                        )
                    )
            if crisis_portrait_task is None and agent_id:
                from app.services.portrait import get_latest_portrait
                crisis_portrait_task = asyncio.create_task(
                    get_latest_portrait(user_id, agent_id)
                )
            # 等轻量 fetch (memory + portrait), 跳 user emotion/relevance/topic/schedule LLM
            crisis_classified: list = []
            crisis_portrait: Any = None
            if crisis_memory_task is not None:
                try:
                    retrieval_result = await crisis_memory_task
                    if isinstance(retrieval_result, dict):
                        crisis_classified = retrieval_result.get("memories") or []
                    elif isinstance(retrieval_result, list):
                        crisis_classified = retrieval_result
                except Exception as e:
                    logger.warning(f"Crisis memory fetch failed: {e}")
            if crisis_portrait_task is not None:
                try:
                    crisis_portrait = await crisis_portrait_task
                except Exception as e:
                    logger.warning(f"Crisis portrait fetch failed: {e}")
            crisis_accessed_ids = [
                getattr(m, "id", "") for m in crisis_classified if getattr(m, "id", "")
            ]
            if crisis_accessed_ids:
                _fire_background(
                    log_memory_access(user_id, crisis_accessed_ids, workspace_id=workspace_id)
                )
            if detected_intent.metadata.get("followup"):
                async for evt in handle_crisis_followup(
                    user_message, sc_ctx,
                    classified_memories=crisis_classified,
                    portrait=crisis_portrait,
                    safety_check_mode=detected_intent.metadata.get(
                        "safety_check_mode", "none",
                    ),
                ):
                    yield evt
            else:
                async for evt in handle_crisis(
                    user_message, sc_ctx,
                    classified_memories=crisis_classified,
                    portrait=crisis_portrait,
                ):
                    yield evt
            # finally 兜底 fire post_process (用 None user_emotion, _bg_user_emotion
            # gracefully 跳过). memory pipeline 仍跑 — crisis 消息应该被记忆.
            return

        # 记录 LLM 数据拉取时刻能看到的最新 user 消息时间, 用于 scheduler dedup gate.
        # 若用户连发多条非碎片, 第一条 LLM 调用的 history 已经隐式包含后续所有 user
        # 消息 → reply 实际覆盖了它们; 写到 reply metadata 后, scheduler 处理后续
        # payload 时凭此跳过, 避免重复回复 (见 jobs/scheduler.py dedup gate).
        # --- Topic tracking (Redis, no LLM) ---
        topic_info = await push_topic(conversation_id, user_message)
        topic_context = format_topic_context(topic_info) if topic_info else None

        # --- Pre-compute personality (MBTI) for downstream timing/emotion calls ---
        mbti = get_mbti(agent)

        # spec §3.1+§3.2 step 2-3: 拉取记忆/情绪/画像/作息. 两条路径:
        # - auto-intent: 已在 intent 之前用 create_task 启动 fetch_task, 这里 await
        #   它的结果, 然后单独跑 L3 awakening (intent + relevance 都已知).
        # - sub_intent_mode (forced_intent): 同步调 fetch_parallel_context 包含 L3,
        #   行为跟之前一致.
        # L3 awakening 跟下游 (短路 handlers / contradiction / prompt build) 并行.
        # L3 仅依赖 fetched.memory_relevance + detected_intent (现已都有), prompt
        # build 不依赖 L3 — L3 结果只塞 long_term_memories 段, await 时机推到
        # build_system_prompt 之前. 短路 handlers 早退时通过 _cancel_l3_task 取消.
        l3_task: asyncio.Task[tuple[list[str], str]] | None = None
        if fetch_task is not None:
            fetched = await fetch_task
            l3_task = asyncio.create_task(maybe_awaken_l3(
                user_message, user_id, workspace_id,
                detected_intent, fetched.memory_relevance,
                _l3_trigger_analyze,
                enhanced_query=fetched.enhanced_query,
                l1_l2_count=len(fetched.classified_memories or []),
                recent_context=format_recent_context(
                    messages_dicts,
                    exclude_message_ids=current_turn_ids,
                ),
            ))
        elif current_state_fast_path:
            schedule = await get_cached_schedule(agent_id) if agent_id else None
            ai_status = get_current_status(schedule) if schedule else None
            schedule_context = format_schedule_context(ai_status) if ai_status else None
            fetched = FetchedContext(
                memory_relevance="weak",
                classified_memories=[],
                memory_strings=[],
                schedule=schedule,
                ai_status=ai_status,
                schedule_context=schedule_context,
            )
            response_diagnostics["memory_retrieval_skipped_reason"] = "current_state_fast_path"
        else:
            parsed_times = (
                parse_time_expressions(user_message)
                if not skip_time_memory_lookup and has_explicit_time(user_message) else []
            )
            fetched = await fetch_parallel_context(
                user_id=user_id, agent_id=agent_id, workspace_id=workspace_id,
                user_message=user_message,
                messages_dicts=messages_dicts, parsed_times=parsed_times,
                detected_intent=detected_intent,
                l3_trigger_classify_fn=_l3_trigger_analyze,
            )
        memory_relevance = fetched.memory_relevance
        classified_memories = fetched.classified_memories
        prompt_user_emotion = fetched.user_emotion
        portrait = fetched.portrait
        schedule = fetched.schedule
        topic_intimacy = fetched.topic_intimacy
        time_memories = fetched.time_memories
        l3_memories = fetched.l3_memories
        ai_status = fetched.ai_status
        schedule_context = fetched.schedule_context
        response_diagnostics.update({
            "memory_relevance": memory_relevance,
            "memory_retrieval_skipped_reason": (
                response_diagnostics.get("memory_retrieval_skipped_reason")
                or ("weak_relevance" if memory_relevance == "weak" else None)
            ),
        })

        async def _cancel_l3_task() -> None:
            """短路 / 异常时调用: cancel L3 task + 等待 propagate, 防 orphan task warning."""
            if l3_task is None or l3_task.done():
                return
            l3_task.cancel()
            try:
                await l3_task
            except (asyncio.CancelledError, Exception):
                pass

        # NOTE: CRISIS dispatch 已在 fetch_parallel_context 之前 (line ~675), 跳过完整
        # fetch 节省 4s 无关 LLM 时间. 这里不再有 crisis 分支.

        delay_context = None
        if reply_context:
            received_status = reply_context.get("received_status") or {}
            received_activity = str(received_status.get("activity", "")).strip() or "处理自己的事"
            received_status_label = str(received_status.get("status", "idle"))
            received_at = str(reply_context.get("received_at", ""))
            elapsed = actual_delay_seconds(reply_context)
            # spec §6.5: ≥1min 时会单独推送"延迟解释回复"，主回复不再重复注入解释
            if elapsed is not None and elapsed < 60:
                rounded_delay = max(1, round(elapsed))
                delay_reason_text = explain_delay_reason(
                    str(reply_context.get("delay_reason", "")),
                    activity=received_activity,
                    status=received_status_label,
                )
                delay_context = (
                    f"你在 {received_at} 收到用户消息时，正在{received_activity}"
                    f"（状态：{received_status_label}）。\n"
                    f"现在距离收到消息已经过去约 {rounded_delay} 秒。\n"
                    f"{delay_reason_text}\n"
                    "只有在确实需要时，才用半句自然带过刚才在忙什么；"
                    "优先回应用户当下情绪或关系信号，解释不要压过聊天本身。"
                )

        relational_context = detect_relational_context(user_message, prompt_user_emotion)

        # --- Time context for prompt (PRD §9.2) ---
        time_context = build_time_context()

        # --- Intimacy stage for prompt (PRD §4.6.2.1) ---
        intimacy_stage = get_relationship_stage(topic_intimacy)

        # ai_status / schedule_context 已由 fetch_parallel_context 在上面计算并赋值

        # §3.4.2 作息调整
        if detected_intent.intent == IntentType.SCHEDULE_ADJUST:
            handled, events = await handle_schedule_adjust(
                user_message, sc_ctx,
                schedule=schedule, ai_status=ai_status,
                topic_intimacy=topic_intimacy, mbti=mbti,
            )
            if handled and events is not None:
                async for evt in events:
                    yield evt
                await _cancel_l3_task()
                return

        # 工程扩展 (Phase 3): 记录请求 — 用户请求 AI 记一件事 / 设提醒.
        # 跟"计划查询"分离: 体感需"好嘞我帮你记上了"而非"你明天要做X". 后台
        # _bg_memory_pipeline 仍跑, 完成实际 memory + timetrigger 创建 (Phase 4.1).
        if detected_intent.intent == IntentType.RECORD_REQUEST:
            handled, events = await handle_record_request(user_message, sc_ctx)
            if handled and events is not None:
                async for evt in events:
                    yield evt
                await _cancel_l3_task()
                return

        detected_intent = _route_current_schedule_query_to_current_state(
            detected_intent,
            user_message,
            response_diagnostics,
        )

        # §3.4.1 计划查询
        if detected_intent.intent == IntentType.SCHEDULE_QUERY:
            query_type = detected_intent.metadata.get("query_type", "current")
            handled, events, schedule_ctx_for_prompt = await handle_schedule_query(
                user_message, sc_ctx,
                schedule=schedule, ai_status=ai_status,
                portrait=portrait, user_emotion=prompt_user_emotion,
                query_type=query_type,
            )
            if schedule_ctx_for_prompt is not None:
                schedule_context = schedule_ctx_for_prompt  # 供下方 rich prompt 复用
            if handled and events is not None:
                async for evt in events:
                    yield evt
                await _cancel_l3_task()
                return

        detected_intent = _downgrade_non_explicit_current_state(
            detected_intent,
            user_message,
            response_diagnostics,
        )

        # §3.4.3 询问当前状态
        if detected_intent.intent == IntentType.CURRENT_STATE:
            handled, events = await handle_current_state(
                user_message, sc_ctx,
                ai_status=ai_status, schedule_context=schedule_context,
                portrait=portrait, user_emotion=prompt_user_emotion,
            )
            if handled and events is not None:
                async for evt in events:
                    yield evt
                await _cancel_l3_task()
                return

        # 5B.4: Get patience prompt instruction (reuse value from check_boundary)
        patience_instruction = get_patience_prompt_instruction(cached_patience)

        # Spec §4 step 1-2: detect NEW contradictions (resolution already handled
        # at the top of the function via pending state check)
        contradiction_inquiry: str | None = None
        if memory_relevance in ("strong", "medium"):
            try:
                conflict = await detect_l1_contradiction(user_message, user_id, workspace_id=workspace_id)
                if conflict:
                    inquiry = await generate_contradiction_inquiry(conflict, agent_name=agent.name if agent else "AI")
                    contradiction_inquiry = inquiry
                    await save_pending_contradiction(conversation_id, conflict)
                    logger.info(
                        f"L1 contradiction detected: {conflict.get('conflict_description', '')}",
                        extra={
                            "event": EVT_MEMORY_CONTRADICTION,
                            "conflict_summary": (conflict.get("conflict_description") or "")[:80],
                        },
                    )
            except Exception as e:
                logger.warning(f"Contradiction detection failed: {e}")

        # --- spec §5.5: n = random.randint(1, 3) 均匀分布 ---
        if relational_context:
            reply_count = 1
        elif contradiction_inquiry:
            reply_count = 1  # contradiction inquiry is a single focused question
        else:
            reply_count = random.randint(1, MAX_REPLY_COUNT)
        max_reply_count = MAX_REPLY_COUNT
        max_total = MAX_TOTAL_CHARS

        # await L3 task (与 short-circuits + contradiction + 上面的 sync 计算并行跑了).
        # 失败 fallback 到 fetched 里的默认 (空列表 / "无").
        if l3_task is not None:
            try:
                l3_memories, l3_trigger_label = await l3_task
                fetched.l3_memories = l3_memories
                fetched.l3_trigger_label = l3_trigger_label
            except (asyncio.CancelledError, Exception) as e:
                logger.warning(f"L3 awakening failed: {e}")

        response_diagnostics.update({
            "memory_relevance": memory_relevance,
            "memory_retrieval_skipped_reason": (
                response_diagnostics.get("memory_retrieval_skipped_reason")
                or ("weak_relevance" if memory_relevance == "weak" else None)
            ),
            "empty_prompt_sections_removed_count": None,
        })

        async def _build_main_chat_messages() -> list[dict]:
            # Build prompt only if generate_reply reaches the main LLM fallback path.
            # Tier replies and contradiction inquiries short-circuit before this.
            # Phase 6: 删 relational_context / graph_context 入参 — 实证冗余/幻觉源
            started = perf_counter()
            prompt_diagnostics: dict[str, Any] = {}
            system_prompt = await build_system_prompt(
                agent=agent,
                memories=classified_memories,
                delay_context=delay_context,
                portrait=portrait,
                topic_context=topic_context,
                user_emotion=prompt_user_emotion,
                # schedule_context 故意不传给 §4 主回复 prompt: spec §4 不要求 AI 当前活动,
                # 只有 §3.4.3 询问当前状态走 short-circuit 时才需要 (handle_current_state
                # 有自己的参数路径). 详见 prompt_builder.py 注释 + commit 0038a13 上下文.
                # ai_status 单独传: 仅供"AI 自洽性约束"段使用 (告知状态 + 禁止主动展开),
                # 防 ≥1min 延迟主回复路径下 LLM 编造跟实际状态矛盾的活动.
                ai_status=ai_status,
                patience_instruction=patience_instruction,
                reply_count=reply_count,
                reply_total=max_total,
                intimacy_stage=intimacy_stage,
                time_context=time_context,
                time_memories=time_memories or None,
                l3_memories=l3_memories or None,
                memory_relevance=memory_relevance,
                diagnostics=prompt_diagnostics,
                canary_user_id=user_id,
            )
            response_diagnostics["main_prompt_built"] = True
            response_diagnostics["main_prompt_build_ms"] = round(
                (perf_counter() - started) * 1000,
                3,
            )
            response_diagnostics.update(prompt_diagnostics)
            return build_chat_messages(system_prompt, messages_dicts)

        # Log memory access for L2 frequency tracking (background, non-blocking)
        accessed_ids: list[str] = []
        if classified_memories:
            accessed_ids.extend(getattr(m, "id", "") for m in classified_memories if getattr(m, "id", ""))
        if accessed_ids:
            _fire_background(log_memory_access(user_id, accessed_ids, workspace_id=workspace_id))

        # spec §6 异步回复机制只规定延迟分布, 没"对方正在输入"占位事件; 早期作为
        # UX 装饰加的, 关闭后前端直接看到流式 token 即可, 无视觉退化.
        # settings.reply_delay_enabled=False (默认) → 跳过 sleep 即时回复.
        from app.config import settings as _settings
        if _settings.reply_delay_enabled:
            reply_delay = calculate_reply_delay(len(user_message), mbti=mbti)
            queued_delay = float((reply_context or {}).get("delay_seconds", 0.0) or 0.0)
            conceptual_delay = max(reply_delay, queued_delay)
            if delivered_from_queue:
                actual_sleep = min(reply_delay, 1.5)
            else:
                actual_sleep = min(conceptual_delay, 2.0)
                if conceptual_delay > 5.0:
                    yield {"event": "delay", "data": json.dumps({"duration": conceptual_delay})}
            if actual_sleep > 0:
                await asyncio.sleep(actual_sleep)

        replies, raw_response, reply_is_fallback, reply_emotion_pre = await _generate_reply(
            contradiction_inquiry=contradiction_inquiry,
            detected_intent=detected_intent,
            memory_relevance=memory_relevance,
            relational_context=relational_context,
            schedule_context=schedule_context,
            delay_context=delay_context,
            l3_memories=l3_memories,
            classified_memories=classified_memories or [],
            messages_dicts=messages_dicts,
            portrait=portrait,
            prompt_user_emotion=prompt_user_emotion,
            user_message=user_message,
            agent=agent,
            chat_messages_factory=_build_main_chat_messages,
            reply_count=reply_count,
            max_reply_count=max_reply_count,
            max_total=max_total,
            tier_fns={
                "weak": _memory_weak_reply,
                "medium": _memory_medium_reply,
                "strong": _memory_strong_reply,
                "l3": _memory_l3_reply,
            },
            # split_llm_fn 已删 — 主 LLM 直接按 || 输出, 不再二次拆分
            # LLM-split 分支用 truncate_fn: 先 _clean_reply_part 把单条内残留 \n 折成空格,
            # 再走 sentence-truncate, 防止 LLM 给的某条单片里嵌空白行/换行被前端
            # pre-wrap 渲染成断行.
            truncate_fn=lambda text, max_len: truncate_at_sentence(_clean_reply_part(text), max_len),
            pipe_fallback_fn=split_and_validate_replies,
            # P0-3: 主 LLM 路径下 split + emotion 在 generate_reply 内并行,
            # 直接拿 reply_emotion_pre 返回, 无需再串行多调一次. tier / contradiction
            # 路径返 None, fallback 兜底见下方.
            reply_emotion_fn=_ai_reply_emotion,
            diagnostics=response_diagnostics,
        )

        # spec §5 step 1：AI 语句情绪识别（基于回复文本）
        # 主 LLM 路径已在 generate_reply 内并行算好, 直接复用; tier/contradiction 路径
        # reply_emotion_pre=None 时兜底再调一次 (这两条路径都是单 LLM, 增量小).
        # full_response 必须无条件计算 — 下方 background post_process 总是引用它.
        full_response = " ".join(replies)
        if reply_emotion_pre is not None:
            reply_emotion = reply_emotion_pre
        else:
            reply_emotion = await _ai_reply_emotion(full_response)
        if reply_emotion.get("emotion"):
            logger.info(
                f"[REPLY-EMO] emotion={reply_emotion['emotion']} "
                f"intensity={reply_emotion.get('intensity', 0)}",
                extra={
                    "event": EVT_REPLY_EMOTION,
                    "ai_emotion": reply_emotion["emotion"],
                    "intensity": reply_emotion.get("intensity", 0),
                    "reply_text_len": len(full_response),
                },
            )

        # spec §5/§6.4-§6.5: emoji/sticker + 延迟解释 + 推送
        emitted_replies: list[dict] = []
        async for evt in _emit_replies(
            replies,
            reply_context=reply_context,
            reply_index_offset=reply_index_offset,
            sub_intent_mode=sub_intent_mode,
            agent=agent,
            user_message=user_message,
            delay_reply_fn=_delay_explanation_reply,
            fallback_fn=_intent_llm_reply,
            emitted_replies=emitted_replies,
            reply_emotion=reply_emotion,
            reply_is_fallback=reply_is_fallback,
        ):
            yield evt

        # Persist replies immediately; trace links become clickable only after public share completes.
        # 把 covered_until_user_ts 注入首条 reply, save_replies 会写入 metadata.
        if emitted_replies and covered_until_user_ts is not None:
            first = emitted_replies[0]
            if isinstance(first, dict):
                first.setdefault("covered_until_user_ts", covered_until_user_ts.isoformat())
        if emitted_replies:
            first = emitted_replies[0]
            if isinstance(first, dict):
                first.setdefault("response_diagnostics", response_diagnostics)
            else:
                emitted_replies[0] = {
                    "text": str(first),
                    "response_diagnostics": response_diagnostics,
                }
            prompt_render_traces = snapshot_prompt_render_traces()
            if prompt_render_traces:
                first = emitted_replies[0]
                if isinstance(first, dict):
                    first.setdefault("prompt_render_traces", prompt_render_traces)
                else:
                    emitted_replies[0] = {
                        "text": str(first),
                        "prompt_render_traces": prompt_render_traces,
                    }
            from app.services.memory.retrieval.trace import (
                build_retrieval_quality_analysis,
                snapshot_retrieval_traces,
            )
            retrieval_traces = snapshot_retrieval_traces()
            if retrieval_traces:
                retrieval_analysis = build_retrieval_quality_analysis(
                    retrieval_traces,
                    assistant_reply=full_response,
                    user_message=user_message,
                )
                first = emitted_replies[0]
                if isinstance(first, dict):
                    first.setdefault("memory_retrievals", retrieval_traces)
                    if retrieval_analysis:
                        first.setdefault("memory_retrieval_analysis", retrieval_analysis)
                else:
                    emitted_replies[0] = {
                        "text": str(first),
                        "memory_retrievals": retrieval_traces,
                    }
                    if retrieval_analysis:
                        emitted_replies[0]["memory_retrieval_analysis"] = retrieval_analysis
        first_assistant_message_id = await _save_replies(
            conversation_id,
            emitted_replies,
            trace_id=tracer.trace_id if tracer.is_active else None,
        )
        # spec §6.4-§6.5 已经 emit; 这里记 final 信号 — Axiom 用 event=reply.emitted
        # 切分一次"完成回复"维度 (跟 reply.llm_main / reply.split 区别: 那两条是
        # 中间步骤, 这条是用户实际收到的最终结果, 包含 emoji/sticker/拆分后)
        logger.info(
            f"[REPLY-EMIT] n={len(emitted_replies)} sub_intent_mode={sub_intent_mode}",
            extra={
                "event": EVT_REPLY_EMITTED,
                "n_replies": len(emitted_replies),
                "sub_intent_mode": sub_intent_mode,
                "is_fallback": reply_is_fallback,
            },
        )

        if sub_intent_mode:
            # 父调用负责后台任务、save_last_reply_timestamp、done、trace 关闭
            return

        # Update conversation title if first exchange (non-blocking)
        if len(recent_messages) <= 1:
            title = user_message[:50] + ("..." if len(user_message) > 50 else "")
            _fire_background(db.conversation.update(
                where={"id": conversation_id},
                data={"title": title},
            ))

        # --- BACKGROUND: fire-and-forget post-processing ---
        _fire_background(_background_post_process(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            user_message=user_message,
            user_message_id=user_message_id,
            full_response=full_response,
            messages_dicts=messages_dicts,
            user_emotion=prompt_user_emotion,
            skip_ai_memory=False,
        ))
        post_process_fired = True

        # spec §3.3 step 3: 主意图回复完成后，依次处理拆分出的子意图片段
        if pending_sub_fragments:
            start_idx = reply_index_offset + len(emitted_replies)
            async for evt in _process_sub_intents(
                pending_sub_fragments, conversation_id, agent, user_id,
                reply_context, start_index=start_idx,
                parent_patience=cached_patience,
                parent_trace_id=tracer.trace_id,
            ):
                yield evt

        await save_last_reply_timestamp(agent_id, user_id)
        if workspace_id and agent_id:
            _fire_background(start_or_restart_proactive_session(
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id=agent_id,
                reason="conversation_end",
            ))
        done_data: dict = {"message_id": "complete"}
        if first_assistant_message_id and tracer.trace_id and tracer.is_active:
            done_data["assistant_message_id"] = first_assistant_message_id
        yield {"event": "done", "data": json.dumps(done_data)}

        # End trace and share publicly in background (updates DB with public URL)
        tracer.close()
    finally:
        # Flush LLM usage 累计到 llm_usage 表. sub_intent_mode 没起 session,
        # token 是 None, 走里面的 short-circuit.
        if usage_token is not None:
            from app.services.llm.usage_repo import write_usage_row
            summary = usage_tracker.flush_session(usage_token)
            if summary:
                try:
                    await write_usage_row(
                        summary=summary,
                        conversation_id=conversation_id,
                        agent_id=agent_id,
                        user_id=user_id,
                        trace_id=getattr(tracer, "trace_id", None),
                    )
                except Exception:
                    logger.warning("[llm-usage] write_usage_row failed", exc_info=True)
        # 还原 ContextVar (防御共享 worker pool 跨 agent leak)
        if retrieval_trace_token is not None:
            from app.services.memory.retrieval.trace import reset_retrieval_trace
            reset_retrieval_trace(retrieval_trace_token)
        if prompt_trace_token is not None:
            reset_prompt_render_trace(prompt_trace_token)
        reset_prompt_runtime_context(_prompt_ctx_token)
        reset_current_agent(_agent_ctx_token)

        # spec §2.1/§2.2 兜底: 短路意图早 return 跳过主路径末尾的 post_process fire,
        # 这里补 fire 让 memory/user-emotion/trait/recovery 后台任务跑全. boundary 短路
        # 已自行处理 (apology fires _bg_memory_pipeline, blocked skips per CLAUDE.md §3.3
        # — 注: blocked 仍丢 user-emotion/trait/recovery, 是已知 spec 偏离, 见 CLAUDE.md) → 跳过.
        # sub_intent_mode 由父调用统一处理后台任务, 子片段不重复 fire.
        # 用 `is not None` 正向判别 last_short_circuit_reply: 短路 handler 完成才 set
        # ctx field, 中途异常 / 未到短路点 → 字段仍为 None → 不会 phantom fire 空 reply.
        sc_reply: str | None = None
        if sc_ctx is not None and sc_ctx.last_short_circuit_reply is not None:
            sc_reply = sc_ctx.last_short_circuit_reply
        elif preflight_ctx is not None and preflight_ctx.last_short_circuit_reply is not None:
            sc_reply = preflight_ctx.last_short_circuit_reply

        boundary_handled = boundary_ctx is not None and boundary_ctx.stopped
        if (sc_reply is not None
                and not post_process_fired
                and not sub_intent_mode
                and not boundary_handled):
            _fire_background(_background_post_process(
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                user_message=user_message,
                user_message_id=user_message_id,
                full_response=sc_reply,
                messages_dicts=messages_dicts,
                user_emotion=prompt_user_emotion,
                skip_ai_memory=(
                    sc_ctx is not None
                    and sc_ctx.last_short_circuit_kind in {"schedule_query", "current_state"}
                ),
            ))
