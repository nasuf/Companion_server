"""Chat service — optimized for low latency.

Hot path (user-facing, ~2s):
  save msg → parallel(vector retrieval + load cached context) → prompt → stream LLM

Background (fire-and-forget, after response):
  user emotion metadata + memory pipeline + trait/patience updates
"""

from app.services.temporal_query import resolve_temporal_query
from app.services.chat.temporal_context import prepare_reply_time, dated_dialogue, build_turn_clock, scope_event_memories, render_event_scope, dated_active_dialogue
from app.services.chat.followup_policy import guard_optional_questions
import asyncio
import functools
import json
import logging
from datetime import datetime
from time import perf_counter
from collections.abc import AsyncGenerator
from typing import Any

from prisma import Json

from app.db import db
from app.observability.events import (
    EVT_FILLER_EMOJI,
    EVT_INTENT_SPLIT,
    EVT_MEMORY_CONTRADICTION,
    EVT_REPLY_EMITTED,
    EVT_REPLY_EMOTION,
)
from app.services.chat import context_policy as _context_policy
from app.services.llm.models import get_chat_model, convert_messages, invoke_text
from app.services.llm.web_search_gate import extract_discussed_titles
from app.services.chat.prompt_builder import (
    build_system_prompt,
    build_chat_messages,
    compute_reengagement_gap_seconds,
)
from app.services.chat_media.prompt import render_message_content_for_prompt
from app.services.prompting.store import (
    get_prompt_text,
    get_prompt_text_or_default,
)
from app.services.prompting.utils import safe_format
from app.services.prompting.trace_components import (
    prompt_hash,
    record_prompt_render,
    reset_prompt_render_trace,
    snapshot_prompt_render_traces,
    start_prompt_render_trace,
)
from app.services.prompts.system_prompts import (
    MAX_REPLY_COUNT, MAX_TOTAL_CHARS,
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
from app.services.chat.location_context import (
    asks_about_user_location,
    load_location_context,
)
from app.services.chat.memory_usage import submit_memory_usage
from app.services.topic import push_topic
from app.services.schedule_domain.time_service import build_time_context
from app.services.schedule_domain.time_parser import parse_time_expressions, has_explicit_time
from app.services.schedule_domain.schedule import (
    format_schedule_context,
    get_cached_schedule,
    get_current_status,
)
from app.services.interaction.boundary import (
    PATIENCE_MAX,
    PATIENCE_NORMAL_MIN,
    get_patience_prompt_instruction,
)
from app.services.relationship.intimacy import get_relationship_stage
from app.services.chat.intent_dispatcher import (
    detect_current_state_fast_path,
    detect_intent_unified,
    IntentType,
    IntentResult,
    LABEL_TO_INTENT,
)
from app.services.chat.multi_intent import (
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
from app.services.chat.expression_learner import sample_expression_habits
from app.services.chat.session_recap import RECAP_GAP_SECONDS, get_or_build_session_recap
from app.services.chat.reply_count_state import (
    load_last_reply_count,
    pick_reply_count_target,
    save_last_reply_count,
)
from app.services.relationship.ai_mood import format_ai_mood_text, load_ai_mood
from app.services.relationship.relation_meta import (
    format_relation_meta_line,
    get_relation_meta,
)
from app.services.chat.filler_reply import build_filler_emoji_reply
# R1 拆分: 消息上下文工具函数移至 message_utils; 此处 re-export 保持
# 既有导入路径兼容 (tests / multi_intent 等 from orchestrator import).
# R3 拆分: 意图路由修正移至 intent_routes; re-export 兼容既有导入.
from app.services.chat.intent_routes import (
    _downgrade_non_explicit_current_schedule_query,
    _downgrade_non_explicit_current_state,
    _downgrade_non_explicit_schedule_adjust,
    _filter_non_explicit_sub_fragments,
    _route_current_schedule_query_to_current_state,
)
# R2 拆分: 回复切分纯函数移至 reply_formatting; re-export 兼容既有导入.
from app.services.chat.reply_formatting import (
    _clean_reply_part,
    split_and_validate_replies,
    split_short_circuit_reply,
    truncate_at_sentence,
)
# R4 拆分: 关系情绪线索检测移至 relationship/relation_context; re-export 兼容.
from app.services.relationship.relation_context import detect_relational_context
from app.services.chat.message_utils import (
    _achievement_turn_id,
    _current_turn_message_ids,
    _ensure_current_user_message,
    _max_user_created_at,
    _parse_message_created_at,
    _previous_assistant_message,
    collapse_turn_fragments,
    is_offering_received_notice,
)
from app.services.chat.preflight import (
    PreflightCtx,
    discard_pending_states_for_crisis,
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
from app.services.chat.repetition_guard import find_repeated_ai_phrases
from app.services.chat.tracing import create_tracer
from app.services.mbti import build_personality_brief, get_mbti
from app.services.chat.turn_lifecycle import finish_assistant_turn
from app.services.interaction.reply_context import actual_delay_seconds
from app.services.chat.topic_continuation import (
    await_topic_continuation,
    build_topic_continuation,
    record_topic_continuation,
)
from app.services.proactive.state import ARM_REASON_CRISIS, ARM_REASON_REPLY, ARM_REASON_SYSTEM
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
    proactive_reason: str | None = "short_circuit",
    turn_user_message_ids: list[str] | None = None,
    workspace_id: str | None = None,
) -> list[dict]:
    """Orchestrator-side adapter that injects `_save_replies`."""
    return await _short_circuit_reply_impl(
        reply, conversation_id, agent_id, user_id, _save_replies,
        sub_intent_mode=sub_intent_mode,
        reply_index_offset=reply_index_offset,
        include_done=include_done,
        extra_metadata=extra_metadata,
        trace_id=trace_id,
        proactive_reason=proactive_reason,
        turn_user_message_ids=turn_user_message_ids,
        workspace_id=workspace_id,
    )


async def _boundary_short_circuit_reply(*args, **kwargs) -> list[dict]:
    """边界系统的回复 (攻击 / 低耐心 / 拉黑 / 道歉) 之后不 arm 主动交流:
    AI 正在生气时不追人, 与改造前行为一致。"""
    return await _short_circuit_reply(*args, proactive_reason=None, **kwargs)


async def _system_short_circuit_reply(*args, **kwargs) -> list[dict]:
    """pending 流程 (矛盾追问的回答 / 删除确认 / 撤回) 的回复: 照常 arm, 但不做 B 追问."""
    return await _short_circuit_reply(*args, proactive_reason=ARM_REASON_SYSTEM, **kwargs)


def _turn_started_at(messages: list[dict], turn_ids: set[str]) -> datetime | None:
    """本轮第一条用户消息的落库时刻 (聚合的多条碎片取最早一条)."""
    times = [
        ts for ts in (
            _parse_message_created_at(m.get("createdAt"))
            for m in messages
            if m.get("role") == "user" and m.get("id") in turn_ids
        )
        if ts is not None
    ]
    return min(times, default=None)


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
    # 结构性包装: instruction 是本次 LLM 调用的全部任务指令 (道别/延迟解释兜底),
    # 停用包装模板时退回代码默认, 否则指令整体丢失、LLM 退化成普通闲聊回复.
    appendix_tpl = await get_prompt_text_or_default("chat.special_instruction_appendix")
    appendix_start = len(prompt)
    # D3: SafeDict 渲染 — admin 编辑模板写入字面大括号 (如 JSON 示例) 或误删
    # 占位符时不 KeyError 炸链路, 未知占位符渲染为 "(无)".
    appendix = safe_format(appendix_tpl, {"instruction": instruction})
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
    # Route through invoke_text (not raw model.ainvoke) so this call gets the
    # resilience layer (timeout / retry / circuit-breaker / Ollama fallback)
    # AND lands in usage_tracker — a raw ainvoke silently bypassed both.
    content = await invoke_text(
        get_chat_model(),
        convert_messages([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]),
    )
    return (content or "").strip().split("||")[0][:60]


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
            previous_assistant_reply=assistant_reply or getattr(previous, "content", None) or "",
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


# 模板上下文轮数和主回复历史长度都由 chat.context_policy 控制 (后台可调).

# 历史消息取数上限的下限. 真正决定注入多少的是 build_chat_messages 的 token 预算
# (默认 4000, 后台可调); 这里只是给 DB 查询一个安全上界, 预算调大时主回复按
# context_policy.history_fetch_limit 放大取数.
#
# 曾经是 30, 于是预算从未生效过: 生产实测 30 行 ≈ 679 token, 只用掉预算的 17%.
# 而 30 是**消息行数**不是对话轮数 —— AI 一次回复拆成多个气泡各占一行, 全库
# user:assistant = 1:2.56, 所以 30 行只剩约 10 轮上下文. 生产事故 2026-07-22:
# 用户和 AI 聊了十几分钟 MBTI, 52 条消息之后再问, AI 答"我之前没研究过MBTI",
# 继而反问用户"会不会是你记错啦" —— 那段讨论正好落在 30 行窗口之外.
#
# 200 行按实测均值 ≈ 4400 token, 略高于预算, 保证是预算而不是行数在收口.
_HISTORY_FETCH_LIMIT = _context_policy.HISTORY_FETCH_FLOOR


async def _fetch_intent_context(
    conversation_id: str,
    *,
    exclude_id: str | None = None,
    exclude_ids: set[str] | None = None,
    exclude_content: str | None = None,
) -> str:
    """拉最近几个用户回合拼成意图识别 prompt 的上下文段落。

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
            take=_HISTORY_FETCH_LIMIT,
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
    kept: list[dict] = []
    content_fallback_consumed = False
    for row in rows:
        content = (getattr(row, "content", "") or "").strip()
        if not content:
            continue
        role = "assistant" if getattr(row, "role", "") == "assistant" else "user"
        if getattr(row, "id", None) in excluded_ids:
            continue
        if (
            not exclude_id
            and exclude_content
            and not content_fallback_consumed
            and role == "user"
            and content == exclude_content
        ):
            content_fallback_consumed = True
            continue
        kept.append({"id": getattr(row, "id", None), "role": role, "content": content,
                     "createdAt": getattr(row, "createdAt", None), "metadata": getattr(row, "metadata", None)})

    # desc 顺序 → 反转为时间顺序
    kept.reverse()
    return format_recent_context(kept, purpose="intent") if kept else ""



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
    achievement_turn_final: bool = True,
    memory_usage_event_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Run the sole chat graph and close its owned tasks on consumer exit."""
    from app.services.chat.graph_executor import stream_graph_response

    stream = stream_graph_response(
        conversation_id, user_message, agent, user_id, reply_context,
        save_user_message=save_user_message, user_message_id=user_message_id,
        delivered_from_queue=delivered_from_queue, sub_intent_mode=sub_intent_mode,
        forced_intent=forced_intent, reply_index_offset=reply_index_offset,
        parent_patience=parent_patience, parent_trace_id=parent_trace_id,
        achievement_turn_final=achievement_turn_final,
        memory_usage_event_id=memory_usage_event_id,
    )
    try:
        async for event in stream:
            yield event
    finally:
        await stream.aclose()
