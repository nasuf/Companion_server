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
from app.services.llm.models import get_chat_model, convert_messages, invoke_text
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
from app.services.memory.retrieval.access_log import log_memory_access
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
from app.services.chat.session_recap import get_or_build_session_recap
from app.services.chat.reply_count_state import (
    load_last_reply_count,
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
    _filter_non_explicit_sub_fragments,
    _route_current_schedule_query_to_current_state,
)
# R2 拆分: 回复切分纯函数移至 reply_formatting; re-export 兼容既有导入.
from app.services.chat.reply_formatting import (
    _clean_reply_part,
    split_and_validate_replies,
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
from app.services.chat.tracing import create_tracer
from app.services.mbti import get_mbti
from app.services.interaction.reply_context import actual_delay_seconds, save_last_reply_timestamp
from app.services.proactive.state import start_or_restart_proactive_session
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
    achievement_turn_final: bool = True,
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

    # --- LLM usage 累加 session ---
    # 父调用启 session, 所有 phase 内 LLM wrapper 自动 record 进来; 出口 finally 写一行
    # llm_usage. sub_intent_mode 共享父 session, 不开新的 (避免子片段重复计费).
    from app.services.llm import usage_tracker
    usage_token = usage_tracker.start_session() if not sub_intent_mode else None

    # --- Trace (本地采集为主, LangSmith legacy) ---
    # 主调用 (sub_intent_mode=False): 开新 chat_request root run.
    # sub_intent 调用: attach 到 parent trace_id, 不开新 root — sub 内的 LLM
    # 调用通过 contextvars 里仍活跃的 parent handler 自动 attach 形成嵌套树,
    # sub 产生的消息 metadata.trace_id 跟 parent 一致, 用户点 trace 跳到 root
    # 视图能看到完整树 (而非只看 sub 子树).
    if sub_intent_mode and parent_trace_id:
        tracer = create_tracer(user_message, conversation_id).attach_to_parent(parent_trace_id)
    else:
        tracer = create_tracer(user_message, conversation_id).enter()
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
                "content": render_message_content_for_prompt(
                    m.content,
                    m.metadata if isinstance(m.metadata, dict) else None,
                ),
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
        if current_turn_ids:
            reply_context = dict(reply_context or {})
            reply_context["turn_message_ids"] = sorted(current_turn_ids)
        # Combined text of this turn, captured before multi-intent may replace
        # user_message with the primary fragment. Used to collapse aggregated
        # fragment rows into one coherent user turn in the reply prompt.
        aggregated_turn_text = user_message
        current_achievement_turn_id = _achievement_turn_id(current_turn_ids)
        covered_until_user_ts = _max_user_created_at(messages_dicts)
        # 重逢感知 (拟人度): 当前轮距上一轮最后一条消息的间隔. ≥30min 时
        # build_system_prompt 注入分档「重逢感知」段, B3 同时用它重置话题上下文.
        reengagement_gap_seconds = compute_reengagement_gap_seconds(
            messages_dicts, exclude_ids=current_turn_ids,
        )
        # W2 中期记忆: 摘要任务延后到主路径分支创建 (与 fetch_task 同处),
        # 避免 boundary/crisis/filler 短路回合白跑 LLM. 这里只声明.
        session_recap_task: asyncio.Task | None = None
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
        # crisis 路径不消费 pending（求救信号优先级高于一切），但也不能留着——
        # 否则用户脱离危机后的第一条消息会被误解析成"矛盾追问的回答/删除确认"。
        # 显式丢弃：矛盾之后会被重新检测，删除可由用户重新发起。
        if crisis_care_turn and not sub_intent_mode:
            await discard_pending_states_for_crisis(conversation_id)
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

            # E2 拟人度: 纯语气词概率性"仅表情"轻回应 — 真人收到"嗯/哈哈"
            # 不会每次都认真回一句话. AI 上一句是提问时不走此路径 (那是答复,
            # 交给意图管线); 未命中概率仍走完整管线. 位置在 preflight 之后:
            # pending 矛盾/删除确认的 "嗯/好" 已被上方消费, 不会误吞.
            filler_emoji = build_filler_emoji_reply(
                user_message,
                previous_assistant_text=getattr(previous_assistant, "content", None),
            )
            if filler_emoji is not None:
                logger.info(
                    f"[FILLER-EMOJI] reply={filler_emoji}",
                    extra={
                        "event": EVT_FILLER_EMOJI,
                        "filler_preview": user_message[:10],
                        "emoji": filler_emoji,
                    },
                )
                preflight_ctx.last_short_circuit_reply = filler_emoji
                for evt in await _short_circuit_reply(
                    filler_emoji, conversation_id, agent_id, user_id,
                    trace_id=tracer.safe_trace_id,
                ):
                    yield evt
                tracer.close()
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
            # W2 中期记忆: 重逢 (gap≥3h) 时并行生成「上次聊到」摘要.
            # 模块内部有 gap 阈值判断 + Redis 缓存 (同一次重逢只调一次 LLM);
            # gap 不足时任务瞬时返回 None, 零成本.
            # 生命周期: 意图短路路径由 _cancel_fetch_task 一并 cancel; tier 路径
            # 不消费但任务自然完成 (模块全链路吞异常, 结果落 Redis 缓存反而预热).
            session_recap_task = asyncio.create_task(get_or_build_session_recap(
                conversation_id, messages_dicts,
                gap_seconds=reengagement_gap_seconds,
                exclude_ids=current_turn_ids,
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
            """短路 / 异常时调用: cancel + 等待 propagate, 避免 orphan task warning.

            session_recap_task 一并取消 (review 发现): 意图短路路径不消费摘要,
            让它跑完是浪费一次 LLM 调用. 模块内部全链路吞异常, cancel 安全.
            """
            for task in (fetch_task, session_recap_task):
                if task is None or task.done():
                    continue
                task.cancel()
                try:
                    await task
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
            workspace_id=workspace_id,
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
            achievement_turn_final=achievement_turn_final,
        )
        response_diagnostics["crisis_followup_check_mode"] = (
            crisis_followup_check_mode if crisis_followup_active else None
        )

        if agent_id:
            try:
                from app.services.achievements.service import handle_intent_event

                intent_metadata = dict(detected_intent.metadata or {})
                intent_metadata["confidence"] = detected_intent.confidence
                intent_metadata["source"] = "chat_intent"
                _fire_background(handle_intent_event(
                    intent=detected_intent.intent.value,
                    user_id=user_id,
                    agent_id=agent_id,
                    workspace_id=workspace_id,
                    conversation_id=conversation_id,
                    message_id=user_message_id,
                    metadata=intent_metadata,
                ))
            except Exception as achievement_err:
                logger.debug(f"[ACH] intent event hook skipped: {achievement_err}")

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
        # topic_info dict 直传 prompt_builder, 由 chat.topic_context_section
        # 模板渲染 (registry 管理, trace 内可编辑).
        topic_info = await push_topic(
            conversation_id, user_message,
            gap_seconds=reengagement_gap_seconds,  # B3: ≥3h 间隔清空话题栈
        )
        topic_context = topic_info or None

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
                delay_reason_text = await explain_delay_reason(
                    str(reply_context.get("delay_reason", "")),
                    activity=received_activity,
                    status=received_status_label,
                )
                # 结构化 dict 直传 prompt_builder, 由 chat.delay_context_section
                # 模板渲染 (registry 管理, trace 内可编辑).
                delay_context = {
                    "received_at": received_at,
                    "activity": received_activity,
                    "status": received_status_label,
                    "delay_seconds": rounded_delay,
                    "delay_reason": delay_reason_text,
                }

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
        patience_instruction = await get_patience_prompt_instruction(cached_patience)

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
            music_context = None
            try:
                from app.services import music as music_service

                active_music = await music_service.get_active_co_listening(
                    conversation_id=conversation_id,
                )
                if active_music and active_music.track:
                    tpl = await get_prompt_text("music.co_listening_context")
                    music_context = safe_format(tpl, {
                        "current_song": active_music.track.title,
                        "current_artist": active_music.track.artist,
                    })
            except Exception as music_context_err:
                logger.debug(f"[MUSIC] co-listening context skipped: {music_context_err}")
            # E3 表达学习: 抽样已学表达 (Redis 读, ~1ms; 失败/为空则不注入)
            try:
                expression_habits = await sample_expression_habits(agent_id, user_id)
            except Exception as expr_err:
                logger.debug(f"[EXPR] habits sample skipped: {expr_err}")
                expression_habits = []
            # W2 中期记忆: 摘要任务已并行跑完/进行中, 这里取结果 (失败不注入)
            session_recap = None
            if session_recap_task is not None:
                try:
                    session_recap = await session_recap_task
                except Exception as recap_err:
                    logger.debug(f"[RECAP] session recap skipped: {recap_err}")
            # W3 关系时长感知: Redis 缓存 6h, miss 时 2 个轻量 DB 查询
            try:
                relation_meta_line = format_relation_meta_line(
                    await get_relation_meta(conversation_id),
                )
            except Exception as meta_err:
                logger.debug(f"[RELMETA] skipped: {meta_err}")
                relation_meta_line = ""
            # W4 AI 情绪连续性: 上一轮回复情绪衰减后作为本轮"当下心情"
            ai_mood_text = format_ai_mood_text(await load_ai_mood(conversation_id))
            # 图灵测试条数变化: 上一轮实际气泡数 → 注入"本轮条数≠上一轮"约束段
            last_reply_count = await load_last_reply_count(conversation_id)
            system_prompt = await build_system_prompt(
                agent=agent,
                memories=classified_memories,
                delay_context=delay_context,
                portrait=portrait,
                topic_context=topic_context,
                music_context=music_context,
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
                reengagement_gap_seconds=reengagement_gap_seconds,
                session_recap=session_recap,
                relation_meta_line=relation_meta_line,
                ai_mood_text=ai_mood_text,
                expression_habits=expression_habits or None,
                last_reply_count=last_reply_count,
                diagnostics=prompt_diagnostics,
            )
            response_diagnostics["main_prompt_built"] = True
            response_diagnostics["main_prompt_build_ms"] = round(
                (perf_counter() - started) * 1000,
                3,
            )
            response_diagnostics.update(prompt_diagnostics)
            # Aggregated fragment turns: collapse the per-fragment DB rows into
            # one coherent user turn so the reply LLM sees the message the user
            # actually meant (not just the last fragment). No-op for single-row
            # turns. Memory pipeline is unaffected (reads original rows).
            reply_messages = collapse_turn_fragments(
                messages_dicts,
                turn_message_ids=current_turn_ids,
                combined_text=aggregated_turn_text,
                combined_id=user_message_id,
            )
            return build_chat_messages(system_prompt, reply_messages)

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
            # W1b 后语义: 主 LLM 的 [EMO] 标记命中时此 fn 不会被调用,
            # 仅作标记缺失/失效时的回退路径 (见 extract_emotion_marker).
            reply_emotion_fn=_ai_reply_emotion,
            # 时间感知收口: gap ≥3h 重逢轮禁走 tier (轻量 prompt 无重逢/摘要段)
            reengagement_gap_seconds=reengagement_gap_seconds,
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
            conversation_id=conversation_id,
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
            turn_user_message_ids=list(current_turn_ids),
            achievement_turn_id=current_achievement_turn_id,
            achievement_turn_final=achievement_turn_final and not pending_sub_fragments,
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
        # 图灵测试条数变化: 记录本轮累计可见气泡数 (代码权威计数, 不信 LLM 自报).
        # 主调用 offset=0 写主回复数; 每个 sub-intent 递归写 offset+本段数 —
        # 最后一次写入即本轮总数 (SET 语义). key 按 conversation 隔离.
        if emitted_replies:
            _fire_background(save_last_reply_count(
                conversation_id, reply_index_offset + len(emitted_replies),
            ))

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
            workspace_id=workspace_id,
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
                workspace_id=workspace_id,
            ))

        # 兜底关闭 trace (幂等): 正常/短路路径都已 close, 这里只兜异常中断.
        # LocalTracer 依赖 close 还原 ContextVar handler; 不还原会让同一
        # task 处理的下一条消息的 LLM 调用错挂到本次 trace 树.
        tracer.close()
