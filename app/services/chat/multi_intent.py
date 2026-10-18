"""Multi-intent 短路分支 + 子意图递归调度（spec §3.3 step 3）。

`stream_chat_response` 的多处短路分支共用的三个工具：

- `short_circuit_reply`: 落库 + 构造 reply/done SSE 事件列表
- `finalize_short_circuit`: 短路分支尾部：主回复 → 子意图循环 → done/trace
- `process_sub_intents`: 按 priority 依次递归处理拆分后的子意图

为避免与 `orchestrator.stream_chat_response` 的循环引用，
`process_sub_intents` 在函数内部延迟导入。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from app.db import db
from app.services.chat.message_utils import _achievement_turn_id
from app.observability.events import EVT_INTENT_SUB_RECURSED
from app.services.chat.intent_dispatcher import (
    INTENT_PRIORITY,
    IntentType,
    LABEL_TO_INTENT,
)
from app.services.chat.turn_lifecycle import finish_assistant_turn
from app.services.runtime.tasks import fire_background as _fire_background

logger = logging.getLogger(__name__)

_DONE_EVENT = {"event": "done", "data": json.dumps({"message_id": "complete"})}


def _count_reply_events(events: list[dict]) -> int:
    return sum(1 for evt in events if evt.get("event") == "reply")


def delivered_reply_text(events: list[dict]) -> str:
    """short_circuit_reply 事件里实际发给用户的全文 (多气泡空格拼接).

    读不出文本时返回空串, 调用方回退到自己手里的原文."""
    texts: list[str] = []
    for evt in events:
        if evt.get("event") != "reply":
            continue
        try:
            text = json.loads(evt["data"]).get("text")
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if isinstance(text, str) and text:
            texts.append(text)
    return " ".join(texts)


def _turn_user_message_ids(reply_context: dict | None) -> list[str]:
    if not isinstance(reply_context, dict):
        return []
    raw_ids = reply_context.get("turn_message_ids")
    if not isinstance(raw_ids, list):
        return []
    return list(dict.fromkeys(item for item in raw_ids if isinstance(item, str) and item))


async def short_circuit_reply(
    reply: str,
    conversation_id: str,
    agent_id: str | None,
    user_id: str,
    save_replies_fn,
    *,
    sub_intent_mode: bool = False,
    reply_index_offset: int = 0,
    include_done: bool = True,
    extra_metadata: dict | None = None,
    trace_id: str | None = None,
    turn_user_message_ids: list[str] | None = None,
    achievement_turn_final: bool = True,
    agent: Any = None,
    reply_context: dict | None = None,
    voice_context: Any = None,
    proactive_reason: str | None = "short_circuit",
    workspace_id: str | None = None,
    defer_turn_finalization: bool = False,
    persona_conflict_fallback: str | None = None,
) -> list[dict]:
    """构造短路分支的 SSE 事件列表。

    - save_replies_fn: 由调用方注入的 `_save_replies(conversation_id, [reply])`
      协程工厂，避免 multi_intent 依赖 orchestrator 的持久化实现.
    - sub_intent_mode=True：父调用负责回合收尾 (finish_assistant_turn) / done.
    - proactive_reason: 回合收尾时 arm 主动交流的原因; None = 本轮之后不主动 (边界系统).
    - workspace_id: 透传给回合收尾 (免一次反查); turn_user_message_ids 同时用作过期 arm 守卫.
    - include_done=False：延后 done（用于主调用随后处理 sub fragments）.
    - extra_metadata：透传给 save_replies_fn 的持久化 metadata（如 boundary/zone/attack_level）.
    - trace_id: 挂到首条 reply.metadata, 让前端 Trace 按钮可点. sub_intent_mode
      或非父调用可不传.
    - reply 可含 || 气泡分隔 (模型按前置【通用回复规则】写的), 在此切成多条
      气泡, 每条一个 reply 事件、index 连续; 返回列表里的 reply 事件数即气泡数.
    """
    # 硬保证: 一个回合最多 1 个 emoji (短路回复不走 emit_replies 的出口兜底,
    # 在此单独收口; 持久化与 SSE 推送用同一份清理后的文本). 切分前对整段收口,
    # 多气泡也只留一个.
    # 系统标记 (EMO/条数) 在切分时逐条剥 — 短路 prompt 也带 reply_prefix, LLM
    # 可能跟着输出标记, 不剥会直接漏给用户 (生产 2026-07-19 复现). 整条都是标记
    # 时给占位省略号, 绝不回退到未清理的原文.
    from app.services.chat.reply_formatting import split_short_circuit_reply
    from app.services.emoji import limit_emojis
    bubbles = split_short_circuit_reply(limit_emojis(reply or ""))
    reply = " ".join(bubbles)
    if agent is not None:
        from app.services.persona_grounding import guard_reply
        guarded, corrected = await guard_reply(agent, reply)
        if corrected:
            reply = persona_conflict_fallback or guarded
            bubbles = [reply]
            extra_metadata = {**(extra_metadata or {}), "persona_location_corrected": True}
    # 语义 metadata (boundary / response_diagnostics.short_circuit_kind / 重复追问
    # 键…) 每条气泡都带: 判定窗与重复追问检测读的是最后一条 AI 消息.
    metadata = dict(extra_metadata or {})
    # 追踪快照只挂首条, 与主回复 persist_reply 一致.
    first_metadata: dict[str, Any] = {}
    if not sub_intent_mode:
        try:
            from app.services.memory.retrieval.trace import snapshot_retrieval_traces
            retrieval_traces = snapshot_retrieval_traces()
            if retrieval_traces:
                first_metadata["memory_retrievals"] = retrieval_traces
        except Exception:
            pass
        try:
            from app.services.prompting.trace_components import snapshot_prompt_render_traces
            prompt_traces = snapshot_prompt_render_traces()
            if prompt_traces:
                first_metadata["prompt_render_traces"] = prompt_traces
        except Exception:
            pass
    prepared_voice = None
    if agent is not None and voice_context is not None:
        from app.services.speech_output.policy import should_generate_voice

        if await should_generate_voice(
            context=voice_context,
            client_supports_voice=bool(
                (reply_context or {}).get("client_supports_voice")
            ),
        ):
            try:
                from app.services.speech_output.delivery import prepare_voice_output

                prepared_voice = await prepare_voice_output(
                    text=reply,
                    user_id=user_id,
                    agent=agent,
                    conversation_id=conversation_id,
                    source="chat",
                    detect_missing_emotion=True,
                )
            except Exception as voice_error:
                logger.warning(
                    "[TTS] short-circuit synthesis failed; falling back to text: %s",
                    type(voice_error).__name__,
                )

    if prepared_voice is not None:
        # 语音只发一条, 念整段 (与主回复 normalize_reply 一致).
        bubbles = [reply]
        metadata.update({
            "display_mode": "voice",
            "attachments": [prepared_voice.metadata],
        })
    reply_payloads: list[str | dict] = []
    for i, text in enumerate(bubbles):
        payload_metadata = {**first_metadata, **metadata} if i == 0 else metadata
        reply_payloads.append(
            {"text": text, **payload_metadata} if payload_metadata else text
        )
    save_args = (
        conversation_id,
        reply_payloads,
    )
    save_kwargs = {
        "trace_id": trace_id,
        "turn_user_message_ids": turn_user_message_ids or [],
        "achievement_turn_id": _achievement_turn_id(turn_user_message_ids or []),
        "achievement_turn_final": achievement_turn_final,
    }
    assistant_message_id = None
    if prepared_voice is not None:
        try:
            assistant_message_id = await save_replies_fn(*save_args, **save_kwargs)
        except BaseException:
            from app.services.speech_output.delivery import discard_prepared_voice_output
            try:
                await discard_prepared_voice_output(prepared_voice)
            except Exception:
                logger.warning("[TTS] short-circuit cleanup failed", exc_info=True)
            raise
        if assistant_message_id:
            try:
                from app.services.speech_output.delivery import (
                    bind_prepared_voice_output,
                )

                await bind_prepared_voice_output(
                    prepared_voice,
                    message_id=assistant_message_id,
                )
            except Exception as bind_error:
                logger.warning(
                    "[TTS] short-circuit bind failed; falling back to text: %s",
                    type(bind_error).__name__,
                )
                from app.services.speech_output.delivery import (
                    discard_prepared_voice_output,
                )

                await discard_prepared_voice_output(prepared_voice)
                await db.execute_raw(
                    """
                    UPDATE messages
                    SET metadata = COALESCE(metadata, '{}'::jsonb)
                        - 'attachments' - 'display_mode'
                    WHERE id = $1
                    """,
                    assistant_message_id,
                )
                prepared_voice = None
        else:
            from app.services.speech_output.delivery import (
                discard_prepared_voice_output,
            )

            await discard_prepared_voice_output(prepared_voice)
            prepared_voice = None
    else:
        assistant_message_id = await save_replies_fn(*save_args, **save_kwargs)
    if not assistant_message_id:
        raise RuntimeError("Short-circuit reply persistence failed")
    # 图灵测试条数变化: 短路回复也计入"上一轮条数" (用户感知的是气泡数,
    # 不区分回复来自哪条管线). 累计 offset+气泡数, 最后一次写入即本轮总数.
    from app.services.chat.reply_count_state import save_last_reply_count
    _fire_background(
        save_last_reply_count(conversation_id, reply_index_offset + len(bubbles))
    )
    if not sub_intent_mode and not defer_turn_finalization:
        await finish_assistant_turn(
            conversation_id=conversation_id,
            agent_id=agent_id,
            user_id=user_id,
            proactive_reason=proactive_reason,
            workspace_id=workspace_id,
            turn_message_ids=turn_user_message_ids,
        )
    events: list[dict] = []
    for i, text in enumerate(bubbles):
        event_data: dict[str, Any] = {
            "text": text,
            "index": reply_index_offset + i,
        }
        if prepared_voice is not None:
            event_data.update({
                "display_mode": "voice",
                "attachments": [prepared_voice.metadata],
                "assistant_message_id": assistant_message_id,
            })
        events.append({
            "event": "reply",
            "data": json.dumps(event_data),
        })
    if include_done and not sub_intent_mode and not defer_turn_finalization:
        events.append(_DONE_EVENT)
    return events


async def process_sub_intents(
    pending_sub_fragments: dict[str, str],
    conversation_id: str,
    agent,
    user_id: str,
    reply_context: dict | None,
    start_index: int,
    parent_patience: int,
    parent_trace_id: str | None = None,
    memory_usage_event_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """spec §3.3 step 3：按 priority 顺序处理拆分后的子意图片段。

    每个片段作为独立子调用进入 stream_chat_response(sub_intent_mode=True)，
    共享 reply_context 沿用首条消息的 due_at（spec §6）。

    parent_trace_id 透传给 sub: sub 的 tracer 用 attach_to_parent 复用
    parent run_id, 让 sub 产生的消息 metadata.trace_id 跟 parent 一致 — 用户点
    任意 reply 的 trace 按钮跳到 parent 视图, 看到完整 root + nested 树.
    """
    if not pending_sub_fragments:
        return

    # 延迟导入避免循环依赖
    from app.services.chat.orchestrator import stream_chat_response

    ordered = sorted(
        pending_sub_fragments.items(),
        key=lambda kv: INTENT_PRIORITY.index(kv[0]) if kv[0] in INTENT_PRIORITY else float("inf"),
    )
    cur_index = start_index
    for index, (label, text) in enumerate(ordered):
        intent_type = LABEL_TO_INTENT.get(label, IntentType.NONE)
        is_last_sub_intent = index == len(ordered) - 1
        logger.info(
            f"[INTENT-SUB] label={label} intent={intent_type.value} "
            f"index={cur_index} text={text[:40]!r}",
            extra={
                "event": EVT_INTENT_SUB_RECURSED,
                "intent_label": label,
                "intent": intent_type.value,
                "fragment_index": cur_index,
                "fragment_len": len(text),
            },
        )
        async for evt in stream_chat_response(
            conversation_id=conversation_id,
            user_message=text,
            agent=agent,
            user_id=user_id,
            reply_context=reply_context,
            save_user_message=False,
            delivered_from_queue=True,
            sub_intent_mode=True,
            forced_intent=intent_type,
            reply_index_offset=cur_index,
            parent_patience=parent_patience,
            parent_trace_id=parent_trace_id,
            achievement_turn_final=is_last_sub_intent,
            memory_usage_event_id=memory_usage_event_id,
        ):
            if evt.get("event") == "done":
                continue
            yield evt
            if evt.get("event") == "reply":
                cur_index += 1


async def finalize_short_circuit(
    reply: str,
    *,
    conversation_id: str,
    agent_id: str | None,
    user_id: str,
    agent,
    reply_context: dict | None,
    tracer: Any,
    save_replies_fn,
    pending_sub_fragments: dict[str, str],
    sub_intent_mode: bool,
    reply_index_offset: int,
    cached_patience: int,
    extra_metadata: dict | None = None,
    achievement_turn_final: bool = True,
    voice_context: Any = None,
    proactive_reason: str | None = "short_circuit",
    workspace_id: str | None = None,
    defer_turn_finalization: bool = False,
    persona_conflict_fallback: str | None = None,
    memory_usage_event_id: str | None = None,
    on_delivered: Callable[[str], None] | None = None,
) -> AsyncGenerator[dict, None]:
    """短路分支尾部：primary reply → sub-intent 循环 → done → trace 关闭。

    sub_intent_mode=True 时跳过 done/trace（由父调用完成）。
    tracer 是 `ChatTracer` 实例（LocalTracer / LangSmithTracer）；本函数仅调 `tracer.close()`。
    on_delivered: 收到本条短路回复实际发出的全文 (多气泡空格拼接, 含人设纠正),
    在推送前回调一次; 子意图的回复不计入.
    """
    # sub_intent_mode 父调用已经保存过 trace_id 到首条 reply.metadata, 不重复挂.
    trace_id = (
        tracer.safe_trace_id if not sub_intent_mode and tracer is not None else None
    )
    events = await short_circuit_reply(
        reply, conversation_id, agent_id, user_id, save_replies_fn,
        sub_intent_mode=sub_intent_mode,
        reply_index_offset=reply_index_offset,
        include_done=False,
        extra_metadata=extra_metadata,
        trace_id=trace_id,
        turn_user_message_ids=_turn_user_message_ids(reply_context),
        achievement_turn_final=achievement_turn_final and not pending_sub_fragments,
        agent=agent,
        reply_context=reply_context,
        voice_context=voice_context,
        proactive_reason=proactive_reason,
        workspace_id=workspace_id,
        defer_turn_finalization=defer_turn_finalization,
        persona_conflict_fallback=persona_conflict_fallback,
    )
    delivered = delivered_reply_text(events) if on_delivered is not None else ""
    if delivered:
        on_delivered(delivered)
    for evt in events:
        yield evt

    if not sub_intent_mode and not defer_turn_finalization and pending_sub_fragments:
        async for evt in process_sub_intents(
            pending_sub_fragments, conversation_id, agent, user_id,
            reply_context,
            start_index=reply_index_offset + _count_reply_events(events),
            parent_patience=cached_patience,
            memory_usage_event_id=memory_usage_event_id,
        ):
            yield evt

    if not sub_intent_mode and not defer_turn_finalization:
        yield _DONE_EVENT
        tracer.close()
