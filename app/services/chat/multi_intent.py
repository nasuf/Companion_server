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
from collections.abc import AsyncGenerator
from typing import Any

from app.services.chat.intent_dispatcher import (
    INTENT_PRIORITY,
    IntentType,
    LABEL_TO_INTENT,
)
from app.services.interaction.reply_context import save_last_reply_timestamp
from app.services.runtime.tasks import fire_background as _fire_background

logger = logging.getLogger(__name__)

_DONE_EVENT = {"event": "done", "data": json.dumps({"message_id": "complete"})}


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
) -> list[dict]:
    """构造短路分支的 SSE 事件列表。

    - save_replies_fn: 由调用方注入的 `_save_replies(conversation_id, [reply])`
      协程工厂，避免 multi_intent 依赖 orchestrator 的持久化实现。
    - sub_intent_mode=True：父调用负责 save_last_reply_timestamp/done。
    - include_done=False：延后 done（用于主调用随后处理 sub fragments）。
    - extra_metadata：透传给 save_replies_fn 的持久化 metadata（如 boundary/zone/attack_level）。
    """
    reply_payload: str | dict = reply
    if extra_metadata:
        reply_payload = {"text": reply, **extra_metadata}
    _fire_background(save_replies_fn(conversation_id, [reply_payload]))
    if not sub_intent_mode and agent_id:
        await save_last_reply_timestamp(agent_id, user_id)
    events: list[dict] = [{
        "event": "reply",
        "data": json.dumps({"text": reply, "index": reply_index_offset}),
    }]
    if include_done and not sub_intent_mode:
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
) -> AsyncGenerator[dict, None]:
    """spec §3.3 step 3：按 priority 顺序处理拆分后的子意图片段。

    每个片段作为独立子调用进入 stream_chat_response(sub_intent_mode=True)，
    共享 reply_context 沿用首条消息的 due_at（spec §6）。
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
    for label, text in ordered:
        intent_type = LABEL_TO_INTENT.get(label, IntentType.NONE)
        logger.info(
            f"[INTENT-SUB] label={label} intent={intent_type.value} "
            f"index={cur_index} text={text[:40]!r}"
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
) -> AsyncGenerator[dict, None]:
    """短路分支尾部：primary reply → sub-intent 循环 → done → trace 关闭。

    sub_intent_mode=True 时跳过 done/trace（由父调用完成）。
    tracer 是 `LangSmithTracer` 实例；本函数仅调 `tracer.close()`。
    """
    events = await short_circuit_reply(
        reply, conversation_id, agent_id, user_id, save_replies_fn,
        sub_intent_mode=sub_intent_mode,
        reply_index_offset=reply_index_offset,
        include_done=False,
    )
    for evt in events:
        yield evt

    if not sub_intent_mode and pending_sub_fragments:
        async for evt in process_sub_intents(
            pending_sub_fragments, conversation_id, agent, user_id,
            reply_context, start_index=reply_index_offset + 1,
            parent_patience=cached_patience,
        ):
            yield evt

    if not sub_intent_mode:
        yield _DONE_EVENT
        tracer.close()
