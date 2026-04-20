"""Spec §4 step 5 + §5.5：主回复生成。

三条路径（按优先级）：
1. 若有 contradiction inquiry → 直接用 inquiry 当回复（跳过 LLM）
2. 纯日常交流 + 无额外上下文 → 走记忆分级 prompt（weak/medium/strong/L3）
3. 兜底 → 主 LLM 流式 + §5.5 句数拆分

不 yield 事件；纯返回 `(replies, raw_response)` 供调用方发布。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.services.chat.intent_dispatcher import IntentResult, IntentType
from app.services.llm.models import convert_messages, get_chat_model
from app.services.prompts.system_prompts import MAX_PER_REPLY

logger = logging.getLogger(__name__)


def can_use_tier_reply(
    *,
    intent: IntentType,
    memory_relevance: str,
    relational_context: str | None,
    schedule_context: str | None,
    delay_context: str | None,
) -> bool:
    """spec §4：仅纯聊天 + NONE/L3_RECALL 意图 + 三种 context 都不需要时可走分级 prompt。"""
    return (
        intent in (IntentType.NONE, IntentType.L3_RECALL)
        and not relational_context
        and not schedule_context
        and not delay_context
        and memory_relevance in ("weak", "medium", "strong")
    )


def _build_tier_call(
    memory_relevance: str,
    l3_memories: list[str],
    combined_memory: str,
    tier_fns: dict[str, Callable[..., Awaitable[str | None]]],
) -> tuple[Callable[..., Awaitable[str | None]], dict[str, Any]]:
    """选择 tier 函数 + 它需要的额外参数。"""
    if l3_memories:
        return tier_fns["l3"], {"l3_memory": "\n".join(f"- {t}" for t in l3_memories)}
    if memory_relevance == "strong":
        return tier_fns["strong"], {"user_memory": combined_memory, "ai_memory": "(同上)"}
    if memory_relevance == "medium":
        return tier_fns["medium"], {"user_memory": combined_memory, "ai_memory": "(同上)"}
    return tier_fns["weak"], {}


async def _run_main_llm(chat_messages: list[dict]) -> str:
    """主 LLM 流式调用，收集完整响应。"""
    model = get_chat_model()
    lc_messages = convert_messages(chat_messages)
    chunks: list[str] = []
    async for chunk in model.astream(lc_messages):
        token = chunk.content
        if token:
            chunks.append(token)
    return "".join(chunks)


async def _split_replies(
    raw_response: str,
    reply_count: int,
    max_reply_count: int,
    max_per_reply: int,
    max_total: int,
    split_llm_fn: Callable[[str, int], Awaitable[list[str] | None]],
    truncate_fn: Callable[[str, int], str],
    pipe_fallback_fn: Callable[[str, int, int, int], list[str]],
) -> tuple[list[str], str]:
    """spec §5.5：n=1 保持；n>=2 调小模型拆分；失败回退 `||`。返回 (replies, split_source)。"""
    if reply_count >= 2:
        llm_result = await split_llm_fn(raw_response, reply_count)
        if llm_result:
            return (
                [truncate_fn(p, max_per_reply) for p in llm_result[:max_reply_count]],
                f"llm_{reply_count}",
            )
    return (
        pipe_fallback_fn(raw_response, max_reply_count, max_per_reply, max_total),
        "single" if reply_count == 1 else "pipe_fallback",
    )


async def generate_reply(
    *,
    contradiction_inquiry: str | None,
    detected_intent: IntentResult,
    memory_relevance: str,
    relational_context: str | None,
    schedule_context: str | None,
    delay_context: str | None,
    l3_memories: list[str],
    classified_memories: list,
    messages_dicts: list[dict],
    portrait: Any,
    prompt_user_emotion: dict | None,
    user_message: str,
    agent: Any,
    chat_messages: list[dict],
    reply_count: int,
    max_reply_count: int,
    max_total: int,
    tier_fns: dict[str, Callable[..., Awaitable[str | None]]],
    split_llm_fn: Callable[[str, int], Awaitable[list[str] | None]],
    truncate_fn: Callable[[str, int], str],
    pipe_fallback_fn: Callable[[str, int, int, int], list[str]],
) -> tuple[list[str], str]:
    """返回 (replies, raw_response)。"""
    if contradiction_inquiry:
        return [contradiction_inquiry], contradiction_inquiry

    tier_reply_text: str | None = None
    if can_use_tier_reply(
        intent=detected_intent.intent,
        memory_relevance=memory_relevance,
        relational_context=relational_context,
        schedule_context=schedule_context,
        delay_context=delay_context,
    ):
        personality_brief = getattr(agent, "name", "") or ""
        context_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages_dicts[-6:]
        ) or "(无)"
        portrait_text = str(portrait) if portrait else "(未知)"
        memory_lines = [m.text for m in (classified_memories or [])]
        combined_memory = "\n".join(f"- {t}" for t in memory_lines) if memory_lines else "(无)"
        base_params = {
            "message": user_message,
            "context": context_text,
            "user_emotion": prompt_user_emotion,
            "personality_brief": personality_brief,
            "user_portrait": portrait_text,
        }
        tier_fn, extra = _build_tier_call(memory_relevance, l3_memories, combined_memory, tier_fns)
        try:
            tier_reply_text = await tier_fn(**base_params, **extra)
        except Exception as e:
            logger.warning(f"Memory tier reply failed, falling back to main prompt: {e}")
            tier_reply_text = None

    if tier_reply_text:
        return [tier_reply_text], tier_reply_text

    raw_response = await _run_main_llm(chat_messages)
    replies, split_source = await _split_replies(
        raw_response,
        reply_count,
        max_reply_count,
        MAX_PER_REPLY,
        max_total,
        split_llm_fn,
        truncate_fn,
        pipe_fallback_fn,
    )
    logger.info(
        f"[REPLY-SPLIT] n_target={reply_count} actual={len(replies)} "
        f"source={split_source}"
    )
    return replies, raw_response
