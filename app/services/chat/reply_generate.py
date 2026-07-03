"""Spec §4 step 5 + §5.5：主回复生成。

三条路径（按优先级）：
1. 若有 contradiction inquiry → 直接用 inquiry 当回复（跳过 LLM）
2. 纯日常交流 + 无额外上下文 → 走记忆分级 prompt（weak/medium/strong/L3）
3. 兜底 → 主 LLM 流式 + §5.5 句数拆分

不 yield 事件；纯返回 `(replies, raw_response)` 供调用方发布。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from app.observability.events import EVT_REPLY_LLM, EVT_REPLY_SPLIT, EVT_REPLY_TIER
from app.services.chat.intent_dispatcher import IntentResult, IntentType
from app.services.llm.models import convert_messages, get_chat_model, get_fallback_chat_model
from app.services.llm.resilience import (
    LLMFailedError,
    collect_stream,
    get_profile,
    provider_name,
)
from app.services.memory.retrieval.context_selector import split_by_source
from app.services.prompts.system_prompts import MAX_PER_REPLY

logger = logging.getLogger(__name__)


# 两级 LLM (primary + Ollama) 全挂时的静态兜底回复. 措辞刻意保持"走神"风格,
# 让用户意识到异常又不吓到. 搭配 reply metadata {reply_failed: true},
# 前端未来可选提供"重新回答"按钮.
_MAIN_REPLY_ULTIMATE_FALLBACK = "诶,我这会儿有点走神……你刚说的是什么?"


def can_use_tier_reply(
    *,
    intent: IntentType,
    memory_relevance: str,
    relational_context: str | None,
    schedule_context: str | None,
    delay_context: dict | None,
) -> bool:
    """spec §4：纯聊天 + 无关系/延迟特殊处理时可走轻量分级 prompt.

    schedule_context 是作息查询分支的参考信息；§4 主回复 prompt 已不再注入它，
    因此不能用它阻塞 tier reply。真正的作息类消息由 intent 条件拦截。
    """
    return (
        intent in (IntentType.NONE, IntentType.L3_RECALL)
        and not relational_context
        and not delay_context
        and memory_relevance in ("weak", "medium", "strong")
    )


def _build_tier_call(
    memory_relevance: str,
    l3_memories: list[str],
    *,
    user_memory: str,
    ai_memory: str,
    tier_fns: dict[str, Callable[..., Awaitable[str | None]]],
) -> tuple[Callable[..., Awaitable[str | None]], dict[str, Any]]:
    """选择 tier 函数 + 它需要的额外参数. 见 ClassifiedMemory.source 分流原因."""
    if l3_memories:
        return tier_fns["l3"], {"l3_memory": "\n".join(f"- {t}" for t in l3_memories)}
    if memory_relevance == "strong":
        return tier_fns["strong"], {"user_memory": user_memory, "ai_memory": ai_memory}
    if memory_relevance == "medium":
        return tier_fns["medium"], {"user_memory": user_memory, "ai_memory": ai_memory}
    return tier_fns["weak"], {}


async def _run_main_llm(chat_messages: list[dict]) -> tuple[str, bool]:
    """主 LLM 流式调用，收集完整响应。

    三级降级策略 (resilience.astream_with_resilience):
    1. primary (Dashscope / Claude / 或配置指定的其他) 流式, 首 chunk 在
       first_chunk_timeout_s 内到达 → commit 到 primary
    2. 首 chunk 未到 → 无副作用切本地 Ollama LOCAL_CHAT_MODEL 流
    3. Ollama 也挂 → 抛 LLMFailedError, 我们落到静态兜底文本

    返回 (text, is_fallback). is_fallback=True 表示走了静态兜底 (两级 LLM 全挂),
    调用方可据此给 reply metadata 打 `{reply_failed: true}` 让前端显示重试按钮等.
    """
    from app.services.llm.models import _resolve_usage_model_key

    primary = get_chat_model()
    primary_prov = provider_name(primary)
    lc_messages = convert_messages(chat_messages)

    # primary 若本就是 Ollama, 不配 fallback (避免本地 → 本地二次重试无意义)
    fallback = get_fallback_chat_model() if primary_prov != "ollama" else None

    def _primary_stream():
        return primary.astream(lc_messages)

    def _fallback_stream():
        return fallback.astream(lc_messages)

    try:
        text = await collect_stream(
            _primary_stream,
            primary_provider=primary_prov,
            profile=get_profile("chat_stream"),
            op="reply_stream",
            fallback_factory=(_fallback_stream if fallback is not None else None),
            primary_model_name=_resolve_usage_model_key(primary),
            fallback_model_name=_resolve_usage_model_key(fallback) if fallback else "",
        )
        return text, False
    except LLMFailedError as e:
        logger.warning(f"[LLM-FALLBACK] reply_stream total failure (primary + ollama both down): {e}")
        return _MAIN_REPLY_ULTIMATE_FALLBACK, True


async def _split_replies(
    raw_response: str,
    max_reply_count: int,
    max_per_reply: int,
    max_total: int,
    truncate_fn: Callable[[str, int], str],
    pipe_fallback_fn: Callable[[str, int, int, int], list[str]],
) -> tuple[list[str], str]:
    """主 LLM 输出按 || / \\n\\n 切分.

    历史 (split_llm 路径) 已删: 主回复 LLM 在 chat.response_instruction 已被指令
    "分N条||分隔", 直接信主 LLM 输出. 不再额外调小模型拆分 (省 1 次 LLM call +
    消除截断/扩写 2 个 bug 源).
    - 主 LLM 给"句1||句2" → 拆 2 条
    - 主 LLM 给单句 → 单条 (LLM 不听话或内容确实简短)

    truncate_fn 仍保留以约束单条最长字数 (max_per_reply).
    """
    parts = pipe_fallback_fn(raw_response, max_reply_count, max_per_reply, max_total)
    parts = [truncate_fn(p, max_per_reply) for p in parts]
    source = "single" if len(parts) <= 1 else "main_split"
    return parts, source


async def generate_reply(
    *,
    contradiction_inquiry: str | None,
    detected_intent: IntentResult,
    memory_relevance: str,
    relational_context: str | None,
    schedule_context: str | None,
    delay_context: dict | None,
    l3_memories: list[str],
    classified_memories: list,
    messages_dicts: list[dict],
    portrait: Any,
    prompt_user_emotion: dict | None,
    user_message: str,
    agent: Any,
    reply_count: int,
    max_reply_count: int,
    max_total: int,
    tier_fns: dict[str, Callable[..., Awaitable[str | None]]],
    truncate_fn: Callable[[str, int], str],
    pipe_fallback_fn: Callable[[str, int, int, int], list[str]],
    chat_messages: list[dict] | None = None,
    chat_messages_factory: Callable[[], Awaitable[list[dict]]] | None = None,
    reply_emotion_fn: Callable[[str], Awaitable[dict]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[str], str, bool, dict | None]:
    """返回 (replies, raw_response, is_fallback, reply_emotion).

    is_fallback=True 表示主 LLM 和 Ollama 都挂, 走了静态兜底文本;
    调用方可据此在 reply metadata 加 `{reply_failed: true}` 供前端显示重试按钮.
    tier 分级回复和 contradiction inquiry 路径始终 is_fallback=False.

    reply_emotion_fn (可选): 主 LLM 流式完成后, 立刻并行启动情绪识别小模型,
    跟 _split_replies 同时跑 (两者都只依赖 raw_response, 互不依赖). 命中时
    返回 dict; 未传 / tier 路径短路 / contradiction_inquiry 路径返 None,
    调用方需 fallback 自行调 _ai_reply_emotion.

    chat_messages_factory 用于懒构建主 prompt；只有兜底到主 LLM 时才会被 await。
    """

    async def _get_chat_messages() -> list[dict]:
        if chat_messages is not None:
            return chat_messages
        if chat_messages_factory is None:
            raise ValueError("generate_reply requires chat_messages or chat_messages_factory")
        return await chat_messages_factory()

    if contradiction_inquiry:
        if diagnostics is not None:
            diagnostics["reply_path"] = "contradiction"
        return [contradiction_inquiry], contradiction_inquiry, False, None

    tier_reply_text: str | None = None
    tier_eligible = can_use_tier_reply(
        intent=detected_intent.intent,
        memory_relevance=memory_relevance,
        relational_context=relational_context,
        schedule_context=schedule_context,
        delay_context=delay_context,
    )
    if diagnostics is not None:
        diagnostics["tier_eligible"] = tier_eligible
        diagnostics["memory_relevance"] = memory_relevance
    if tier_eligible:
        personality_brief = getattr(agent, "name", "") or ""
        context_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages_dicts[-6:]
        ) or "(无)"
        portrait_text = str(portrait) if portrait else "(未知)"
        user_lines, ai_lines = split_by_source(classified_memories)
        user_memory_text = "\n".join(f"- {t}" for t in user_lines) if user_lines else "(无)"
        ai_memory_text = "\n".join(f"- {t}" for t in ai_lines) if ai_lines else "(无)"
        base_params = {
            "message": user_message,
            "context": context_text,
            "user_emotion": prompt_user_emotion,
            "personality_brief": personality_brief,
            "user_portrait": portrait_text,
            # Phase: 让 tier reply 也享受 random 1-3 条 (跟主回复一致, 微信多条体感).
            # tier prompt 已加 {n}/{max_per}/{total} 占位符 + || 分隔指令.
            "n": reply_count,
            "max_per": MAX_PER_REPLY,
            "max_total": max_total,
        }
        tier_fn, extra = _build_tier_call(
            memory_relevance, l3_memories,
            user_memory=user_memory_text, ai_memory=ai_memory_text,
            tier_fns=tier_fns,
        )
        if diagnostics is not None:
            diagnostics["tier_kind"] = (
                "l3" if l3_memories else memory_relevance
            )
        try:
            tier_reply_text = await tier_fn(**base_params, **extra)
        except Exception as e:
            logger.warning(f"Memory tier reply failed, falling back to main prompt: {e}")
            tier_reply_text = None
            if diagnostics is not None:
                diagnostics["tier_error"] = type(e).__name__

    if tier_reply_text:
        if diagnostics is not None:
            diagnostics["reply_path"] = "tier"
        # Phase: tier reply 输出可能含 || 多条 (n≥2 时), 走 split_and_validate_replies
        # 拆分. n=1 单条时也走 (含 || 时仍能正确切, 不含时返单条).
        tier_replies = pipe_fallback_fn(
            tier_reply_text, max_reply_count, MAX_PER_REPLY, max_total,
        )
        logger.info(
            f"[REPLY-TIER] memory_relevance={memory_relevance} "
            f"raw_len={len(tier_reply_text)} n_target={reply_count} actual={len(tier_replies)}",
            extra={
                "event": EVT_REPLY_TIER,
                "memory_relevance": memory_relevance,
                "reply_text_len": len(tier_reply_text),
                "has_l3": bool(l3_memories),
                "n_target": reply_count,
                "n_actual": len(tier_replies),
            },
        )
        return tier_replies, tier_reply_text, False, None

    if diagnostics is not None:
        diagnostics["reply_path"] = "main_llm"
        if tier_eligible:
            diagnostics["tier_empty_or_failed"] = True
    raw_response, is_fallback = await _run_main_llm(await _get_chat_messages())
    logger.info(
        f"[REPLY-LLM] main reply len={len(raw_response)} fallback={is_fallback}",
        extra={
            "event": EVT_REPLY_LLM,
            "raw_response_len": len(raw_response),
            "is_fallback": is_fallback,
        },
    )

    # split + emotion 都只依赖 raw_response, 互不依赖 → 并行省 ~400-1000ms.
    # is_fallback=True (主+本地全挂) 时 raw_response 是静态兜底文案, 仍可情绪识别.
    split_coro = _split_replies(
        raw_response, max_reply_count, MAX_PER_REPLY,
        max_total, truncate_fn, pipe_fallback_fn,
    )
    if reply_emotion_fn is not None:
        emotion_coro = reply_emotion_fn(raw_response)
        (replies, split_source), reply_emotion = await asyncio.gather(
            split_coro, emotion_coro, return_exceptions=False,
        )
    else:
        replies, split_source = await split_coro
        reply_emotion = None

    logger.info(
        f"[REPLY-SPLIT] n_target={reply_count} actual={len(replies)} "
        f"source={split_source} is_fallback={is_fallback}",
        extra={
            "event": EVT_REPLY_SPLIT,
            "n_target": reply_count,
            "n_actual": len(replies),
            "split_source": split_source,
            "is_fallback": is_fallback,
        },
    )
    return replies, raw_response, is_fallback, reply_emotion
