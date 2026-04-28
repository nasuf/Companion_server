"""Spec §2-§6 意图/边界/记忆/延迟 各分级 prompt 的统一调用入口。

把 12 个已注册但分散接线的 prompt 整合到一处，避免 orchestrator 再写 inline prompt。
每个函数只负责：取模板 → format → 调 LLM → 回裁剪过的字符串。
失败时返回 fallback（或 None 让调用方决定）。
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.llm.models import get_chat_model, get_utility_model, invoke_json, invoke_text
from app.services.prompting.utils import EMPTY_RECENT_CONTEXT, pad_params, render_prompt

logger = logging.getLogger(__name__)

_pad_params = pad_params  # backward-compat alias inside module


async def _render_llm(
    prompt_key: str,
    params: dict[str, Any],
    *,
    max_chars: int = 120,
) -> str | None:
    """聊天模型文本回复：取 prompt → format → LLM → 裁剪。"""
    result = await render_prompt(
        prompt_key,
        params,
        lambda p: invoke_text(get_chat_model(), p),
        max_chars=max_chars,
    )
    return result or None


async def _classify_label(
    prompt_key: str,
    params: dict[str, Any],
    labels: tuple[str, ...],
) -> str | None:
    """小模型标签分类：在 labels 中选第一个命中的，未命中返回 None。"""
    raw = await render_prompt(
        prompt_key,
        params,
        lambda p: invoke_text(get_utility_model(), p),
        strip_split=False,
    )
    text = (raw or "").strip()
    if not text:
        return None
    for label in labels:
        if label in text:
            return label
    return None


# ═══════════════════════════════════════════════════════════════════
# §3.4 各意图回复
# ═══════════════════════════════════════════════════════════════════


async def apology_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    new_patience: int | None = None,
) -> str | None:
    """§3.4.4 道歉或承诺回复（接受用户道歉）。"""
    params: dict[str, Any] = {
        "message": message,
        "context": context or "(无)",
        "personality_brief": personality_brief or "真诚朋友",
        "user_portrait": user_portrait or "(未知)",
        **_pad_params(user_emotion),
    }
    if new_patience is not None:
        # 参考信息，可选
        params["personality_brief"] = (
            f"{params['personality_brief']}（当前耐心值 {new_patience}）"
        )
    return await _render_llm("boundary.apology_reply", params)


async def end_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
) -> str | None:
    """§3.4.6 终结意图回复。"""
    return await _render_llm(
        "intent.end_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            **_pad_params(user_emotion),
        },
        max_chars=60,
    )


async def schedule_query_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    current_activity: str = "",
    ai_schedule: str = "",
    ai_portrait: str = "",
) -> str | None:
    """§3.4.1 计划查询回复。"""
    return await _render_llm(
        "intent.schedule_query_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "current_activity": current_activity or "(未知)",
            "ai_schedule": ai_schedule or "(未知)",
            "ai_portrait": ai_portrait or "(未知)",
            **_pad_params(user_emotion),
        },
    )


async def schedule_adjust_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    current_activity: str = "",
    ai_schedule: str = "",
) -> dict[str, str] | None:
    """§3.4.2 作息调整回复（返回 JSON: reply + adjustment）。"""
    params = {
        "message": message,
        "context": context or "(无)",
        "personality_brief": personality_brief or "真诚朋友",
        "user_portrait": user_portrait or "(未知)",
        "current_activity": current_activity or "(未知)",
        "ai_schedule": ai_schedule or "(未知)",
        **_pad_params(user_emotion),
    }
    result = await render_prompt(
        "intent.schedule_adjust_reply",
        params,
        lambda p: invoke_json(get_chat_model(), p),
    )
    if not isinstance(result, dict):
        return None
    return {
        "reply": str(result.get("reply", "")).strip(),
        "adjustment": str(result.get("adjustment", "")).strip(),
    }


async def current_state_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    current_activity: str = "",
    ai_schedule: str = "",
) -> str | None:
    """§3.4.3 询问当前状态回复。"""
    return await _render_llm(
        "intent.current_state_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "current_activity": current_activity or "(未知)",
            "ai_schedule": ai_schedule or "(未知)",
            **_pad_params(user_emotion),
        },
    )


async def deletion_confirm_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    candidate_memories: str = "",
) -> str | None:
    """§3.4.7 删除意图确认（询问具体要删什么）。"""
    return await _render_llm(
        "intent.deletion_confirm",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "candidate_memories": candidate_memories or "(无)",
            **_pad_params(user_emotion),
        },
    )


async def deletion_done_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    deleted_memories: str = "",
) -> str | None:
    """§3.4.7 删除回复（表示已忘记）。"""
    return await _render_llm(
        "intent.deletion_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "deleted_memories": deleted_memories or "(无)",
            **_pad_params(user_emotion),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §4 日常交流分级回复
# ═══════════════════════════════════════════════════════════════════

async def memory_weak_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
) -> str | None:
    """§4 step 2：弱相关回复。人设【限定】已内联在 prompt 文本中。"""
    return await _render_llm(
        "memory.weak_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            **_pad_params(user_emotion),
        },
    )


async def memory_medium_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    user_memory: str = "",
    ai_memory: str = "",
) -> str | None:
    """§4 step 4：中相关回复。"""
    return await _render_llm(
        "memory.medium_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "user_memory": user_memory or "(无)",
            "ai_memory": ai_memory or "(无)",
            **_pad_params(user_emotion),
        },
    )


async def memory_strong_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    user_memory: str = "",
    ai_memory: str = "",
) -> str | None:
    """§4 step 5B：强相关回复（不需 L3）。"""
    return await _render_llm(
        "memory.strong_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "user_memory": user_memory or "(无)",
            "ai_memory": ai_memory or "(无)",
            **_pad_params(user_emotion),
        },
    )


async def memory_l3_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    l3_memory: str = "",
) -> str | None:
    """§4 step 5A / §3.4.5：久远记忆回复。"""
    return await _render_llm(
        "memory.l3_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "l3_memory": l3_memory or "(无)",
            **_pad_params(user_emotion),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §6 延迟解释
# ═══════════════════════════════════════════════════════════════════


async def delay_explanation_reply(
    *,
    received_time: str,
    current_time: str,
    activity: str,
    status: str,
    delay_minutes: int,
) -> str | None:
    """§6.5 延迟解释回复（独立一条，不拆分、不加 emoji/sticker）。"""
    return await _render_llm(
        "reply.delay_explanation",
        {
            "received_time": received_time,
            "current_time": current_time,
            "activity": activity or "处理自己的事",
            "status": status or "idle",
            "delay_minutes": str(delay_minutes),
        },
        max_chars=50,
    )


# ═══════════════════════════════════════════════════════════════════
# §3 意图识别（小模型）
# ═══════════════════════════════════════════════════════════════════

_INTENT_LABELS = {
    "终结意图", "计划查询", "作息调整", "询问当前状态",
    "道歉承诺", "删除", "调用久远记忆", "日常交流",
}


async def unified_intent_recognize(
    user_message: str,
    context: str = "",
) -> list[str]:
    """§3.3 step 1：统一意图识别，返回 spec 的 8 类标签（多选）。失败返回 ["日常交流"]。

    spec §3.3 要求 "输入用户消息**及上下文**". context 由调用方传入,
    一般格式是最近 N 轮对话的 "AI: ... / 用户: ..." 换行拼接.
    context 为空时 prompt 填 "(无)", 模型按单条消息判断.
    """
    raw = await render_prompt(
        "intent.unified",
        {"user_message": user_message, "context": context or "(无)"},
        lambda p: invoke_text(get_utility_model(), p),
        strip_split=False,
    )
    text = (raw or "").strip()
    if not text:
        return ["日常交流"]
    # 输出可能是顿号/逗号/换行分隔
    for sep in ("、", ",", "\n", "，"):
        if sep in text:
            parts = [p.strip() for p in text.split(sep)]
            break
    else:
        parts = [text]
    hits = [p for p in parts if p in _INTENT_LABELS]
    return hits or ["日常交流"]


async def split_multi_intent(user_message: str, intents: list[str]) -> dict[str, str]:
    """§3.3 step 2：多意图拆分。输入意图列表，输出 {intent: text_fragment}。"""
    if len(intents) <= 1:
        return {intents[0] if intents else "日常交流": user_message}
    result = await render_prompt(
        "intent.split",
        {"user_message": user_message, "intents": "、".join(intents)},
        lambda p: invoke_json(get_utility_model(), p),
    )
    if isinstance(result, dict):
        fragments = {
            k: str(v) for k, v in result.items()
            if k in _INTENT_LABELS and str(v).strip()
        }
        if fragments:
            return fragments
    return {intents[0]: user_message}


# ═══════════════════════════════════════════════════════════════════
# §2.6 步骤 3/4：违禁/攻击目标识别（小模型上热路径用）
# ═══════════════════════════════════════════════════════════════════


async def banned_word_check(message: str) -> bool | None:
    """§2.6 步骤 3：小模型违禁判断。返回 True/False/None（歧义）。"""
    label = await _classify_label("boundary.banned_word", {"message": message}, ("是", "否"))
    if label == "是":
        return True
    if label == "否":
        return False
    return None


async def attack_target_classify(
    message: str, recent_context: str = EMPTY_RECENT_CONTEXT,
) -> str | None:
    """§2.6 步骤 4：攻击目标识别。攻击AI / 攻击第三方 / 无负面意图 / 无目标脏话.

    `recent_context`: 最近几轮对话, 帮 LLM 在连续骂战里识别"你"指 AI
    (e.g. 用户上轮骂"你个傻逼", 这轮"不然呢" 也是攻击 AI).
    """
    return await _classify_label(
        "boundary.attack_target",
        {"message": message, "recent_context": recent_context},
        ("攻击AI", "攻击第三方", "无负面意图", "无目标脏话"),
    )


async def attack_level_classify(message: str) -> str | None:
    """§2.6 步骤 5：攻击级别识别。K1 / K2 / K3。"""
    # K3 在前防子串 "K1" 匹配到 "K13" 这类异常输出
    return await _classify_label(
        "boundary.attack_level", {"message": message}, ("K3", "K2", "K1"),
    )


async def l3_trigger_classify(message: str) -> str:
    """§4 step 5.1-5.2：「调用L3」判断。失败保守返回 "无"。"""
    return await _classify_label(
        "memory.l3_trigger",
        {"message": message},
        ("不满纠正", "请求更久", "无"),
    ) or "无"


async def ai_reply_emotion(reply_text: str) -> dict:
    """spec §5 step 1：调小模型「AI语句情绪」prompt 解析 AI 回复的情绪。

    返回 `{"emotion": str, "intensity": int}`；任一字段缺失或调用失败返回 `{}`。
    """
    if not reply_text or not reply_text.strip():
        return {}
    result = await render_prompt(
        "reply.emotion_detection",
        {"ai_reply": reply_text},
        lambda p: invoke_json(get_utility_model(), p),
    )
    if not isinstance(result, dict):
        return {}
    emotion = str(result.get("emotion", "")).strip()
    if not emotion:
        return {}
    try:
        intensity = int(result.get("intensity", 0))
    except (TypeError, ValueError):
        intensity = 0
    return {"emotion": emotion, "intensity": intensity}


async def split_reply_to_n_sentences(original_reply: str, n: int) -> list[str] | None:
    """§5.5：小模型「AI语句拆分（2/3句）」。失败/不足 n 条返回 None 让调用方兜底。"""
    if n not in (2, 3):
        return None
    raw = await render_prompt(
        f"reply.split_{n}",
        {"original_reply": original_reply},
        lambda p: invoke_text(get_utility_model(), p),
        strip_split=False,
    )
    text = (raw or "").strip()
    if not text:
        return None
    # spec §5.5：小模型输出换行分隔；可能带序号前缀或用 || 混写
    parts = [
        p.strip().lstrip("0123456789.、) ").strip()
        for p in text.split("\n") if p.strip()
    ]
    if len(parts) < n and "||" in text:
        parts = [p.strip() for p in text.split("||") if p.strip()]
    return parts[:n] if len(parts) >= n else None


__all__ = [
    "apology_reply",
    "end_reply",
    "schedule_query_reply",
    "schedule_adjust_reply",
    "current_state_reply",
    "deletion_confirm_reply",
    "deletion_done_reply",
    "memory_weak_reply",
    "memory_medium_reply",
    "memory_strong_reply",
    "memory_l3_reply",
    "delay_explanation_reply",
    "unified_intent_recognize",
    "split_multi_intent",
    "banned_word_check",
    "attack_target_classify",
    "attack_level_classify",
    "l3_trigger_classify",
    "split_reply_to_n_sentences",
    "ai_reply_emotion",
]
