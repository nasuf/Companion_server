"""Spec §2-§6 意图/边界/记忆/延迟 各分级 prompt 的统一调用入口。

把 12 个已注册但分散接线的 prompt 整合到一处，避免 orchestrator 再写 inline prompt。
每个函数只负责：取模板 → format → 调 LLM → 回裁剪过的字符串。
失败时返回 fallback（或 None 让调用方决定）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.services.llm.models import get_chat_model, get_utility_model, invoke_json, invoke_text
from app.services.prompting.utils import EMPTY_RECENT_CONTEXT, render_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class L3TriggerResult:
    label: str
    retrieval_query: str = ""


_OPTIONAL_REFERENCE_KEYS = frozenset({
    "context",
    "user_portrait",
    "user_memory",
    "ai_memory",
    "candidate_memories",
    "deleted_memories",
    "l3_memory",
    "ai_schedule",
    "current_activity",
})


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
        optional_keys=_OPTIONAL_REFERENCE_KEYS,
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
            "user_portrait": user_portrait or "",
            "current_activity": current_activity or "",
            "ai_schedule": ai_schedule or "(未知)",
            "ai_portrait": ai_portrait or "(未知)",
        },
        # 默认 120 太紧 — 该 prompt 要求 LLM "回答 + 延展话题", LLM 自然产 130-200
        # 字回复. 之前 120 cap 经常切掉延展部分 (生产 bug 2026-05-03 trace 019decd3:
        # 邀请的"我肯定去"答案被切掉, 留下来的全是 AI 自言自语). 180 留余量, 仍由
        # 主回复管道的 truncate_at_sentence 保底句末切.
        max_chars=180,
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
    }
    result = await render_prompt(
        "intent.schedule_adjust_reply",
        params,
        lambda p: invoke_json(get_chat_model(), p),
        optional_keys=_OPTIONAL_REFERENCE_KEYS,
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
            "user_portrait": user_portrait or "",
            "current_activity": current_activity or "(未知)",
            "ai_schedule": ai_schedule or "",
        },
    )


async def crisis_reply(
    *,
    message: str,
    context: str = "",
    personality_brief: str = "",
    user_portrait: str = "",
    user_memory: str = "",
) -> str | None:
    """P0 危机安全网回复 (handle_crisis).

    切掉主 system_prompt 14 段, 只用 intent.crisis_reply 单 prompt 生成. user_memory
    传入用户 L1/L2 中跟情绪/求助/边界相关的条目, 帮 LLM 知道这是不是 Ta 第一次说.
    max_chars 给到 200 (比普通 *_reply 的 120 大), 因为 crisis 回复需要 1) 接住情绪
    2) 想了解 3) 柔和提议求助 — 三步压在 100 字内会被截断成机械回复.
    """
    return await _render_llm(
        "intent.crisis_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "user_memory": user_memory or "(无)",
        },
        max_chars=200,
    )


async def crisis_followup_reply(
    *,
    message: str,
    context: str = "",
    personality_brief: str = "",
    user_portrait: str = "",
    user_memory: str = "",
    safety_check_instruction: str = "",
) -> str | None:
    """危机后一段时间内的普通追问仍需安全优先。"""
    return await _render_llm(
        "intent.crisis_followup_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "user_memory": user_memory or "(无)",
            "safety_check_instruction": (
                safety_check_instruction
                or "本轮不主动复核安全状态; 除非用户主动提到风险, 不要追问安全。"
            ),
        },
        max_chars=200,
    )


async def crisis_followup_classify(
    *,
    message: str,
    context: str = "",
) -> str:
    """Return guard/release for recent-crisis follow-up state."""
    result = await render_prompt(
        "intent.crisis_followup_classify",
        {
            "message": message,
            "context": context or "(无)",
        },
        lambda p: invoke_json(get_utility_model(), p),
    )
    if isinstance(result, dict):
        status = str(result.get("status", "")).strip().lower()
        if status in {"guard", "release"}:
            return status
    return "guard"


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
        },
    )


# 反问时间硬 timeout. RECORD_REQUEST parse 失败时调 LLM 反问, 但 utility model
# 卡 50s 会让用户等到天荒地老 (RECORD_REQUEST 短路本来 ~3s 体感). 5s 上限,
# 失败/超时 fallback 固定模板. 同 reminder_pre_check 的容错策略.
_ASK_TIME_TIMEOUT_SEC = 5.0
_ASK_TIME_FALLBACK = "嗯嗯, 大概什么时候呀? 我好定准点的~"


async def record_ask_time(
    *,
    user_message: str,
    personality_brief: str = "",
) -> str:
    """RECORD_REQUEST 时间解析失败 → 反问让用户补全时间.

    LLM 针对用户漏掉的时间粒度反问 ("下周提醒我X" → "下周哪天?"; "提醒我Y" → "啥时候?").
    LLM 失败 / 超时 / 输出明显异常 → fallback 固定模板, 保证用户**总能拿到一个反问**.
    """
    import asyncio as _asyncio

    async def _do_ask() -> str | None:
        return await _render_llm(
            "intent.record_ask_time",
            {
                "user_message": user_message[:120] or "(空)",
                "personality_brief": personality_brief or "真诚朋友",
            },
            max_chars=60,
        )

    try:
        result = await _asyncio.wait_for(
            _do_ask(), timeout=_ASK_TIME_TIMEOUT_SEC,
        )
    except (_asyncio.TimeoutError, Exception) as e:
        logger.info(
            f"record_ask_time timeout/error after "
            f"{_ASK_TIME_TIMEOUT_SEC}s ({type(e).__name__}); "
            "falling back to fixed template"
        )
        return _ASK_TIME_FALLBACK

    text = (result or "").strip()
    # LLM 跑题守门: 长度异常 / 缺问号 (反问句必含 "?" 或 "?") → fallback
    if not text or len(text) > 50 or not any(q in text for q in ("?", "？")):
        return _ASK_TIME_FALLBACK
    return text


async def record_confirm_reply(
    *,
    summary: str,
    when_text: str = "",
    is_recurring: bool = False,
    personality_brief: str = "",
) -> str | None:
    """RECORD_REQUEST 短路确认回复. spec §4.2 工程扩展, 见 CLAUDE.md §6 偏离表.
    用于 "提醒我X" / "记得X" / "X 月 X 日是我生日" 类语句的体感确认 (1-2 句, 不延展)."""
    return await _render_llm(
        "intent.record_confirm_reply",
        {
            "personality_brief": personality_brief or "真诚朋友",
            "summary": summary or "(未指定事项)",
            "when_text": when_text or "(未指定时间)",
            "is_recurring": "是" if is_recurring else "否",
        },
        max_chars=80,
    )


# pre-check 是 reminder 触发的"在路径上"LLM 调用. 默认 LLM 链路 (resilience)
# 12s timeout × 3 attempts + ollama fallback, 整体 ~50s, 直接拉爆提醒延迟
# (生产观察: 用户期望 17:10:18 提醒, 实际 17:11:21 才发, 多花的 ~50s
# 全是 pre-check LLM 卡住). 给 pre-check 一个短 timeout, 失败立刻 fallback
# 到"needed" — 保守语义"不知道用户是否已做, 照常提醒不漏".
_REMINDER_PRECHECK_TIMEOUT_SEC = 5.0


async def reminder_pre_check(
    *,
    summary: str,
    trigger_time: str,
    recent_messages: str = "",
) -> dict:
    """统一提醒触发前的状态判别 (completed/cancelled/rescheduled/needed).

    返回形如 `{"state": "...", "new_time": "ISO|None", "reason": "..."}`.
    LLM 失败 / 超时 / 输出非 JSON → fallback 到 "needed"
    (保守语义: 照常发送提醒, 宁可让用户被多提醒一次也不漏掉).

    硬 timeout 5s: pre-check 是触发热路径, 长 LLM 卡顿直接拉爆体感延迟.
    """
    import asyncio as _asyncio

    fallback = {"state": "needed", "new_time": None, "reason": "llm_fallback"}
    if not summary:
        return fallback

    async def _do_check() -> dict | None:
        return await render_prompt(
            "proactive.reminder_pre_check",
            {
                "summary": summary,
                "trigger_time": trigger_time or "(未知)",
                "recent_messages": recent_messages or "(无)",
            },
            lambda p: invoke_json(get_utility_model(), p),
        )

    try:
        result = await _asyncio.wait_for(
            _do_check(), timeout=_REMINDER_PRECHECK_TIMEOUT_SEC,
        )
    except (_asyncio.TimeoutError, Exception) as e:
        logger.info(
            f"reminder_pre_check timeout/error after "
            f"{_REMINDER_PRECHECK_TIMEOUT_SEC}s ({type(e).__name__}); "
            "falling back to 'needed' (will fire reminder)"
        )
        return fallback

    if not isinstance(result, dict):
        return fallback
    state = str(result.get("state", "")).strip().lower()
    if state not in {"completed", "cancelled", "rescheduled", "needed"}:
        return fallback
    raw_new = result.get("new_time")
    new_time = str(raw_new).strip() if raw_new not in (None, "", "null") else None
    return {
        "state": state,
        "new_time": new_time,
        "reason": str(result.get("reason", "")).strip()[:32],
    }


async def reminder_message(
    *,
    summary: str,
    personality_brief: str = "",
) -> str | None:
    """统一提醒系统的实际触发消息. 替代旧 special_reminder 路径 (后者保留给节日/生日)."""
    return await _render_llm(
        "proactive.reminder_message",
        {
            "personality_brief": personality_brief or "真诚朋友",
            "summary": summary or "(未指定事项)",
        },
        max_chars=80,
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
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §4 日常交流分级回复
# ═══════════════════════════════════════════════════════════════════

# Phase: tier reply 也支持 random 1-3 条 (跟主回复一致, 提升微信多条体感).
# 调用方 (reply_generate) 传 n/max_per/max_total, 拿到含 || 的输出后走
# split_and_validate_replies 切. strip_split=False 保留 ||, max_chars=max_total
# 防"句1||句2||句3" 超 120 默认被截.


async def _render_tier_reply(
    prompt_key: str,
    params: dict[str, Any],
    *,
    max_total: int,
) -> str | None:
    """tier reply 专用 render: strip_split=False (允许 ||), max_chars=max_total."""
    return await render_prompt(
        prompt_key,
        params,
        lambda p: invoke_text(get_chat_model(), p),
        max_chars=max_total,
        strip_split=False,
        optional_keys=_OPTIONAL_REFERENCE_KEYS,
    )


async def memory_weak_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    n: int = 1,
    max_per: int = 60,
    max_total: int = 150,
) -> str | None:
    """§4 step 2：弱相关回复. n=1 单条, n≥2 按 prompt 内 || 指令多条."""
    return await _render_tier_reply(
        "memory.weak_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "n": n,
            "max_per": max_per,
            "total": max_total,
        },
        max_total=max_total,
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
    n: int = 1,
    max_per: int = 60,
    max_total: int = 150,
) -> str | None:
    """§4 step 4：中相关回复. n=1 单条, n≥2 按 prompt 内 || 指令多条."""
    return await _render_tier_reply(
        "memory.medium_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "user_memory": user_memory or "(无)",
            "ai_memory": ai_memory or "(无)",
            "n": n,
            "max_per": max_per,
            "total": max_total,
        },
        max_total=max_total,
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
    n: int = 1,
    max_per: int = 60,
    max_total: int = 150,
) -> str | None:
    """§4 step 5B：强相关回复. n=1 单条, n≥2 按 prompt 内 || 指令多条."""
    return await _render_tier_reply(
        "memory.strong_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "user_memory": user_memory or "(无)",
            "ai_memory": ai_memory or "(无)",
            "n": n,
            "max_per": max_per,
            "total": max_total,
        },
        max_total=max_total,
    )


async def memory_l3_reply(
    *,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    l3_memory: str = "",
    n: int = 1,
    max_per: int = 60,
    max_total: int = 150,
) -> str | None:
    """§4 step 5A：久远记忆回复. n=1 单条, n≥2 按 prompt 内 || 指令多条."""
    return await _render_tier_reply(
        "memory.l3_reply",
        {
            "message": message,
            "context": context or "(无)",
            "personality_brief": personality_brief or "真诚朋友",
            "user_portrait": user_portrait or "(未知)",
            "l3_memory": l3_memory or "(无)",
            "n": n,
            "max_per": max_per,
            "total": max_total,
        },
        max_total=max_total,
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
    "道歉承诺", "删除", "调用久远记忆", "记录请求", "日常交流",
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


async def split_multi_intent(
    user_message: str,
    intents: list[str],
    context: str = "",
) -> dict[str, str]:
    """§3.3 step 2：多意图拆分。输入意图列表，输出 {intent: text_fragment}。

    `context` 是 spec §3.3 偏离 (字面只输入用户原话). 生产实测口语融合句
    "算了别提醒了，我吃过了" 字面拆出来的子片段意图跟整句相反, 加上下文让
    LLM 看到"用户在管理之前刚设的提醒", 拆分时不至于乱分到"计划查询"等无关
    intent. context 为空时退化到 spec 字面行为.
    """
    if len(intents) <= 1:
        return {intents[0] if intents else "日常交流": user_message}
    result = await render_prompt(
        "intent.split",
        {
            "user_message": user_message,
            "intents": "、".join(intents),
            "context": context or "(无)",
        },
        lambda p: invoke_json(get_utility_model(), p),
    )
    if isinstance(result, dict):
        # 空串表示"该 intent 已被主意图消化完, 不需要单独处理" (新 prompt 规则).
        # 之前是 `if str(v).strip()` 过滤掉空串, 现在保持一致 — 空串不进 fragments,
        # orchestrator 端就少跑一个 sub-intent.
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


async def positive_interaction_check(message: str) -> bool:
    """spec §2.5 正向互动: 小模型判定用户消息是否表达感谢/善意/积极反馈/正向情绪.
    返回 True 才让 +20 耐心恢复生效, 防中性应答 (嗯/哦/好) 也加分.
    LLM 失败/歧义 → 保守返 False, 不发放恢复.
    """
    label = await _classify_label(
        "boundary.positive_interaction", {"message": message}, ("是", "否"),
    )
    return label == "是"


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


def _parse_l3_trigger_result(raw: Any) -> L3TriggerResult:
    labels = ("不满纠正", "请求更久", "无")
    label = ""
    retrieval_query = ""
    if isinstance(raw, dict):
        label = str(raw.get("label", "")).strip()
        retrieval_query = str(raw.get("retrieval_query", "")).strip()
    elif isinstance(raw, str):
        text = raw.strip()
        for candidate in labels:
            if candidate in text:
                label = candidate
                break
    if label not in labels:
        label = "无"
    if label == "无":
        retrieval_query = ""
    if len(retrieval_query) > 50:
        retrieval_query = retrieval_query[:50]
    return L3TriggerResult(label=label, retrieval_query=retrieval_query)


async def l3_trigger_analyze(
    message: str,
    recent_context: str = EMPTY_RECENT_CONTEXT,
) -> L3TriggerResult:
    """§4 step 5.1-5.2：L3 唤醒判断 + 同源检索 query。"""
    params = {
        "message": message,
        "recent_context": recent_context or EMPTY_RECENT_CONTEXT,
    }
    raw = await render_prompt(
        "memory.l3_trigger",
        params,
        lambda p: invoke_json(get_utility_model(), p),
    )
    result = _parse_l3_trigger_result(raw)
    if result.label != "无":
        return result

    # Backward compatibility for deployed/custom prompts that still emit a bare label.
    label = await _classify_label(
        "memory.l3_trigger",
        params,
        ("不满纠正", "请求更久", "无"),
    )
    if label and label != "无":
        return L3TriggerResult(label=label)
    return result


async def l3_trigger_classify(
    message: str,
    recent_context: str = EMPTY_RECENT_CONTEXT,
) -> str:
    """§4 step 5.1-5.2：「调用L3」判断。失败保守返回 "无"。"""
    return (await l3_trigger_analyze(message, recent_context)).label


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


# Phase: 删除 split_reply_to_n_sentences 整层 — 拆分应由主回复 LLM 自己按
# prompt 内 "{n} 条 || 分隔" 指令完成. 历史额外调小模型拆分有两个 bug:
# 1. lstrip("0123456789.、) ") 字符集贪心 → "12月8号" 首字符 "1"/"2" 被误删 (trace 019e0116)
# 2. prompt 含 "可适当扩写" → LLM 看 🎂 自作主张编出"祝你生日快乐" (5 月份语境错)
# 移除整层后, 主 LLM 给单句就单条, 给"句1||句2"就两条 — 信任 LLM 输出, 不再
# 二次调用. 其他路径 (tier reply / intent reply / contradiction / boundary)
# 本就是单条短回复, 不需要拆分.


__all__ = [
    "apology_reply",
    "end_reply",
    "schedule_query_reply",
    "schedule_adjust_reply",
    "current_state_reply",
    "crisis_reply",
    "crisis_followup_reply",
    "crisis_followup_classify",
    "deletion_confirm_reply",
    "deletion_done_reply",
    "record_confirm_reply",
    "record_ask_time",
    "reminder_pre_check",
    "reminder_message",
    "memory_weak_reply",
    "memory_medium_reply",
    "memory_strong_reply",
    "memory_l3_reply",
    "delay_explanation_reply",
    "unified_intent_recognize",
    "split_multi_intent",
    "banned_word_check",
    "positive_interaction_check",
    "attack_target_classify",
    "attack_level_classify",
    "L3TriggerResult",
    "l3_trigger_analyze",
    "l3_trigger_classify",
    "ai_reply_emotion",
]
