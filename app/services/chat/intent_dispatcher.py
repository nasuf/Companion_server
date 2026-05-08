"""统一意图识别模块。

单次关键词扫描，按优先级返回最高匹配的意图。
边界检查（boundary）在本模块之前独立处理，不纳入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.observability.events import EVT_INTENT_DETECTED, EVT_LLM_FAIL
from app.services.chat.conversation_end import CONVERSATION_END_KEYWORDS
from app.services.memory.interaction.deletion import DELETION_KEYWORDS
from app.services.interaction.boundary import APOLOGY_KEYWORDS
from app.services.schedule_domain.schedule import resolve_schedule_query_scope


class IntentType(Enum):
    # spec §3.3 将"道歉承诺"合并为一个意图
    APOLOGY_PROMISE = "apology_promise"
    CONVERSATION_END = "conversation_end"
    DELETION = "deletion"
    SCHEDULE_ADJUST = "schedule_adjust"
    SCHEDULE_QUERY = "schedule_query"
    CURRENT_STATE = "current_state"  # spec §3.4.3 询问当前状态
    L3_RECALL = "l3_recall"          # spec §3.4.5 调用久远记忆
    # 工程扩展 (Phase 3): 用户请求 AI 记一件事 / 设提醒 / 告知日历事件.
    # 跟"计划查询"分离让回复体感正确 ("好嘞我帮你记上了" vs 错误的"你明天要做X").
    # 详见 CLAUDE.md §6 偏离表.
    RECORD_REQUEST = "record_request"
    # 工程扩展 (P0 危机安全网): 用户消息含自伤/极端念头/对生命负面想法等求救信号.
    # 走独立 short-circuit handler (handle_crisis), 用专属 minimal prompt + 用户
    # 记忆 fetch, 完全切掉主路径 system_prompt 14 段干扰. 触发由 orchestrator
    # 关键字层 (_is_crisis_message) 强制 force, 不依赖 intent LLM (实证 LLM 把
    # "我想跳楼" 误归"询问当前状态"). 详见 CLAUDE.md §6 偏离表 + handle_crisis 注释.
    CRISIS = "crisis"
    NONE = "none"


# spec §3.3 中文标签 → IntentType 映射
LABEL_TO_INTENT: dict[str, IntentType] = {
    "危机求助": IntentType.CRISIS,
    "终结意图": IntentType.CONVERSATION_END,
    "计划查询": IntentType.SCHEDULE_QUERY,
    "作息调整": IntentType.SCHEDULE_ADJUST,
    "询问当前状态": IntentType.CURRENT_STATE,
    "道歉承诺": IntentType.APOLOGY_PROMISE,
    "删除": IntentType.DELETION,
    "调用久远记忆": IntentType.L3_RECALL,
    "记录请求": IntentType.RECORD_REQUEST,
    "日常交流": IntentType.NONE,
}

# spec §3.3 multi-intent 优先级（高优先级先处理；用于 primary 选择和 sub-intent 处理顺序）
# 危机求助排首位 — 即使 LLM 把它跟其他意图同时识别 (e.g. "我活不下去了, 算了不聊了"
# 误打成"危机+终结"), 也必须 crisis 主路径胜出, 用户安全永远在前.
# 记录请求排在删除之后但在其他短路意图之前: 用户同时说 "记得 X" + 别的意图时,
# 先确认记下 (体感优先), 再处理别的; 但 "忘了 X" 仍优先以避免冲突.
INTENT_PRIORITY: list[str] = [
    "危机求助", "删除", "记录请求", "作息调整", "终结意图", "计划查询",
    "询问当前状态", "道歉承诺", "调用久远记忆", "日常交流",
]


@dataclass
class IntentResult:
    intent: IntentType = IntentType.NONE
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


PROMISE_KEYWORDS = [
    "我以后不会了", "我保证", "我发誓", "不会再这样", "再也不会",
    "保证不再", "我承诺", "绝对不会", "我答应你",
]

SCHEDULE_ADJUST_KEYWORDS = [
    "你能不能晚点睡", "晚点睡", "早点睡", "早点休息", "别睡了",
    "能不能抽空", "陪我聊", "别忙了", "你先别忙", "能不能陪我",
    "你今天早点", "今天晚点",
]

_CURRENT_STATE_FAST_PHRASES = frozenset({
    "你在干嘛",
    "你在干嘛呢",
    "在干嘛",
    "在干嘛呢",
    "干嘛呢",
    "你干嘛呢",
    "忙啥",
    "你忙啥",
    "在忙啥",
    "你在忙啥",
    "在忙什么",
    "你在忙什么",
    "现在忙吗",
    "你现在忙吗",
    "你现在在干嘛",
    "你现在在干嘛呢",
    "你现在做什么",
    "你现在在做什么",
    "你在做什么",
    "你在做啥",
    "你现在干啥",
    "你现在干啥呢",
})

_CURRENT_STATE_TIME_BLOCKERS = (
    "明天", "后天", "大后天", "昨天", "前天", "今晚", "晚上", "下午",
    "上午", "中午", "凌晨", "周末", "星期", "礼拜", "几点", "什么时候",
    "等会", "待会", "一会", "以后", "之前", "刚才",
)
_CURRENT_STATE_HISTORY_BLOCKERS = (
    "刚才", "刚刚", "刚说", "刚说的", "之前说", "前面说", "上一句", "上句",
)
_CURRENT_STATE_EXPLICIT_PHRASES = frozenset({
    "忙吗",
    "不忙吗",
    "你忙吗",
    "你不忙吗",
    "有空吗",
    "你有空吗",
    "现在有空吗",
    "你现在有空吗",
    "最近怎么样",
    "你最近怎么样",
    "你怎么样",
    "你还好吗",
    "你开心吗",
    "你心情怎么样",
})
_CURRENT_STATE_SUBJECT_TERMS = ("你", "你现在", "你最近", "你今天", "你那边")
_CURRENT_STATE_PREDICATE_TERMS = (
    "干嘛", "做什么", "做啥", "忙", "有空", "怎么样", "还好吗",
    "心情", "感觉", "开心", "难过", "状态",
)

def detect_current_state_fast_path(message: str) -> bool:
    """Hot-path for common "what are you doing now" messages.

    Keep this deliberately narrow. Future/past/scheduled variants still go
    through the LLM classifier so they can become schedule queries when needed.
    """
    normalized = re.sub(r"[\s，。！？!?~～…,.、]+", "", message.strip())
    if not normalized:
        return False
    if any(word in normalized for word in _CURRENT_STATE_TIME_BLOCKERS):
        return False
    return normalized in _CURRENT_STATE_FAST_PHRASES


def is_explicit_current_state_query(message: str) -> bool:
    """Return true only for direct questions about AI's current/recent state.

    The LLM intent classifier can over-label follow-ups about the previous AI
    reply as CURRENT_STATE. This gate keeps the short-circuit narrow so
    context follow-ups fall back to the normal reply path.
    """
    normalized = re.sub(r"[\s，。！？!?~～…,.、]+", "", message.strip())
    if not normalized:
        return False
    if any(word in normalized for word in _CURRENT_STATE_HISTORY_BLOCKERS):
        return False
    if detect_current_state_fast_path(message):
        return True
    if normalized in _CURRENT_STATE_EXPLICIT_PHRASES:
        return True
    if not any(term in normalized for term in _CURRENT_STATE_SUBJECT_TERMS):
        return False
    return any(term in normalized for term in _CURRENT_STATE_PREDICATE_TERMS)


def infer_schedule_query_type(message: str, *, require_query_cue: bool = True) -> str | None:
    """Infer schedule query subtype through the schedule-domain scope parser."""
    scope = resolve_schedule_query_scope(message, require_query_cue=require_query_cue)
    return scope.query_type if scope else None

# 优先级顺序：前面的优先匹配。spec §3.3 道歉承诺合并为一个意图
_CHECKS: list[tuple[IntentType, list[str]]] = [
    (IntentType.APOLOGY_PROMISE, APOLOGY_KEYWORDS + PROMISE_KEYWORDS),
    (IntentType.CONVERSATION_END, CONVERSATION_END_KEYWORDS),
    (IntentType.DELETION, DELETION_KEYWORDS),
    (IntentType.SCHEDULE_ADJUST, SCHEDULE_ADJUST_KEYWORDS),
]


def detect_intent(message: str, patience_zone: str = "normal") -> IntentResult:
    """单次关键词扫描，返回最高优先级的匹配意图。

    Args:
        message: 用户消息文本
        patience_zone: 当前耐心区间

    Returns:
        IntentResult
    """
    msg = message.strip()

    # 按优先级扫描固定关键词列表
    for intent_type, keywords in _CHECKS:
        if any(kw in msg for kw in keywords):
            return IntentResult(intent=intent_type, confidence=0.8)

    # SCHEDULE_QUERY：结构化关键词，返回 query_type
    query_type = infer_schedule_query_type(msg)
    if query_type:
        return IntentResult(
            intent=IntentType.SCHEDULE_QUERY,
            confidence=0.7,
            metadata={"query_type": query_type},
        )

    return IntentResult()


async def detect_intent_unified(
    message: str,
    context: str = "",
) -> IntentResult:
    """spec §3.3 step 1-2 标准入口：始终调小模型统一识别 + 多意图拆分。

    `context` 为 spec §3.3 step 1 要求的"对话上下文"，格式由调用方组装，
    一般是最近 N 轮 "AI: ... / 用户: ..." 换行拼接。传空串时 LLM 回退到
    "仅凭当前消息判断"（首轮对话场景）。

    fallback 关键词扫描**仅在 LLM 异常时**触发. **不在** LLM 返 NONE 时触发 —
    NONE 是 LLM 的合法结果, 表示"日常交流, 没特殊意图". 之前的实现把 NONE 也当
    "弱信号"落到 keyword scan, 导致用户随口说"明天是我的生日" → keyword scan
    匹配 _SCHEDULE_QUERY_KEYWORDS['date'] 含 "明天" → 错路由到 SCHEDULE_QUERY
    (生产 bug 复现 2026-05-03 trace 019dec46: AI 跑去播报自己日程).
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        result = await detect_intent_llm(message, context=context)
        logger.info(
            f"[INTENT-LLM] intent={result.intent.name} confidence={result.confidence}",
            extra={
                "event": EVT_INTENT_DETECTED,
                "intent_primary": result.intent.name,
                "confidence": result.confidence,
                "intent_labels": (result.metadata or {}).get("llm_labels"),
                "msg_len": len(message),
            },
        )
        return result
    except Exception as e:
        logger.warning(
            f"Unified LLM intent failed, fallback to keyword: {e}",
            extra={"event": EVT_LLM_FAIL, "stage": "intent_unified", "error_type": type(e).__name__},
        )
        return detect_intent(message)


async def detect_intent_llm(message: str, *, context: str = "") -> IntentResult:
    """Spec §3.3 step 1-2 — LLM 统一意图识别 + 多意图拆分。

    结果命中多个意图时，优先级：
      删除 > 作息调整 > 终结意图 > 计划查询 > 询问当前状态 > 道歉承诺 > 调用久远记忆 > 日常交流
    """
    # 延迟导入避免循环依赖（intent_replies 依赖 prompting.store）
    from app.services.chat.intent_replies import unified_intent_recognize, split_multi_intent

    # LLM 异常**不要**在这吞 — caller (detect_intent_unified) 据此区分:
    # - LLM 异常 → caller fallback 到 keyword scan
    # - LLM 返 "日常交流" / 空 labels → 是合法的 NONE, caller 不该 fallback
    # (生产 bug 复现 2026-05-03 trace 019dec46: 之前这里吞掉异常返 NONE, caller
    # 把 NONE 当 "弱信号" 又跑 keyword scan, 用户 "明天是我的生日" 被 "明天" 关键词
    # 错路由到 SCHEDULE_QUERY)
    labels = await unified_intent_recognize(message, context=context)

    if not labels or labels == ["日常交流"]:
        return IntentResult()

    # 多意图：调 split 以便下游可按 fragment 处理.
    # 把 context 传给 split — spec §3.3 字面只输入"用户原话", 但生产场景的
    # 口语融合句拆分依赖上下文 (e.g. "算了别提醒了" 在"刚设了提醒"语境下应
    # 整句给 RECORD_REQUEST, 不该拆给"计划查询"). 详见 CLAUDE.md §6 偏离表.
    fragments: dict[str, str] | None = None
    if len(labels) > 1:
        try:
            fragments = await split_multi_intent(message, labels, context=context)
        except Exception:
            fragments = None

    primary = next((l for l in INTENT_PRIORITY if l in labels), labels[0])
    intent = LABEL_TO_INTENT.get(primary, IntentType.NONE)
    if intent == IntentType.NONE:
        return IntentResult()

    metadata: dict[str, Any] = {"llm_labels": labels}
    if intent == IntentType.SCHEDULE_QUERY:
        metadata["query_type"] = infer_schedule_query_type(
            message, require_query_cue=False,
        ) or "current"
    if fragments:
        metadata["fragments"] = fragments
    return IntentResult(intent=intent, confidence=0.75, metadata=metadata)
