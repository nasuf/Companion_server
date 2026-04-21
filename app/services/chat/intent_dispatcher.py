"""统一意图识别模块。

单次关键词扫描，按优先级返回最高匹配的意图。
边界检查（boundary）在本模块之前独立处理，不纳入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.services.chat.conversation_end import CONVERSATION_END_KEYWORDS
from app.services.memory.interaction.deletion import DELETION_KEYWORDS
from app.services.interaction.boundary import APOLOGY_KEYWORDS
from app.services.schedule_domain.schedule import _SCHEDULE_QUERY_KEYWORDS


class IntentType(Enum):
    # spec §3.3 将"道歉承诺"合并为一个意图
    APOLOGY_PROMISE = "apology_promise"
    CONVERSATION_END = "conversation_end"
    DELETION = "deletion"
    SCHEDULE_ADJUST = "schedule_adjust"
    SCHEDULE_QUERY = "schedule_query"
    CURRENT_STATE = "current_state"  # spec §3.4.3 询问当前状态
    L3_RECALL = "l3_recall"          # spec §3.4.5 调用久远记忆
    NONE = "none"


# spec §3.3 中文标签 → IntentType 映射
LABEL_TO_INTENT: dict[str, IntentType] = {
    "终结意图": IntentType.CONVERSATION_END,
    "计划查询": IntentType.SCHEDULE_QUERY,
    "作息调整": IntentType.SCHEDULE_ADJUST,
    "询问当前状态": IntentType.CURRENT_STATE,
    "道歉承诺": IntentType.APOLOGY_PROMISE,
    "删除": IntentType.DELETION,
    "调用久远记忆": IntentType.L3_RECALL,
    "日常交流": IntentType.NONE,
}

# spec §3.3 multi-intent 优先级（高优先级先处理；用于 primary 选择和 sub-intent 处理顺序）
INTENT_PRIORITY: list[str] = [
    "删除", "作息调整", "终结意图", "计划查询",
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
    for query_type, keywords in _SCHEDULE_QUERY_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            return IntentResult(
                intent=IntentType.SCHEDULE_QUERY,
                confidence=0.7,
                metadata={"query_type": query_type},
            )

    return IntentResult()


async def detect_intent_unified(message: str) -> IntentResult:
    """spec §3.3 step 1-2 标准入口：始终调小模型统一识别 + 多意图拆分。

    LLM 失败时 fallback 到关键字扫描（保底单意图）。极短消息（≤4 字符）
    应在调用前由调用方决定是否绕过此函数（"嗯"/"好" 这类无价值召回）。
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        result = await detect_intent_llm(message)
        if result.intent != IntentType.NONE:
            return result
    except Exception as e:
        logger.warning(f"Unified LLM intent failed, fallback to keyword: {e}")
    return detect_intent(message)


async def detect_intent_llm(message: str) -> IntentResult:
    """Spec §3.3 step 1-2 — LLM 统一意图识别 + 多意图拆分。

    只有关键词扫描落空时才调。结果命中多个意图时，优先级：
      删除 > 作息调整 > 终结意图 > 计划查询 > 询问当前状态 > 道歉承诺 > 调用久远记忆 > 日常交流
    """
    # 延迟导入避免循环依赖（intent_replies 依赖 prompting.store）
    from app.services.chat.intent_replies import unified_intent_recognize, split_multi_intent

    try:
        labels = await unified_intent_recognize(message)
    except Exception:
        return IntentResult()

    if not labels or labels == ["日常交流"]:
        return IntentResult()

    # 多意图：调 split 以便下游可按 fragment 处理
    fragments: dict[str, str] | None = None
    if len(labels) > 1:
        try:
            fragments = await split_multi_intent(message, labels)
        except Exception:
            fragments = None

    primary = next((l for l in INTENT_PRIORITY if l in labels), labels[0])
    intent = LABEL_TO_INTENT.get(primary, IntentType.NONE)
    if intent == IntentType.NONE:
        return IntentResult()

    metadata: dict[str, Any] = {"llm_labels": labels}
    if fragments:
        metadata["fragments"] = fragments
    return IntentResult(intent=intent, confidence=0.75, metadata=metadata)
