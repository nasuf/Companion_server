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
from app.services.relationship.boundary import APOLOGY_KEYWORDS
from app.services.schedule_domain.schedule import _SCHEDULE_QUERY_KEYWORDS


class IntentType(Enum):
    APOLOGY = "apology"
    PROMISE = "promise"
    CONVERSATION_END = "conversation_end"
    DELETION = "deletion"
    SCHEDULE_ADJUST = "schedule_adjust"
    SCHEDULE_QUERY = "schedule_query"
    NONE = "none"


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

# 优先级顺序：前面的优先匹配
_CHECKS: list[tuple[IntentType, list[str]]] = [
    (IntentType.APOLOGY, APOLOGY_KEYWORDS),
    (IntentType.PROMISE, PROMISE_KEYWORDS),
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
