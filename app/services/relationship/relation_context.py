"""关系情绪线索检测（orchestrator 拆分 R4）。

"降低模板化回复机器人感"的工程经验层：用户在表达被忽略感/低落情绪时，
给主回复 prompt 注入"先接住情绪再解释"的指引（relational_context）。
归属 relationship/ — 它读的是关系与情绪信号，不是聊天编排逻辑。
"""

from __future__ import annotations

from app.services.relationship.emotion import is_negative_emotion
from app.services.rules.chat_keywords import (
    DISTRESS_KEYWORDS,
    RELATIONAL_COMPLAINT_KEYWORDS,
)


def detect_relational_context(message: str, user_emotion: dict | None) -> str | None:
    """Detect relationship repair / distress cues that need more human handling."""
    text = message.strip()
    if any(keyword in text for keyword in RELATIONAL_COMPLAINT_KEYWORDS):
        return (
            "用户这句更像是在确认你有没有在意Ta，或者在表达被忽略感。"
            "先短促地接住关系情绪，比如安抚、解释半句、表明你不是故意的；"
            "不要一上来就长解释，也不要立刻抛万能反问。"
        )

    negative_emotion = is_negative_emotion(user_emotion)
    if any(keyword in text for keyword in DISTRESS_KEYWORDS) or negative_emotion:
        return (
            "用户这句带明显低落或烦闷情绪。"
            "先回应当下感受，语气真一点、短一点；"
            "不要套模板安慰，不要一下子给很多建议，追问也只问最贴当前情绪的一句。"
        )
    return None
