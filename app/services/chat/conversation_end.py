"""终结对话识别服务。

检测用户是否要结束对话，纯关键词匹配，无LLM调用。
PRD §3.2.3
"""

from __future__ import annotations

from app.services.rules.chat_keywords import CONVERSATION_END_KEYWORDS


def check_conversation_end(message: str) -> bool:
    """检测用户是否要结束对话。"""
    msg = message.strip().lower()
    return any(kw in msg for kw in CONVERSATION_END_KEYWORDS)
