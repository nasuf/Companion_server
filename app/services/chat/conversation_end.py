"""终结对话识别服务。

检测用户是否要结束对话，纯关键词匹配，无LLM调用。
PRD §3.2.3
"""

from __future__ import annotations

CONVERSATION_END_KEYWORDS = [
    "拜拜", "88", "886", "先忙了", "睡了", "睡觉了", "回头聊",
    "不说了", "下次聊", "告辞", "再见", "晚安", "拜", "byebye",
    "bye", "先走了", "下次再聊", "改天聊", "先这样", "我先走了",
    "不聊了", "去忙了",
]


def check_conversation_end(message: str) -> bool:
    """检测用户是否要结束对话。"""
    msg = message.strip().lower()
    return any(kw in msg for kw in CONVERSATION_END_KEYWORDS)
