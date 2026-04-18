"""Small-model memory pre-filter (spec §2.1.2 / §2.2.2).

A fast, cheap LLM call that binary-classifies a message as "记" (worth
remembering) or "不记" (skip). Runs BEFORE the expensive big-model
extraction step, saving cost on messages that are clearly not memorable.
"""

from __future__ import annotations

import logging

from app.services.llm.models import get_utility_model, invoke_json

logger = logging.getLogger(__name__)


async def should_memorize(message: str) -> bool:
    """Return True if the message is worth extracting memories from.

    Uses the smallest available model for speed. Expected latency: <500ms.
    """
    prompt = f"""判断这条消息是否包含值得记忆的信息。

消息: "{message}"

判断标准:
- "记": 包含个人身份、经历、偏好、情绪、观点、事件、计划等有信息量的内容
- "不记": 纯寒暄(你好/嗯/哈哈)、单纯提问(天气怎么样)、无信息量的回应

输出 JSON: {{"decision": "记"}} 或 {{"decision": "不记"}}
"""
    try:
        result = await invoke_json(get_utility_model(), prompt)
        decision = result.get("decision", "记") if isinstance(result, dict) else "记"
        return decision == "记"
    except Exception as e:
        logger.warning(f"Memory pre-filter LLM failed: {e}")
        # Fail open: if pre-filter fails, let the big model decide
        return True
