"""Memory relevance classification and reranking.

Product spec §3: 每条用户消息先判断与记忆的相关程度 (强/中/弱),
决定是否调取记忆以及调取多少。

强: 用户明确要求回忆, 或话题与记忆高度绑定 → 搜 L1+L2 前50, 考虑 L3
中: 话题与记忆有关联但不强制 → 搜 L1+L2 前50, 不触发 L3
弱: 与记忆完全无关 → 不调任何记忆
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from app.services.llm.models import get_utility_model, invoke_json

logger = logging.getLogger(__name__)

RelevanceLevel = Literal["strong", "medium", "weak"]


async def classify_memory_relevance(user_message: str) -> RelevanceLevel:
    """Use a small/fast LLM call to classify how relevant the user's
    message is to stored memory.

    Returns "strong", "medium", or "weak".
    """
    prompt = f"""你是一个记忆相关度分析器。判断用户消息与AI助手自身记忆的相关程度。
注意：这里的"记忆"包括AI自身的身份信息（名字、职业、住所、年龄、血型、爱好等）以及与用户的共同经历。

定义（严格遵循）:
「强」: 询问AI个人信息（身份/职业/住所/年龄/喜好/经历等），或明确要求回忆。
  例: 你是做什么工作的 / 你住在哪 / 你多大了 / 你叫什么 / 你喜欢什么 / 你还记得吗 / 上次我说的 / 你的血型
「中」: 话题可能与记忆有关联，聊天涉及生活/情感/经历。
  例: 今天心情不好 / 最近工作好累 / 周末去了哪里 / 我想你了
「弱」: 与记忆完全无关，纯寒暄或通用知识问答。
  例: 你好 / 哈哈 / 今天天气怎么样 / 帮我写一首诗 / 1+1等于几

用户消息: {user_message}

输出严格 JSON: {{"level": "强"|"中"|"弱"}}"""
    try:
        result = await invoke_json(get_utility_model(), prompt)
        level = result.get("level", "中") if isinstance(result, dict) else "中"
        mapping: dict[str, RelevanceLevel] = {"强": "strong", "中": "medium", "弱": "weak"}
        return mapping.get(level, "medium")
    except Exception as e:
        logger.warning(f"Memory relevance classification failed: {e}; defaulting to 'medium'")
        return "medium"


def compute_display_score(
    importance: float,
    last_accessed_at: datetime | str | None,
    similarity: float = 1.0,
) -> float:
    """Product spec §3.2 reranking formula:
    display_score = current_score × time_freshness × topic_match

    - current_score: importance (0-1)
    - time_freshness: based on how recently the memory was accessed/created
    - topic_match: vector similarity (0-1)
    """
    # Time freshness factor (spec §3.2):
    # <1 month: 1.2  |  1-3 months: 1.0  |  3-6 months: 0.8
    # 6-12 months: 0.6  |  >12 months: 0.4
    now = datetime.now(timezone.utc)
    if isinstance(last_accessed_at, str):
        try:
            last_accessed_at = datetime.fromisoformat(last_accessed_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            last_accessed_at = None

    if last_accessed_at and last_accessed_at.tzinfo:
        days = (now - last_accessed_at).days
    else:
        days = 30  # Default: 1 month freshness

    if days < 30:
        freshness = 1.2
    elif days < 90:
        freshness = 1.0
    elif days < 180:
        freshness = 0.8
    elif days < 365:
        freshness = 0.6
    else:
        freshness = 0.4

    return importance * freshness * similarity
