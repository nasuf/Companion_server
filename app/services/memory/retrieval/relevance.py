"""Memory relevance classification and reranking.

Product spec §3.1：每条用户消息先判断与记忆的相关程度 (强/中/弱),
决定是否调取记忆以及调取多少。

强: 用户明确要求回忆, 或话题与记忆高度绑定 → 搜 L1+L2 前50, 考虑 L3
中: 话题与记忆有关联但不强制 → 搜 L1+L2 前50, 不触发 L3
弱: 与记忆完全无关 → 不调任何记忆

工程偏离 spec §3.1：spec 输入只列了"用户消息"单条文本, 实测对省略式追问
("颜色呢？") 误判为弱 → 漏召回. 这里复用 intent.unified 已用的最近几轮对话
上下文格式让 LLM 自己解指代——prompt 走 registry (memory.relevance), 仅在
模板基础上新增 {context} 占位符, 其他 spec 措辞 / 输出格式保持不变。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.utils import render_prompt

logger = logging.getLogger(__name__)

RelevanceLevel = Literal["strong", "medium", "weak"]

_LEVEL_MAP: dict[str, RelevanceLevel] = {"强": "strong", "中": "medium", "弱": "weak"}


async def classify_memory_relevance(
    user_message: str,
    context: str = "",
) -> RelevanceLevel:
    """spec §3.1 小模型「判断回忆相关」, 输出 strong/medium/weak.

    `context` 与 intent.unified 同格式: 最近几轮 "AI: ... / 用户: ..." 换行拼接,
    用于解析省略式追问. 空串时填 "(无)" — LLM 退化到仅看当前消息。
    """
    try:
        raw = await render_prompt(
            "memory.relevance",
            {"message": user_message, "context": context or "(无)"},
            lambda p: invoke_text(get_utility_model(), p),
        )
        text = (raw or "").strip()
        # spec 输出是单字 "强"/"中"/"弱", 取首个匹配字符兜底 LLM 多嘴
        for ch in text:
            if ch in _LEVEL_MAP:
                return _LEVEL_MAP[ch]
        return "medium"
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
