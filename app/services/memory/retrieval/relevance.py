"""Memory relevance classification and reranking.

Product spec §3.1：每条用户消息先判断与记忆的相关程度 (强/中/弱),
决定是否调取记忆以及调取多少。

强: 用户明确要求回忆, 或话题与记忆高度绑定 → 搜 L1+L2 前50, 考虑 L3
中: 话题与记忆有关联但不强制 → 搜 L1+L2 前50, 不触发 L3
弱: 与记忆完全无关 → 不调任何记忆

工程偏离 spec §3.1：
- 输入: 复用 intent.unified 的最近几轮对话上下文格式让 LLM 解指代
  ("颜色呢？" → 解出"颜色"指代上一轮的具体话题)
- Phase 2.4 输出: JSON {"level": "强|中|弱", "enhanced_query": "..."}
  - level 同 spec
  - enhanced_query 是 LLM 把省略指代还原后的可检索 query
    (e.g. "那他怎样了?" + 上下文有"妈妈" → enhanced_query="用户的妈妈现状")
  - hybrid retrieval 用 enhanced_query 做 embedding, 提升省略式追问召回率
  - 弱相关时 enhanced_query 可空 (反正不检索)
- 输出 schema 升级 + 兼容旧单字符响应 (LLM 偶尔忽略 schema)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.utils import render_prompt

logger = logging.getLogger(__name__)

RelevanceLevel = Literal["strong", "medium", "weak"]

_LEVEL_MAP: dict[str, RelevanceLevel] = {"强": "strong", "中": "medium", "弱": "weak"}


@dataclass
class RelevanceResult:
    """Phase 2.4: relevance 分类 + LLM 解指代后的增强 query.

    enhanced_query: 用户原话经 LLM 解省略指代后的完整可检索查询. 空串表示
    LLM 没解码出有用 query (无指代/弱相关), retrieval 应 fallback 到原 message.
    """
    level: RelevanceLevel
    enhanced_query: str = ""


def _parse_relevance_response(raw: str) -> RelevanceResult:
    """解析 LLM 输出. 优先 JSON, 失败 fallback 到旧"强/中/弱"单字符模式.

    LLM 可能的输出形态:
    1. 标准 JSON: {"level": "强", "enhanced_query": "..."}
    2. JSON 含前后缀文字: "好的, {...JSON...}"
    3. 旧格式单字符: "强"
    4. 多字符废话: "我觉得是中"
    """
    text = (raw or "").strip()
    if not text:
        return RelevanceResult(level="medium", enhanced_query="")

    # Try JSON first
    json_text = text
    # 容忍前后冗余文字, 提取 {...} 部分
    if "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_text = text[start:end]
    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            level_raw = str(data.get("level", "")).strip()
            level = _LEVEL_MAP.get(level_raw[:1], None) if level_raw else None
            enhanced = str(data.get("enhanced_query", "")).strip()
            if level:
                # cap enhanced_query 长度防 LLM 啰嗦 (>50 字反而稀释 embedding 信号)
                if len(enhanced) > 50:
                    enhanced = enhanced[:50]
                return RelevanceResult(level=level, enhanced_query=enhanced)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: 旧单字符模式
    for ch in text:
        if ch in _LEVEL_MAP:
            return RelevanceResult(level=_LEVEL_MAP[ch], enhanced_query="")

    # 完全解析失败 → medium 默认 (跟历史行为一致)
    return RelevanceResult(level="medium", enhanced_query="")


async def classify_memory_relevance(
    user_message: str,
    context: str = "",
) -> RelevanceResult:
    """Phase 2.4: 返回 RelevanceResult(level, enhanced_query).

    `context` 与 intent.unified 同格式: 最近几轮 "AI: ... / 用户: ..." 换行拼接,
    用于解析省略式追问. 空串时填 "(无)" — LLM 退化到仅看当前消息且 enhanced_query 空。

    返回 RelevanceResult 而非 RelevanceLevel (历史 ABI 改造) — 旧 caller 取 .level
    字段即可获得原 RelevanceLevel str.
    """
    try:
        raw = await render_prompt(
            "memory.relevance",
            {"message": user_message, "context": context or "(无)"},
            lambda p: invoke_text(get_utility_model(), p),
        )
        return _parse_relevance_response(raw or "")
    except Exception as e:
        logger.warning(f"Memory relevance classification failed: {e}; defaulting to 'medium'")
        return RelevanceResult(level="medium", enhanced_query="")


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

    Phase 2.1: L1 (importance ≥ 0.85) 不衰减 — 身份记忆 (姓名/年龄/家人 等核心
    事实) 即便很久没访问也是事实, time_freshness 衰减没意义. 历史 bug: 用户半年
    没说自己名字 → freshness=0.4 → display_score 被新琐事压低 → AI 答非所问.
    L1 freshness floor 取 1.0, 让 importance × similarity 主导排序.
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

    # Phase 2.1: L1 不衰减 (核心身份事实 永恒)
    if importance >= 0.85:
        freshness = max(freshness, 1.0)

    return importance * freshness * similarity
