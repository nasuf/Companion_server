"""Hybrid retrieval orchestrator.

Combines vector search + explicit time search for comprehensive memory retrieval.

Pipeline (no LLM calls — pure data operations):

  parallel(vector_search + explicit time search) -> fusion -> ranker -> context_selector

Includes Redis caching for retrieval results.
"""

import asyncio
import logging
import re
from datetime import datetime

from app.services.memory.retrieval.vector_search import search_similar, search_by_time_range
from app.services.memory.retrieval.context_selector import select_context
from app.services.memory.retrieval.relevance import compute_display_score
from app.services.runtime.cache import (
    cache_retrieval,
    cache_set_retrieval,
)
from app.services.schedule_domain.time_parser import has_explicit_time, parse_time_expressions

logger = logging.getLogger(__name__)

# 无需检索的短消息/语气词/纯问候（跳过向量搜索）
_TRIVIAL_WORDS = {
    "嗯", "嗯嗯", "哦", "哦哦", "好", "好的", "行", "行吧", "ok", "OK",
    "哈哈", "哈哈哈", "呵呵", "嘻嘻", "嘿嘿", "哇", "额", "唔",
    "是", "是的", "对", "对对", "没", "没有", "不是", "不会",
    "谢谢", "感谢", "好吧", "可以", "当然", "知道了", "收到",
    "早", "早上好", "晚安", "你好", "hello", "hi", "嗨",
    "啊", "啊啊", "了", "吧", "呢", "吗", "呀", "喔", "噢",
    "666", "hh", "hhh", "哭", "累",
}

_EMPTY_RESULT = {
    "memories": None,
    "memory_strings": None,
    "graph_context": None,
}

# Spec §3.2 前级过滤相似度阈值。Spec 原值 0.7; bge-m3 对中文短文本召回
# 能力不足, 降到 0.5 以保证召回率 (见 docs/spec-audit-2026-04-23.md)。
_SIMILARITY_THRESHOLD = 0.50


def _is_trivial_message(message: str) -> bool:
    """快速判断消息是否为不需要记忆检索的无意义短消息。"""
    text = message.strip()
    if not text:
        return True
    # 纯 emoji / 纯标点
    cleaned = re.sub(r'[\s\U00010000-\U0010ffff.,!?。，！？…~～、]+', '', text)
    if not cleaned:
        return True
    # 精确匹配语气词表
    if text.lower() in _TRIVIAL_WORDS:
        return True
    # 极短纯重复字符 (如 "嗯嗯嗯嗯")
    if len(text) <= 6 and len(set(text)) <= 2:
        return True
    return False


async def hybrid_retrieve(
    message: str,
    user_id: str,
    workspace_id: str | None = None,
    token_budget: int = 800,
    enhanced_query: str | None = None,
) -> dict:
    """Perform hybrid retrieval and return context for prompt.

    No LLM calls — only vector search + graph queries + ranking.

    Phase 2.4: enhanced_query 是 LLM 解省略指代后的完整短语 (e.g. "妈妈病情"
    替代原"那他怎样了"). 优先用 enhanced_query 做 vector embedding, fallback
    到 message. 时间解析仍用原 message (时间词通常在原话, e.g. "上周那个事").
    """
    # 快速跳过无意义短消息（避免向量搜索的开销）
    if _is_trivial_message(message):
        logger.debug("Skipping retrieval for trivial message: %s", message[:20])
        return _EMPTY_RESULT

    # Phase 2.4: cache key 用 effective_query (含 enhanced) 避免不同指代复用同 cache
    effective_query = enhanced_query or message
    cache_key = effective_query if enhanced_query else message
    cached = await cache_retrieval(cache_key, user_id, workspace_id=workspace_id)
    if cached:
        logger.debug("Hybrid retrieval cache hit (key=%s)", cache_key[:30])
        return cached

    # Spec §3.2 step 1: 向量搜索 L1+L2 + 时间搜索（若有显式时间）
    # 时间范围由时间系统（纯规则）解析，无 LLM 调用. 时间词通常在原话, 用 message.
    time_range: tuple[datetime, datetime] | None = None
    if has_explicit_time(message):
        parsed = parse_time_expressions(message)
        if parsed:
            best = max(parsed, key=lambda p: p.confidence)
            if not best.is_future:
                time_range = (best.start, best.end)

    levels = [1, 2]

    # Phase 2.4: vector embedding 用 effective_query (enhanced 优先)
    if enhanced_query:
        logger.info(
            f"[DEBUG-VEC] using enhanced_query='{enhanced_query[:40]}' "
            f"(original message='{message[:40]}')"
        )
    vector_task = search_similar(
        effective_query, user_id, top_k=50, workspace_id=workspace_id, levels=levels,
    )
    time_task = (
        search_by_time_range(
            user_id, time_range[0], time_range[1],
            limit=20, workspace_id=workspace_id,
        )
        if time_range else asyncio.sleep(0, result=[])
    )

    vector_results, time_results = await asyncio.gather(
        vector_task, time_task, return_exceptions=True
    )

    # Log raw vector search results for debugging
    if isinstance(vector_results, Exception):
        logger.info(f"[DEBUG-VEC] vector search EXCEPTION: {vector_results}")
    else:
        total = len(vector_results) if vector_results else 0
        logger.info(f"[DEBUG-VEC] vector search returned {total} raw results for '{message[:50]}'")
        if vector_results:
            for r in sorted(vector_results, key=lambda x: float(x.get("similarity", 0)), reverse=True)[:5]:
                logger.info(f"[DEBUG-VEC]   sim={float(r.get('similarity',0)):.3f} '{(r.get('summary') or r.get('content',''))[:60]}'")

    # Merge vector + time results (union by id), apply similarity threshold.
    all_candidates: list[dict] = []
    seen_ids: set[str] = set()
    for source_results, label in [(vector_results, "vector"), (time_results, "time")]:
        if isinstance(source_results, Exception):
            logger.warning(f"{label} search failed: {source_results}")
            continue
        for mem in (source_results or []):
            mid = mem.get("id", "")
            if not mid or mid in seen_ids:
                continue
            # Time-range matches are explicit user intent ("去年生日那天",
            # "上周那件事") and rows from search_by_time_range do not carry a
            # vector similarity. Do not run them through the semantic threshold.
            if label == "time":
                mem.setdefault("similarity", 1.0)
                mem["_retrieval_source"] = "time"
                seen_ids.add(mid)
                all_candidates.append(mem)
                continue

            sim = float(mem.get("similarity", 0))
            if sim >= _SIMILARITY_THRESHOLD:
                seen_ids.add(mid)
                all_candidates.append(mem)

    logger.info(f"[DEBUG-VEC] after threshold={_SIMILARITY_THRESHOLD}: {len(all_candidates)} candidates")

    # Spec §3.2 step 4: rerank by display_score = importance × time_freshness × similarity.
    # last_accessed_at comes from updated_at (touched by access_log) with created_at fallback,
    # so prompt-injected memories become fresh in the next retrieval pass.
    # 只写 rank_score — ClassifiedMemory.display_score 由下游 data_fetch_phase
    # 统一赋值 + 截断到 10 条。
    #
    # Phase 3.2: polarity 降权 — bge-m3 反义对 cosine 0.84+, 跟同义难分.
    # 用户 query 有显式否定 + candidate 没有 → 极性 mismatch → 降权 0.3
    # (不删, 极端情况 LLM 仍可见). 仅在 user_query 有否定时触发, 防 positive
    # query 误过滤 negative candidate (用户问"我喜欢什么", 应该看到所有偏好,
    # 包括"不喜欢" 类记忆).
    from app.services.memory.polarity import has_negation
    user_has_neg = has_negation(effective_query)

    for m in all_candidates:
        score = compute_display_score(
            importance=float(m.get("importance", 0)),
            last_accessed_at=(
                m.get("last_accessed_at")
                or m.get("updated_at")
                or m.get("created_at")
            ),
            similarity=float(m.get("similarity", 1.0)),
        )
        # Phase 3.2: 用户显式否定 query → candidate 无否定 → 极性 mismatch 降权
        if user_has_neg:
            cand_text = m.get("summary") or m.get("content", "")
            if not has_negation(cand_text):
                score *= 0.3
                logger.debug(
                    f"[POLARITY] downweight pos candidate "
                    f"(user_query has negation): '{cand_text[:30]}'"
                )
        m["rank_score"] = score
    all_candidates.sort(key=lambda m: float(m.get("rank_score", 0)), reverse=True)

    # Select within token budget (returns ClassifiedMemory list)
    classified_memories = select_context(all_candidates, token_budget)

    # Plain text list for consumers that don't need ClassifiedMemory metadata
    memory_strings = [m.text for m in classified_memories] if classified_memories else None

    result = {
        "memories": classified_memories if classified_memories else None,
        "memory_strings": memory_strings,
        "graph_context": None,
    }

    # Cache the result. Phase 2.4: cache write key 必须跟 GET 用同一个 cache_key
    # (effective_query), 否则 enhanced_query 路径 GET 永远 miss → caching 失效.
    try:
        await cache_set_retrieval(cache_key, user_id, result, workspace_id=workspace_id)
    except Exception:
        pass

    return result
