"""Hybrid retrieval orchestrator.

Combines vector search + entity recall + explicit time search for comprehensive memory retrieval.

Pipeline (no LLM calls — pure data operations):

  parallel(vector_search + entity recall + explicit time search) -> fusion -> ranker -> context_selector

Includes Redis caching for retrieval results.
"""

import asyncio
import logging
import re
from datetime import datetime

from app.services.memory.retrieval.vector_search import search_similar, search_by_time_range
from app.services.memory.retrieval.context_selector import ClassifiedMemory, select_context
from app.services.memory.retrieval.ranking import rank_memory_candidate
from app.services.memory.retrieval.trace import record_retrieval_session
from app.services.memory.storage.entity_repo import search_related_memories_for_query
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
_TRIVIAL_REPEAT_CHARS = set("嗯哦喔噢啊哈呵嘻嘿呃额唔哇吧呀呢吗啦了")

_EMPTY_RESULT = {
    "memories": None,
    "memory_strings": None,
    "graph_context": None,
}

# Spec §3.2 前级过滤相似度阈值。Spec 原值 0.7; bge-m3 对中文短文本召回
# 能力不足, 降到 0.5 以保证召回率 (见 docs/spec-audit-2026-04-23.md)。
_SIMILARITY_THRESHOLD = 0.50
_ENTITY_RECALL_SIMILARITY = 0.78


def _memory_to_cache_dict(memory: ClassifiedMemory) -> dict:
    return {
        "text": memory.text,
        "relevance": memory.relevance,
        "score": memory.score,
        "id": memory.id,
        "importance": memory.importance,
        "similarity": memory.similarity,
        "mention_count": memory.mention_count,
        "main_category": memory.main_category,
        "sub_category": memory.sub_category,
        "created_at": memory.created_at,
        "last_accessed_at": memory.last_accessed_at,
        "display_score": memory.display_score,
        "rank_reasons": list(memory.rank_reasons or []),
        "source": memory.source,
    }


def _memory_from_cache_dict(item: dict) -> ClassifiedMemory | None:
    text = str(item.get("text") or item.get("summary") or item.get("content") or "").strip()
    if not text:
        return None
    try:
        score = float(item.get("score", item.get("display_score", 0.5)) or 0.5)
        importance = float(item.get("importance", 0.5) or 0.5)
        similarity = float(item.get("similarity", 0.8) or 0.8)
        display_score = float(item.get("display_score", score) or score)
        mention_count = int(item.get("mention_count") or 0)
    except (TypeError, ValueError):
        return None
    source = "ai" if item.get("source") == "ai" else "user"
    relevance = str(item.get("relevance") or "")
    if relevance not in {"strong", "medium"}:
        relevance = "strong" if score >= 0.7 else "medium"
    return ClassifiedMemory(
        text=text,
        relevance=relevance,
        score=score,
        id=str(item.get("id") or ""),
        importance=importance,
        similarity=similarity,
        mention_count=mention_count,
        main_category=item.get("main_category"),
        sub_category=item.get("sub_category"),
        created_at=item.get("created_at"),
        last_accessed_at=item.get("last_accessed_at"),
        display_score=display_score,
        rank_reasons=list(item.get("rank_reasons") or []),
        source=source,
    )


def _rehydrate_cached_memories(cached: dict) -> dict:
    memories = cached.get("memories")
    if not isinstance(memories, list):
        return cached

    hydrated: list[ClassifiedMemory] = []
    for item in memories:
        if isinstance(item, ClassifiedMemory):
            hydrated.append(item)
        elif isinstance(item, dict):
            memory = _memory_from_cache_dict(item)
            if memory:
                hydrated.append(memory)

    # Backward compatibility for short-lived old cache entries written before
    # memories became structured. They lost ids/scores, but this avoids a crash
    # and lets Redis TTL naturally age them out.
    if not hydrated:
        strings = cached.get("memory_strings")
        if isinstance(strings, list):
            hydrated = [
                ClassifiedMemory(text=str(text), relevance="medium", score=0.5)
                for text in strings
                if str(text).strip()
            ]

    cached["memories"] = hydrated or None
    if hydrated:
        cached["memory_strings"] = [m.text for m in hydrated]
    return cached


def _cacheable_retrieval_result(result: dict) -> dict:
    cached = dict(result)
    memories = result.get("memories")
    if isinstance(memories, list):
        cached["memories"] = [
            _memory_to_cache_dict(memory)
            for memory in memories
            if isinstance(memory, ClassifiedMemory)
        ] or None
    return cached


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
    # 极短语气词重复 (如 "嗯嗯嗯嗯"). 不要误伤 "妈妈呢" 这类实体追问。
    if (
        len(text) <= 6
        and len(set(text)) <= 2
        and all(ch in _TRIVIAL_REPEAT_CHARS for ch in text)
    ):
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

    No LLM calls — only vector search + explicit time search + ranking.

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
        cached = _rehydrate_cached_memories(cached)
        cached_memories = cached.get("memories") if isinstance(cached, dict) else None
        record_retrieval_session(
            strategy="hybrid_l1_l2",
            query=message,
            enhanced_query=enhanced_query,
            workspace_id=workspace_id,
            cache_hit=True,
            selected=cached_memories if isinstance(cached_memories, list) else [],
            selected_count=len(cached_memories) if isinstance(cached_memories, list) else 0,
            notes={"cache_key": cache_key[:80]},
        )
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
    entity_task = search_related_memories_for_query(
        user_id=user_id,
        workspace_id=workspace_id,
        query=effective_query,
        entity_limit=5,
        memory_limit=20,
        levels=levels,
    )

    vector_results, time_results, entity_results = await asyncio.gather(
        vector_task, time_task, entity_task, return_exceptions=True
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

    # Merge vector + entity + time results (union by id), applying semantic
    # threshold only to vector results. Entity/time matches are explicit user
    # anchors and should reach the reranker even if the embedding model misses.
    all_candidates: list[dict] = []
    candidate_by_id: dict[str, dict] = {}

    def _merge_candidate(mem: dict, label: str) -> None:
        mid = mem.get("id", "")
        if not mid:
            return
        existing = candidate_by_id.get(mid)
        if existing is not None:
            sources = set(str(existing.get("_retrieval_source") or "vector").split("+"))
            sources.add(label)
            existing["_retrieval_source"] = "+".join(sorted(sources))
            if label == "entity":
                existing["_entity_match"] = True
                if mem.get("matched_entity"):
                    existing["matched_entity"] = mem.get("matched_entity")
                existing["similarity"] = max(
                    float(existing.get("similarity", 0) or 0),
                    _ENTITY_RECALL_SIMILARITY,
                )
            if label == "time":
                existing["similarity"] = 1.0
            return
        candidate_by_id[mid] = mem
        all_candidates.append(mem)

    for source_results, label in [
        (vector_results, "vector"),
        (entity_results, "entity"),
        (time_results, "time"),
    ]:
        if isinstance(source_results, Exception):
            logger.warning(f"{label} search failed: {source_results}")
            continue
        for mem in (source_results or []):
            # Time-range matches are explicit user intent ("去年生日那天",
            # "上周那件事") and rows from search_by_time_range do not carry a
            # vector similarity. Do not run them through the semantic threshold.
            if label == "time":
                mem.setdefault("similarity", 1.0)
                mem["_retrieval_source"] = "time"
                _merge_candidate(mem, label)
                continue
            if label == "entity":
                mem.setdefault("similarity", _ENTITY_RECALL_SIMILARITY)
                mem["_retrieval_source"] = "entity"
                mem["_entity_match"] = True
                _merge_candidate(mem, label)
                continue

            sim = float(mem.get("similarity", 0))
            if sim >= _SIMILARITY_THRESHOLD:
                mem["_retrieval_source"] = "vector"
                _merge_candidate(mem, label)

    logger.info(f"[DEBUG-VEC] after threshold={_SIMILARITY_THRESHOLD}: {len(all_candidates)} candidates")

    # Spec §3.2 step 4 + retrieval v2: rerank by display_score plus lightweight
    # keyword/category/safety boosts. The vector model recalls broadly; these
    # deterministic signals stop critical emotional or literal-topic memories
    # from being buried by generic high-importance facts.
    for m in all_candidates:
        score, reasons = rank_memory_candidate(m, effective_query)
        m["rank_score"] = score
        m["rank_reasons"] = reasons
    all_candidates.sort(key=lambda m: float(m.get("rank_score", 0)), reverse=True)

    # Select within token budget (returns ClassifiedMemory list)
    classified_memories = select_context(
        all_candidates,
        token_budget,
        query=effective_query,
    )
    record_retrieval_session(
        strategy="hybrid_l1_l2",
        query=message,
        enhanced_query=enhanced_query,
        workspace_id=workspace_id,
        cache_hit=False,
        raw_count=(
            (len(vector_results) if isinstance(vector_results, list) else 0)
            + (len(time_results) if isinstance(time_results, list) else 0)
            + (len(entity_results) if isinstance(entity_results, list) else 0)
        ),
        candidate_count=len(all_candidates),
        selected_count=len(classified_memories),
        candidates=all_candidates,
        selected=classified_memories,
        notes={
            "similarity_threshold": _SIMILARITY_THRESHOLD,
            "has_explicit_time": bool(time_range),
            "effective_query": effective_query[:80],
            "entity_result_count": (
                len(entity_results) if isinstance(entity_results, list) else 0
            ),
        },
    )

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
        await cache_set_retrieval(
            cache_key,
            user_id,
            _cacheable_retrieval_result(result),
            workspace_id=workspace_id,
        )
    except Exception:
        pass

    return result
