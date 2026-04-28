"""Hybrid retrieval orchestrator.

Combines vector search + graph queries for comprehensive memory retrieval.

Pipeline (no LLM calls — pure data operations):

  parallel(vector_search + graph_context) -> fusion -> ranker -> context_selector

Includes Redis caching for retrieval results and graph context.
"""

import asyncio
import logging
import re
from datetime import datetime

from app.db import db
from app.services.memory.retrieval.vector_search import search_similar, search_by_time_range
from app.services.memory.retrieval.context_selector import select_context
from app.services.memory.retrieval.relevance import compute_display_score
from app.services.memory.storage.entity_repo import get_relationship_context
from app.services.runtime.cache import (
    cache_retrieval,
    cache_set_retrieval,
    cache_graph_context,
    cache_set_graph_context,
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


async def _keyword_fallback_search(
    message: str,
    user_id: str,
    workspace_id: str | None,
    already_seen: set[str],
    levels: list[int],
    recent_context: str = "",
) -> list[dict]:
    """Keyword-based fallback: direct ILIKE on content/summary.

    Safety net for cases where vector embedding can't bridge semantic gap
    (代词/省略 — "它叫什么" 跟 "饲养一只名为'拿铁'..." 之间). 提取关键词的
    来源不仅是 user message 自身, 还包括 recent_context 里近 1-2 轮对话的
    名词 (如最近 AI 消息提到 "猫"), 这样 "它叫什么" 也能命中含 "猫" 的记忆.
    """
    # 关键词抽取范围: user message 优先, recent_context 提供 entity 锚点
    _STOP = {"什么", "怎么", "哪里", "哪个", "为什么", "怎样", "如何", "可以", "是不是", "能不能"}
    msg_words = [w for w in re.findall(r'[\u4e00-\u9fff]{2,}', message) if w not in _STOP]
    ctx_words: list[str] = []
    if recent_context:
        ctx_words = [w for w in re.findall(r'[\u4e00-\u9fff]{2,}', recent_context) if w not in _STOP]
    # 用 dict 去重保序: message 词在前 (相关性更高), context 词补充
    seen: set[str] = set()
    words: list[str] = []
    for w in msg_words + ctx_words:
        if w not in seen:
            seen.add(w)
            words.append(w)
    words = words[:8]  # message 5 + context 3 上限, 防 SQL OR 太长
    if not words:
        return []

    # Build OR conditions
    conditions: list[str] = []
    params: list = [user_id]
    idx = 2
    for w in words:
        conditions.append(f'("content" ILIKE ${idx} OR "summary" ILIKE ${idx})')
        params.append(f"%{w}%")
        idx += 1

    level_list = ",".join(str(l) for l in levels)
    where_clause = " OR ".join(conditions)

    results: list[dict] = []
    # source 标签必须随 row 一起返回, 否则下游 prompt_builder 会把
    # memories_ai 误标为"用户告诉过你的事" → 人设串戏. 见 ClassifiedMemory.source.
    for table, source_label in (("memories_ai", "ai"), ("memories_user", "user")):
        ws_filter = f'AND "workspace_id" = ${idx}' if workspace_id else ""
        query_params = params + ([workspace_id] if workspace_id else [])

        try:
            rows = await db.query_raw(
                f"""
                SELECT id, content, summary, level, importance, type,
                       main_category, sub_category, created_at,
                       0.75::float AS similarity,
                       '{source_label}' AS source
                FROM "{table}"
                WHERE "user_id" = $1
                  AND "level" IN ({level_list})
                  AND "is_archived" = false
                  AND ({where_clause})
                  {ws_filter}
                ORDER BY importance DESC
                LIMIT 10
                """,
                *query_params,
            )
            for r in rows:
                mid = r.get("id", "")
                if mid and mid not in already_seen:
                    already_seen.add(mid)
                    results.append(r)
        except Exception as e:
            logger.debug(f"Keyword fallback on {table} failed: {e}")

    if results:
        logger.info(f"Keyword fallback found {len(results)} extra memories for '{message[:30]}'")
    return results


async def hybrid_retrieve(
    message: str,
    user_id: str,
    workspace_id: str | None = None,
    token_budget: int = 800,
    recent_context: str = "",
) -> dict:
    """Perform hybrid retrieval and return context for prompt.

    No LLM calls — only vector search + graph queries + ranking.

    `recent_context`: 最近 N 轮对话, 仅供 keyword fallback 抽取实体名词作为
    兜底关键词 (e.g. AI 上句提到 "猫" → 当前 "它叫什么" 兜底 ILIKE %猫%).
    **不**拼到 vector embedding 里 — 评估显示 context 会把"主题切换" query 的
    expected L1 稀释 (e.g. 风/猫上下文 + "你身高多少" 会让身高 L1 沉到阈值下).
    """
    # 快速跳过无意义短消息（避免向量搜索的开销）
    if _is_trivial_message(message):
        logger.debug("Skipping retrieval for trivial message: %s", message[:20])
        return _EMPTY_RESULT

    cached = await cache_retrieval(message, user_id, workspace_id=workspace_id)
    if cached:
        logger.debug("Hybrid retrieval cache hit")
        return cached

    # Spec §3.2 step 1: 向量搜索 L1+L2 + 时间搜索（若有显式时间）
    # 时间范围由时间系统（纯规则）解析，无 LLM 调用。
    time_range: tuple[datetime, datetime] | None = None
    if has_explicit_time(message):
        parsed = parse_time_expressions(message)
        if parsed:
            best = max(parsed, key=lambda p: p.confidence)
            if not best.is_future:
                time_range = (best.start, best.end)

    levels = [1, 2]

    vector_task = search_similar(
        message, user_id, top_k=50, workspace_id=workspace_id, levels=levels,
    )
    graph_task = get_relationship_context(
        user_id=user_id, workspace_id=workspace_id,
    )
    time_task = (
        search_by_time_range(
            user_id, time_range[0], time_range[1],
            limit=20, workspace_id=workspace_id,
        )
        if time_range else asyncio.sleep(0, result=[])
    )

    vector_results, graph_result, time_results = await asyncio.gather(
        vector_task, graph_task, time_task, return_exceptions=True
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
            sim = float(mem.get("similarity", 0))
            if mid and mid not in seen_ids and sim >= _SIMILARITY_THRESHOLD:
                seen_ids.add(mid)
                all_candidates.append(mem)

    logger.info(f"[DEBUG-VEC] after threshold={_SIMILARITY_THRESHOLD}: {len(all_candidates)} candidates")

    # Keyword fallback 始终跑 (跟 vector 互补): SQL ILIKE 查询 ~10ms, 防御性兜底.
    # 评估显示对召回率无害无增益, 但保留作为字面命中保险 (vector 跨语义弱时的备份).
    keyword_results = await _keyword_fallback_search(
        message, user_id, workspace_id, seen_ids, levels,
        recent_context=recent_context,
    )
    all_candidates.extend(keyword_results)

    # Spec §3.2 step 4: rerank by display_score = importance × time_freshness × similarity.
    # 我们没有 last_accessed_at 列, 用 created_at 作为 freshness 代理
    # (spec 意图是 "越久没被触达的记忆越靠后", created_at 与此大致一致)。
    # 只写 rank_score — ClassifiedMemory.display_score 由下游 data_fetch_phase
    # 统一赋值 + 截断到 10 条。
    for m in all_candidates:
        m["rank_score"] = compute_display_score(
            importance=float(m.get("importance", 0)),
            last_accessed_at=m.get("created_at"),
            similarity=float(m.get("similarity", 1.0)),
        )
    all_candidates.sort(key=lambda m: float(m.get("rank_score", 0)), reverse=True)

    # Select within token budget (returns ClassifiedMemory list)
    classified_memories = select_context(all_candidates, token_budget)

    # Plain text list for consumers that don't need ClassifiedMemory metadata
    memory_strings = [m.text for m in classified_memories] if classified_memories else None

    # Graph context (with caching)
    graph_context = None
    if isinstance(graph_result, Exception):
        logger.warning(f"Graph context failed: {graph_result}")
        graph_context = await cache_graph_context(user_id, workspace_id=workspace_id)
    else:
        graph_context = graph_result
        if graph_context:
            try:
                await cache_set_graph_context(user_id, graph_context, workspace_id=workspace_id)
            except Exception:
                pass

    result = {
        "memories": classified_memories if classified_memories else None,
        "memory_strings": memory_strings,
        "graph_context": graph_context,
    }

    # Cache the result
    try:
        await cache_set_retrieval(message, user_id, result, workspace_id=workspace_id)
    except Exception:
        pass

    return result
