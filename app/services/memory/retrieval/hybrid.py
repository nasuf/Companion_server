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

from app.services.memory.retrieval.timeline import (
    build_timeline,
    format_timeline,
    is_aggregate_time_question,
)
from app.services.memory.retrieval.vector_search import (
    search_by_time_range,
    search_similar_tiers,
)
from app.services.memory.retrieval.context_selector import ClassifiedMemory, select_context
from app.services.memory.retrieval.legacy import _L3_SIMILARITY_FLOOR
from app.services.memory.retrieval.query_patterns import asks_shared_history
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
    "timeline": None,
}

# Spec §3.2 前级过滤相似度阈值。Spec 原值 0.7; 中文短文本上过严会漏召
# (见 docs/spec-audit-2026-04-23.md)。
#
# 2026-07 换 embedding (bge-m3 → qwen3-embedding:0.6b) 时重标: 新模型动态范围
# 更宽, 不相关文本对被压得更低 (query↔memory 均值 0.434 → 0.332), 照搬 0.50
# 会砍掉大部分召回。取 0.35 是为了**保持召回不变**而不是顺手收紧 —— 换模型这
# 一步只改模型, 操作点留到之后单独测。依据是 423 条人工判定过的真实候选:
#
#     bge-m3  @0.50   有用记忆保住 100%   每条消息注入 9.4 条
#     qwen3   @0.35   有用记忆保住  97%   每条消息注入 8.8 条   ← 取这个
#     qwen3   @0.40   有用记忆保住  85%
#     qwen3   @0.45   有用记忆保住  69%
#
# 按分布配对映射算出来是 0.46, 但那保持的是"随机文本对的放行比例", 不是"有用
# 记忆的留存率" —— 在这个位置上后者才是要守的东西。
_SIMILARITY_THRESHOLD = 0.35

# L3 有界采样的两个旋钮 (Phase 2)。κ=0 完全退回"L3 不可检索"的旧行为, 是这一步的
# 回滚开关。
#
# κ=3: 注入上限本就是 user/ai 各 10 条, 给冷层 3 个候选名额意味着它们最多占掉不到
# 六分之一, 且还要在 rank 里跟热候选竞争才进得去 —— 上界可控。
#
# 门槛直接取 L3 唤醒路径校准过的那个值, 不另抄一份 —— 两条冷层通路用同一个门,
# 免得同一条记忆在一条路上够格、另一条路上不够格。
WARM_SAMPLE_BUDGET = 3
WARM_SAMPLE_THRESHOLD = _L3_SIMILARITY_FLOOR
# 共同经历 (AI 侧 生活/交互) 专用的更低门 — 仅当用户明确问"我们之间"时启用.
# 0.35 → 0.24: 同上换模型重标 (旧分布 10% 分位).
_RELATIONSHIP_RECALL_THRESHOLD = 0.24
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
    text = str(item.get("text") or item.get("content") or "").strip()
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
    # "我们之间"类提问用原话判定 (共同经历线索通常在原话, 不在指代改写里).
    wants_shared_history = asks_shared_history(message)
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
    # 热层和冷层共用一次嵌入 —— 分别调 search_similar 会对同一段 query 嵌两遍,
    # 而两路并行时 Redis 缓存挡不住同时 miss, 等于每轮给 Ollama 多压一次调用。
    tiers = [(levels, 50)]
    if WARM_SAMPLE_BUDGET > 0:
        tiers.append(([3], WARM_SAMPLE_BUDGET))
    vector_task = search_similar_tiers(
        effective_query, user_id, tiers, workspace_id=workspace_id,
    )
    # L3 有界采样 (AMV-L 的 Sample(T_W)): 冷层不再是完全排除。
    #
    # 之前 L3 被彻底挡在检索之外, 于是降级成了单向悬崖 —— 一条记忆掉下去就再也
    # 回不来, 结果我们不敢降级, 分层名存实亡。给冷层留 κ 个名额后, 降级变成
    # "降低权重"而不是"删除", 惰性衰减才敢真的把东西降下去。
    #
    # 三重约束让它不至于把噪声灌进 prompt:
    #   数量  最多 κ 条 (WARM_SAMPLE_BUDGET), κ=0 即完全退回旧行为
    #   门槛  用比 L1/L2 更高的相似度门 —— 冷记忆要够像才值得翻出来
    #   排序  仍与其他候选一起过 rank + select_context 的 token 预算
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
                logger.info(f"[DEBUG-VEC]   sim={float(r.get('similarity',0)):.3f} '{r.get('content','')[:60]}'")

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

            if int(mem.get("level") or 0) == 3:
                # 冷层用更高的门 —— 只有足够像才值得从冷层翻出来。数量上限已由
                # 检索时的 top_k=WARM_SAMPLE_BUDGET 保证, 这里只管质量。
                if float(mem.get("similarity", 0)) >= WARM_SAMPLE_THRESHOLD:
                    mem["_retrieval_source"] = "warm"
                    _merge_candidate(mem, "warm")
                continue

            sim = float(mem.get("similarity", 0))
            # 关系记忆抢救: 共同经历 (AI 侧 生活/交互) 是叙事长句, 向量相似度天然
            # 偏低, 常卡在 0.50 门下被丢. 而 ranking 的 +0.60 boost 与 context
            # 保护槽都在门之后, 救不回被门拦掉的候选. 用户明确问"我们之间"时, 对
            # 这一小类放宽到更低门, 让 boost/保护槽有机会发挥 (其余记忆门不变).
            threshold = _SIMILARITY_THRESHOLD
            if (
                wants_shared_history
                and mem.get("source") == "ai"
                and mem.get("sub_category") == "交互"
            ):
                threshold = _RELATIONSHIP_RECALL_THRESHOLD
            if sim >= threshold:
                mem["_retrieval_source"] = "vector"
                _merge_candidate(mem, label)

    n_warm = sum(
        1 for m in all_candidates
        if "warm" in str(m.get("_retrieval_source") or "")
    )
    logger.info(
        f"[DEBUG-VEC] after threshold={_SIMILARITY_THRESHOLD}: "
        f"{len(all_candidates)} candidates ({n_warm} from L3 warm sample)"
    )

    # Spec §3.2 step 4 + retrieval v2: rerank by display_score plus lightweight
    # keyword/category/safety boosts. The vector model recalls broadly; these
    # deterministic signals stop critical emotional or literal-topic memories
    # from being buried by generic high-importance facts.
    for m in all_candidates:
        score, reasons = rank_memory_candidate(m, effective_query)
        m["rank_score"] = score
        m["rank_reasons"] = reasons
    all_candidates.sort(key=lambda m: float(m.get("rank_score", 0)), reverse=True)

    # Select complete memories with independent user/AI quotas.
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

    # 候选集 ID 供惰性衰减用作弱使用信号 (lifecycle/value.py 的 α): 进了候选说明
    # 这条记忆至少跟本轮话题沾边, 值得比"完全没人问津"多留一会儿。注入集 (强信号)
    # 调用方从 memories 自取。只带 ID 不带正文, 避免把整个候选集塞进缓存。
    # 聚合类时间问题 ("上次…是几个月前" / "参加过几次…") 需要**穷举**某主题下的所有
    # 事件才能作答, 而注入集只有 10 条。LongMemEval 实测这类题要 k=26~42 才拿得全,
    # 但它们不需要记忆全文 —— 压成"日期 + 短标签"后 400 token 就装得下。
    # 只在命中聚合特征时才建, 其余消息完全不受影响。
    timeline_text = None
    if is_aggregate_time_question(message):
        timeline_text = format_timeline(build_timeline(all_candidates)) or None

    result = {
        "memories": classified_memories if classified_memories else None,
        "memory_strings": memory_strings,
        "graph_context": None,
        "timeline": timeline_text,
        "candidate_ids": [
            mid for mid in (m.get("id") for m in all_candidates) if mid
        ],
    }

    # Cache the result. Phase 2.4: cache write key 必须跟 GET 用同一个 cache_key
    # (effective_query), 否则 enhanced_query 路径 GET 永远 miss → caching 失效.
    #
    # 缓存污染防护: vector arm 失败 (embedding 503 / pgvector 故障) 时本次结果
    # 是"因故障而空", 不是"真的没有相关记忆". 写入缓存会让接下来 5 分钟内同
    # query 全部命中空结果 — 一次 embedding 抖动放大成持续失忆 (生产实测复现).
    # 合法空结果 (检索成功但无候选) 仍然缓存, 避免重复无效搜索.
    if isinstance(vector_results, Exception):
        return result
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
