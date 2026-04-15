"""Memory consolidation service.

Clusters similar memories and creates summaries.
Daily: L3 > 72h -> merge to L2
Weekly: L2 patterns -> L1
"""

import logging
from datetime import datetime, timedelta, timezone

from app.services.memory import memory_repo
from app.services.memory.lifecycle import compute_dynamic_weight
from app.services.memory.embedding import generate_embedding, store_embedding
from app.services.memory.taxonomy import (
    get_compression_rule,
    get_promotion_rule,
    summarize_batch_taxonomy,
)
from app.services.memory.vector_search import search_by_embedding
from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """请把下面这些记忆整合成一条稳定的长期中文记忆。
保持简洁，控制在 1-2 句。

记忆：
{memories}

要求：
- 输出必须是自然中文
- 不要输出英文总结，除非专有名词本身是英文

总结："""


async def _consolidate_level(
    source_level: int,
    target_level: int,
    cutoff: datetime,
    take: int,
) -> None:
    """Shared consolidation logic for any level transition."""
    old_memories = await memory_repo.find_many(
        where={
            "level": source_level,
            "isArchived": False,
            "createdAt": {"lt": cutoff},
        },
        order={"createdAt": "asc"},
        take=take,
        allow_cross_user=True,
    )

    if not old_memories:
        logger.info(f"No L{source_level} memories to consolidate")
        return

    user_memories: dict[str, list] = {}
    for m in old_memories:
        user_memories.setdefault(m.userId, []).append(m)

    for user_id, memories in user_memories.items():
        await _consolidate_user_memories(user_id, memories, target_level=target_level)


async def consolidate_daily():
    """Daily consolidation: merge old L3 memories into L2."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    await _consolidate_level(source_level=3, target_level=2, cutoff=cutoff, take=100)


async def consolidate_weekly():
    """Weekly consolidation: merge L2 patterns into L1."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    await _consolidate_level(source_level=2, target_level=1, cutoff=cutoff, take=50)

    # Check for L2→L1 upgrade candidates
    await check_l2_upgrade_candidates()
    # Check for L2→L3 demotion candidates
    await check_l2_demotion_candidates()


# PRD要求年提及≥10次，但初期数据稀疏时门槛太高。
# 折中：至少被提及3次才允许升级，确保记忆确实被用户反复关注。
MIN_MENTION_COUNT_FOR_L1 = 3


async def check_l2_upgrade_candidates() -> None:
    """检测满足升级条件的L2记忆并升级到L1。

    条件:
      1. 综合分数(importance × time_factor × freq_factor) ≥ 0.85
      2. mentionCount ≥ MIN_MENTION_COUNT_FOR_L1 (PRD §3.7.3 频率门槛)
      3. category规则允许
      4. 同类L1不超过5条
    """
    now = datetime.now(timezone.utc)
    candidates = await memory_repo.find_many(
        where={"level": 2, "isArchived": False},
        take=100,
        allow_cross_user=True,
    )

    if not candidates:
        return

    for mem in candidates:
        months_age = (now - mem.createdAt.replace(tzinfo=timezone.utc if mem.createdAt.tzinfo is None else mem.createdAt.tzinfo)).total_seconds() / 86400 / 30
        score = compute_dynamic_weight(mem.importance, months_age, mem.mentionCount)
        if score < 0.85:
            continue

        # PRD §3.7.3: 频率门槛 — 记忆需被多次提及才有资格升级
        if mem.mentionCount < MIN_MENTION_COUNT_FOR_L1:
            continue

        rule = get_promotion_rule(mem.mainCategory, mem.subCategory)
        if not bool(rule.get("allow_l1", False)):
            continue

        # 检查同主题L1数量
        existing_l1 = await memory_repo.find_many(
            source=mem.source,
            where={
                "userId": mem.userId,
                "level": 1,
                "isArchived": False,
                "mainCategory": mem.mainCategory,
                "subCategory": mem.subCategory,
            },
            take=5,
        )
        if len(existing_l1) >= 5:
            continue

        await memory_repo.update(mem.id, source=mem.source, record=mem, level=1)
        logger.info(f"Upgraded L2→L1: {mem.id} (score={score:.2f}, mentions={mem.mentionCount})")


async def check_l2_demotion_candidates() -> None:
    """检测满足降级条件的L2记忆并降级到L3。

    条件: 综合分数 < 0.49 AND 创建时间 > 30天。
    """
    from app.services.memory.storage import log_memory_changelog

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)

    candidates = await memory_repo.find_many(
        where={
            "level": 2,
            "isArchived": False,
            "createdAt": {"lt": cutoff},
        },
        take=100,
        allow_cross_user=True,
    )

    if not candidates:
        return

    demoted = 0
    for mem in candidates:
        months_age = (now - mem.createdAt.replace(tzinfo=timezone.utc if mem.createdAt.tzinfo is None else mem.createdAt.tzinfo)).total_seconds() / 86400 / 30
        score = compute_dynamic_weight(mem.importance, months_age, mem.mentionCount)
        if score >= 0.49:
            continue

        await memory_repo.update(mem.id, source=mem.source, record=mem, level=3)
        try:
            await log_memory_changelog(
                mem.userId, mem.id, "demote",
                old_value=f"L2 (score={score:.2f})",
                new_value="L3",
            )
        except Exception:
            pass
        demoted += 1

    if demoted:
        logger.info(f"Demoted {demoted} L2→L3 memories (score<0.49, age>30d)")


async def _consolidate_user_memories(
    user_id: str,
    memories: list,
    target_level: int,
) -> None:
    """Cluster and merge similar memories for a user."""
    # Pre-compute embeddings and search results once per memory (O(n) not O(n^2))
    search_results: dict[str, list[dict]] = {}
    for mem in memories:
        embedding = await generate_embedding(mem.content)
        results = await search_by_embedding(embedding, user_id, top_k=10)
        search_results[mem.id] = results

    # Cluster similar memories using pre-computed results
    clusters: list[list] = []
    used: set[str] = set()

    for i, mem_a in enumerate(memories):
        if mem_a.id in used:
            continue

        cluster = [mem_a]
        used.add(mem_a.id)

        # Check which other memories are similar using pre-computed search results
        similar_ids = {
            r.get("id")
            for r in search_results.get(mem_a.id, [])
            if float(r.get("similarity", 0)) > 0.85
        }

        for j in range(i + 1, len(memories)):
            mem_b = memories[j]
            if mem_b.id in used:
                continue
            if mem_b.id in similar_ids:
                cluster.append(mem_b)
                used.add(mem_b.id)

        if len(cluster) >= 2:
            clusters.append(cluster)

    # Merge each cluster
    model = get_utility_model()
    for cluster in clusters:
        main_category, sub_category = summarize_batch_taxonomy(cluster)
        compression_rule = get_compression_rule(main_category)
        if not bool(compression_rule.get("allow_cross_subcategory", True)):
            unique_sub_categories = {
                getattr(item, "subCategory", None) or "其他"
                for item in cluster
            }
            if len(unique_sub_categories) > 1:
                continue
        if len(cluster) < int(compression_rule.get("batch_size", 2)):
            continue

        mem_texts = "\n".join(f"- {m.content}" for m in cluster)
        prompt = (await get_prompt_text("memory.consolidation")).format(memories=mem_texts)

        try:
            summary = await invoke_text(model, prompt)
            summary = summary.strip()

            max_importance = max(m.importance for m in cluster)
            # Inherit source from cluster — if any member is "ai", mark consolidated as "ai"
            cluster_source = "ai" if any(m.source == "ai" for m in cluster) else "user"
            merged = await memory_repo.create(
                source=cluster_source,
                userId=user_id,
                content=summary,
                summary=summary,
                level=target_level,
                importance=min(1.0, max_importance + 0.1),
                type="consolidated",
                mainCategory=main_category,
                subCategory=sub_category,
            )

            embedding = await generate_embedding(summary)
            await store_embedding(merged.id, embedding)

            # Batch archive old memories (source=None to cover mixed-source clusters)
            cluster_ids = [m.id for m in cluster]
            await memory_repo.update_many(
                where={"id": {"in": cluster_ids}},
                data={"isArchived": True},
            )

            logger.info(f"Consolidated {len(cluster)} memories into L{target_level}: {summary[:50]}")

        except Exception as e:
            logger.error(f"Consolidation merge failed: {e}")
