"""Memory consolidation service.

Clusters similar memories and creates summaries.
Daily: L3 > 72h -> merge to L2
Weekly: L2 patterns -> L1
"""

import logging
from datetime import datetime, timedelta, timezone

from app.db import db
from app.services.memory.embedding import generate_embedding, store_embedding
from app.services.memory.vector_search import search_by_embedding
from app.services.llm.models import get_utility_model, invoke_text

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Summarize the following memories into a single stable long-term memory.
Be concise (1-2 sentences).

Memories:
{memories}

Summary:"""


async def _consolidate_level(
    source_level: int,
    target_level: int,
    cutoff: datetime,
    take: int,
) -> None:
    """Shared consolidation logic for any level transition."""
    old_memories = await db.memory.find_many(
        where={
            "level": source_level,
            "isArchived": False,
            "createdAt": {"lt": cutoff},
        },
        order={"createdAt": "asc"},
        take=take,
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
        mem_texts = "\n".join(f"- {m.content}" for m in cluster)
        prompt = SUMMARIZE_PROMPT.format(memories=mem_texts)

        try:
            summary = await invoke_text(model, prompt)
            summary = summary.strip()

            max_importance = max(m.importance for m in cluster)
            merged = await db.memory.create(
                data={
                    "user": {"connect": {"id": user_id}},
                    "content": summary,
                    "summary": summary,
                    "level": target_level,
                    "importance": min(1.0, max_importance + 0.1),
                    "type": "consolidated",
                }
            )

            embedding = await generate_embedding(summary)
            await store_embedding(merged.id, embedding)

            # Batch archive old memories
            cluster_ids = [m.id for m in cluster]
            await db.memory.update_many(
                where={"id": {"in": cluster_ids}},
                data={"isArchived": True},
            )

            logger.info(f"Consolidated {len(cluster)} memories into L{target_level}: {summary[:50]}")

        except Exception as e:
            logger.error(f"Consolidation merge failed: {e}")
