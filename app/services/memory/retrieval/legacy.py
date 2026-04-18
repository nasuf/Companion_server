"""Memory retrieval service.

Retrieves memories using a combination strategy:
- 5 semantic (vector similarity)
- 3 recent
- 2 high importance
"""

import asyncio
import logging

from app.services.memory.storage import repo as memory_repo
from app.services.memory.lifecycle.decay import increment_mention_count
from app.services.memory.retrieval.vector_search import search_similar

logger = logging.getLogger(__name__)


def _memory_to_dict(m, similarity: float = 0.0) -> dict:
    """Convert a MemoryRecord or Prisma object to a dict."""
    return {
        "id": m.id,
        "content": m.content,
        "summary": m.summary,
        "level": m.level,
        "importance": m.importance,
        "type": m.type,
        "main_category": getattr(m, "mainCategory", None),
        "sub_category": getattr(m, "subCategory", None),
        "created_at": str(m.createdAt),
        "similarity": similarity,
    }


async def retrieve_memories(
    query: str,
    user_id: str,
    semantic_k: int = 5,
    recent_k: int = 3,
    important_k: int = 2,
    workspace_id: str | None = None,
    main_category: str | None = None,
    sub_category: str | None = None,
) -> list[dict]:
    """Retrieve relevant memories using combined strategy.

    Runs all three queries concurrently, then deduplicates.
    """
    # Build tasks — skip semantic search if k=0 or empty query
    async def _semantic():
        if semantic_k > 0 and query:
            return await search_similar(
                query,
                user_id,
                top_k=semantic_k * 2,
                workspace_id=workspace_id,
                main_categories=[main_category] if main_category else None,
                sub_categories=[sub_category] if sub_category else None,
            )
        return []

    semantic_results, recent, important = await asyncio.gather(
        _semantic(),
        memory_repo.find_many(
            where={
                "userId": user_id,
                "workspaceId": workspace_id,
                "isArchived": False,
                **({"mainCategory": main_category} if main_category else {}),
                **({"subCategory": sub_category} if sub_category else {}),
            },
            order={"createdAt": "desc"},
            take=recent_k * 2,
        ),
        memory_repo.find_many(
            where={
                "userId": user_id,
                "workspaceId": workspace_id,
                "isArchived": False,
                **({"mainCategory": main_category} if main_category else {}),
                **({"subCategory": sub_category} if sub_category else {}),
            },
            order={"importance": "desc"},
            take=important_k * 2,
        ),
    )

    seen_ids: set[str] = set()
    results: list[dict] = []

    # 1. Semantic search
    for r in semantic_results[:semantic_k]:
        mid = r["id"]
        if mid not in seen_ids:
            seen_ids.add(mid)
            results.append(r)

    # 2. Recent memories
    for m in recent[:recent_k]:
        if m.id not in seen_ids:
            seen_ids.add(m.id)
            results.append(_memory_to_dict(m))

    # 3. High importance memories
    for m in important[:important_k]:
        if m.id not in seen_ids:
            seen_ids.add(m.id)
            results.append(_memory_to_dict(m))

    # 4. L3 awakening: find similar archived/L3 memories for fuzzy recall
    awakened = await _find_awakening_candidates(
        query,
        user_id,
        seen_ids,
        workspace_id=workspace_id,
        main_category=main_category,
        sub_category=sub_category,
    )
    results.extend(awakened)

    # Increment mention counts for all retrieved memories (fire-and-forget)
    for r in results:
        mid = r.get("id")
        if mid:
            try:
                await increment_mention_count(mid)
            except Exception:
                pass

    return results


async def _find_awakening_candidates(
    query: str,
    user_id: str,
    exclude_ids: set[str],
    workspace_id: str | None = None,
    main_category: str | None = None,
    sub_category: str | None = None,
) -> list[dict]:
    """L3记忆唤醒机制。

    相似度≥0.6的L3记忆 → 标记为模糊回忆，在prompt中以"好像听你提过…"形式出现。
    """
    if not query:
        return []

    try:
        l3_results = await search_similar(
            query,
            user_id,
            top_k=3,
            workspace_id=workspace_id,
            main_categories=[main_category] if main_category else None,
            sub_categories=[sub_category] if sub_category else None,
            levels=[3],
        )
    except Exception:
        return []

    awakened = []
    for r in l3_results:
        mid = r.get("id", "")
        similarity = float(r.get("similarity", 0))
        level = int(r.get("level", 3))

        if mid in exclude_ids or level != 3:
            continue
        if similarity < 0.6:
            continue

        # Mark as fuzzy recall
        content = r.get("summary") or r.get("content", "")
        r["content"] = f"（模糊记忆）好像听你提过：{content}"
        r["awakened"] = True
        awakened.append(r)

    return awakened[:2]  # Max 2 awakened memories


def format_memories_for_prompt(memories: list[dict]) -> list[str]:
    """Format memory dicts into strings suitable for prompt injection."""
    return [
        m.get("summary") or m.get("content", "")
        for m in memories
        if m.get("summary") or m.get("content")
    ]
