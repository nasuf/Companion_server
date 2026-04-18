"""Core L1 memory loading policy.

Returns (main_category, text) tuples so prompt_builder can split identity
facts (hard constraints the LLM must never contradict) from other memories
(soft references the LLM can weave in naturally).
"""

from __future__ import annotations

from app.services.memory.storage import repo as memory_repo
from app.services.memory.taxonomy import l1_category_quotas


async def load_core_memory_strings(
    *,
    user_id: str,
    workspace_id: str | None,
    source: str = "user",
) -> list[tuple[str, str]]:
    """Load L1 memories as ``(main_category, summary_text)`` pairs.

    Iterates categories in quota order (身份 first), so identity facts are
    always at the front of the returned list.  Falls back to a flat top-20
    query if per-category search returns nothing (e.g. brand-new agent with
    no taxonomy tags yet).
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for category, limit in l1_category_quotas():
        rows = await memory_repo.find_many(
            source=source,
            where={
                "userId": user_id,
                "workspaceId": workspace_id,
                "level": 1,
                "isArchived": False,
                "mainCategory": category,
            },
            order={"importance": "desc"},
            take=limit,
        )
        for row in rows:
            text = row.summary or row.content
            if not text or row.id in seen:
                continue
            seen.add(row.id)
            results.append((category, text))

    if results:
        return results

    # Fallback: no per-category rows (legacy data without mainCategory).
    rows = await memory_repo.find_many(
        source=source,
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "level": 1,
            "isArchived": False,
        },
        order={"importance": "desc"},
        take=20,
    )
    return [
        (getattr(row, "mainCategory", "") or "生活", row.summary or row.content)
        for row in rows
        if (row.summary or row.content)
    ]
