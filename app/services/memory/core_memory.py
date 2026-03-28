"""Core L1 memory loading policy."""

from __future__ import annotations

from app.services.memory import memory_repo
from app.services.memory.taxonomy import l1_category_quotas


async def load_core_memory_strings(
    *,
    user_id: str,
    workspace_id: str | None,
    source: str = "user",
) -> list[str]:
    results: list[str] = []
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
            results.append(text)

    if results:
        return results

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
    return [row.summary or row.content for row in rows if (row.summary or row.content)]
