from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.models.memory import (
    MemoryResponse,
    MemorySearchRequest,
    MemoryStatsBucket,
    MemoryStatsResponse,
)
from app.services.memory import memory_repo
from app.services.memory.retrieval import retrieve_memories

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("", response_model=list[MemoryResponse])
async def list_memories(
    user_id: str,
    workspace_id: str | None = None,
    level: int | None = None,
    main_category: str | None = None,
    sub_category: str | None = None,
    source: Literal["user", "ai"] | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    where: dict = {"userId": user_id, "isArchived": False}
    if workspace_id:
        where["workspaceId"] = workspace_id
    if level is not None:
        where["level"] = level
    if main_category:
        where["mainCategory"] = main_category
    if sub_category:
        where["subCategory"] = sub_category

    memories = await memory_repo.find_many(
        source=source,
        where=where,
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )
    return [
        MemoryResponse(
            id=m.id,
            user_id=m.userId,
            type=m.type,
            main_category=m.mainCategory,
            sub_category=m.subCategory,
            source=m.source,
            level=m.level,
            content=m.content,
            summary=m.summary,
            importance=m.importance,
            created_at=str(m.createdAt),
        )
        for m in memories
    ]


@router.get("/stats", response_model=MemoryStatsResponse)
async def memory_stats(
    user_id: str,
    workspace_id: str | None = None,
    level: int | None = None,
    main_category: str | None = None,
    sub_category: str | None = None,
    source: Literal["user", "ai"] | None = None,
):
    where: dict = {"userId": user_id, "isArchived": False}
    if workspace_id:
        where["workspaceId"] = workspace_id
    if level is not None:
        where["level"] = level
    if main_category:
        where["mainCategory"] = main_category
    if sub_category:
        where["subCategory"] = sub_category

    memories = await memory_repo.find_many(source=source, where=where)

    by_level: dict[str, int] = {}
    by_main_category: dict[str, int] = {}
    by_sub_category: dict[str, int] = {}

    for memory in memories:
        level_key = f"L{memory.level}"
        main_key = memory.mainCategory or "未分类"
        sub_key = memory.subCategory or "其他"
        by_level[level_key] = by_level.get(level_key, 0) + 1
        by_main_category[main_key] = by_main_category.get(main_key, 0) + 1
        by_sub_category[sub_key] = by_sub_category.get(sub_key, 0) + 1

    def _serialize(data: dict[str, int]) -> list[MemoryStatsBucket]:
        return [
            MemoryStatsBucket(key=key, count=count)
            for key, count in sorted(data.items(), key=lambda item: (-item[1], item[0]))
        ]

    return MemoryStatsResponse(
        total=len(memories),
        by_level=_serialize(by_level),
        by_main_category=_serialize(by_main_category),
        by_sub_category=_serialize(by_sub_category),
    )


@router.post("/search")
async def search_memories(data: MemorySearchRequest, user_id: str = Query(default="")):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    results = await retrieve_memories(
        data.query,
        user_id=user_id,
        semantic_k=data.top_k,
        workspace_id=data.workspace_id,
        main_category=data.main_category,
        sub_category=data.sub_category,
    )
    return results


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    m = await memory_repo.find_unique(memory_id)
    if not m:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=m.id,
        user_id=m.userId,
        type=m.type,
        main_category=m.mainCategory,
        sub_category=m.subCategory,
        source=m.source,
        level=m.level,
        content=m.content,
        summary=m.summary,
        importance=m.importance,
        created_at=str(m.createdAt),
    )
