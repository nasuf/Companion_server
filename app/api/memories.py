from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.models.memory import MemoryResponse, MemorySearchRequest
from app.services.memory import memory_repo
from app.services.memory.retrieval import retrieve_memories

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("", response_model=list[MemoryResponse])
async def list_memories(
    user_id: str,
    level: int | None = None,
    source: Literal["user", "ai"] | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    where: dict = {"userId": user_id, "isArchived": False}
    if level is not None:
        where["level"] = level

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
            source=m.source,
            level=m.level,
            content=m.content,
            summary=m.summary,
            importance=m.importance,
            created_at=str(m.createdAt),
        )
        for m in memories
    ]


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    m = await memory_repo.find_unique(memory_id)
    if not m:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=m.id,
        user_id=m.userId,
        type=m.type,
        source=m.source,
        level=m.level,
        content=m.content,
        summary=m.summary,
        importance=m.importance,
        created_at=str(m.createdAt),
    )


@router.post("/search")
async def search_memories(data: MemorySearchRequest, user_id: str = Query(default="")):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    results = await retrieve_memories(data.query, user_id=user_id, semantic_k=data.top_k)
    return results
