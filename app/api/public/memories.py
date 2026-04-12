from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.db import db
from app.models.memory import (
    MemoryResponse,
    MemorySearchRequest,
    MemoryStatsGroup,
    MemoryStatsResponse,
)
from app.services.memory import memory_repo
from app.services.memory.retrieval import retrieve_memories
from app.services.workspace.workspaces import resolve_workspace_id

router = APIRouter(prefix="/memories", tags=["memories"])


async def _compute_stats(
    workspace_id: str | None,
    source: str | None = None,
) -> MemoryStatsResponse:
    """Return raw (level, main_category, sub_category, count) groups.

    Frontend computes cross-filtered counts from these groups.
    """
    if not workspace_id:
        return MemoryStatsResponse(total=0, groups=[])

    tables: list[str] = []
    if source in (None, "user"):
        tables.append("memories_user")
    if source in (None, "ai"):
        tables.append("memories_ai")

    # Aggregate across tables using a dict key
    agg: dict[tuple[int, str, str], int] = {}
    for table in tables:
        rows = await db.query_raw(
            f"""
            SELECT level, main_category, sub_category, COUNT(*)::int AS cnt
            FROM {table}
            WHERE is_archived = FALSE AND workspace_id = $1
            GROUP BY level, main_category, sub_category
            """,
            workspace_id,
        )
        for r in rows:
            key = (int(r["level"]), r.get("main_category") or "未分类", r.get("sub_category") or "其他")
            agg[key] = agg.get(key, 0) + int(r["cnt"])

    groups = [
        MemoryStatsGroup(level=lv, main_category=mc, sub_category=sc, count=cnt)
        for (lv, mc, sc), cnt in agg.items()
    ]
    total = sum(g.count for g in groups)
    return MemoryStatsResponse(total=total, groups=groups)


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
    source: Literal["user", "ai"] | None = None,
):
    """Return raw grouped counts. Frontend computes cross-filtered totals."""
    ws_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    return await _compute_stats(ws_id, source)


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
