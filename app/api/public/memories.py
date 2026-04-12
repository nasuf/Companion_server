from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.db import db
from app.models.memory import (
    MemoryResponse,
    MemorySearchRequest,
    MemoryStatsBucket,
    MemoryStatsResponse,
)
from app.services.memory import memory_repo
from app.services.memory.retrieval import retrieve_memories
from app.services.workspace.workspaces import resolve_workspace_id

router = APIRouter(prefix="/memories", tags=["memories"])


async def _compute_stats(
    workspace_id: str | None,
    source: str | None = None,
    level: int | None = None,
    main_category: str | None = None,
    sub_category: str | None = None,
) -> MemoryStatsResponse:
    """Compute precise memory stats via SQL COUNT + GROUP BY.

    Accepts optional filter params so dropdown counts reflect cross-filters.
    """
    if not workspace_id:
        return MemoryStatsResponse(
            total=0, by_level=[], by_main_category=[], by_sub_category=[], by_main_sub={},
        )

    tables: list[str] = []
    if source in (None, "user"):
        tables.append("memories_user")
    if source in (None, "ai"):
        tables.append("memories_ai")

    conditions = ["is_archived = FALSE", "workspace_id = $1"]
    params: list = [workspace_id]
    idx = 2
    if level is not None:
        conditions.append(f"level = ${idx}")
        params.append(level)
        idx += 1
    if main_category:
        conditions.append(f"main_category = ${idx}")
        params.append(main_category)
        idx += 1
    if sub_category:
        conditions.append(f"sub_category = ${idx}")
        params.append(sub_category)
        idx += 1

    where_clause = " AND ".join(conditions)

    by_level: dict[str, int] = {}
    by_main: dict[str, int] = {}
    by_main_sub: dict[str, int] = {}
    total = 0

    for table in tables:
        rows = await db.query_raw(
            f"""
            SELECT level, main_category, sub_category, COUNT(*)::int AS cnt
            FROM {table}
            WHERE {where_clause}
            GROUP BY level, main_category, sub_category
            """,
            *params,
        )
        for r in rows:
            cnt = int(r["cnt"])
            total += cnt
            lk = f"L{r['level']}"
            mk = r.get("main_category") or "未分类"
            sk = r.get("sub_category") or "其他"
            by_level[lk] = by_level.get(lk, 0) + cnt
            by_main[mk] = by_main.get(mk, 0) + cnt
            by_main_sub[f"{mk}-{sk}"] = by_main_sub.get(f"{mk}-{sk}", 0) + cnt

    def _ser(d: dict[str, int]) -> list[MemoryStatsBucket]:
        return [MemoryStatsBucket(key=k, count=c) for k, c in sorted(d.items(), key=lambda x: (-x[1], x[0]))]

    return MemoryStatsResponse(
        total=total,
        by_level=_ser(by_level),
        by_main_category=_ser(by_main),
        by_sub_category=[],
        by_main_sub=by_main_sub,
    )


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
    level: int | None = None,
    main_category: str | None = None,
    sub_category: str | None = None,
):
    """Precise memory statistics via SQL COUNT + GROUP BY."""
    ws_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    return await _compute_stats(ws_id, source, level, main_category, sub_category)


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
