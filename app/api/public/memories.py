from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.ownership import require_memory_owner, require_user_self
from app.db import db
from app.models.memory import (
    MemoryBulkDeleteRequest,
    MemoryBulkDeleteResponse,
    MemoryExportResponse,
    MemoryHygieneRequest,
    MemoryHygieneResponse,
    MemoryResponse,
    MemorySearchRequest,
    MemoryStatsGroup,
    MemoryStatsResponse,
    MemoryUpdateRequest,
    WorkspaceMemoryWipeRequest,
    WorkspaceMemoryWipeResponse,
)
from app.services.memory.lifecycle.quality import derive_memory_quality, serialize_quality
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import log_memory_changelog
from app.services.memory.lifecycle.hygiene import run_memory_hygiene
from app.services.memory.storage import repo as memory_repo
from app.services.memory.retrieval.context_selector import exceeds_injection_limit
from app.services.memory.retrieval.legacy import retrieve_memories
from app.services.workspace.workspaces import resolve_workspace_id

router = APIRouter(prefix="/memories", tags=["memories"])


async def _verify_workspace_owner(workspace_id: str | None, user_id: str) -> None:
    """Reject an explicitly-supplied workspace the caller doesn't own.

    All memory queries are already user_id-scoped (so a foreign workspace_id
    currently yields an empty result), but this turns cross-workspace access
    into an explicit 403 and keeps the guarantee even if a future query path
    forgets the user_id filter. No-op when workspace_id is omitted.
    """
    if not workspace_id:
        return
    workspace = await db.chatworkspace.find_unique(where={"id": workspace_id})
    if not workspace or workspace.userId != user_id:
        raise HTTPException(status_code=403, detail="Not your workspace")


def _serialize_memory(m, quality=None) -> MemoryResponse:
    return MemoryResponse(
        id=m.id,
        user_id=m.userId,
        type=m.type,
        main_category=m.mainCategory,
        sub_category=m.subCategory,
        source=m.source,
        level=m.level,
        content=m.content,
        importance=m.importance,
        created_at=str(m.createdAt),
        quality=serialize_quality(quality),
    )


async def _quality_map(memories: list, include_quality: bool) -> dict[str, object]:
    if not include_quality or not memories:
        return {}
    return await derive_memory_quality(memories)


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
    search: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    include_quality: bool = False,
    _user=Depends(require_user_self),
):
    await _verify_workspace_owner(workspace_id, user_id)
    where: dict = {"userId": user_id, "isArchived": False}
    if workspace_id:
        where["workspaceId"] = workspace_id
    if level is not None:
        where["level"] = level
    if main_category:
        where["mainCategory"] = main_category
    if sub_category:
        where["subCategory"] = sub_category
    if search and search.strip():
        # Substring match on content; case-insensitive.
        # Same UX as admin's adminGetMemories `search` param so the chat
        # Inspector and Agent Overview behave identically.
        where["content"] = {"contains": search.strip(), "mode": "insensitive"}

    memories = await memory_repo.find_many(
        source=source,
        where=where,
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )
    qualities = await _quality_map(memories, include_quality)
    return [_serialize_memory(m, qualities.get(m.id)) for m in memories]


@router.get("/export", response_model=MemoryExportResponse)
async def export_memories(
    user_id: str,
    workspace_id: str | None = None,
    source: Literal["user", "ai"] | None = None,
    include_quality: bool = False,
    _user=Depends(require_user_self),
):
    await _verify_workspace_owner(workspace_id, user_id)
    ws_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    memories = await memory_repo.find_many(
        source=source,
        where={"userId": user_id, "workspaceId": ws_id, "isArchived": False},
        order={"createdAt": "desc"},
    )
    qualities = await _quality_map(memories, include_quality)
    return MemoryExportResponse(
        user_id=user_id,
        workspace_id=ws_id,
        total=len(memories),
        memories=[_serialize_memory(m, qualities.get(m.id)) for m in memories],
    )


@router.get("/stats", response_model=MemoryStatsResponse)
async def memory_stats(
    user_id: str,
    workspace_id: str | None = None,
    source: Literal["user", "ai"] | None = None,
    _user=Depends(require_user_self),
):
    """Return raw grouped counts. Frontend computes cross-filtered totals."""
    await _verify_workspace_owner(workspace_id, user_id)
    ws_id = workspace_id or await resolve_workspace_id(user_id=user_id)
    return await _compute_stats(ws_id, source)


@router.post("/search")
async def search_memories(
    data: MemorySearchRequest,
    user_id: str = Query(...),
    _user=Depends(require_user_self),
):
    await _verify_workspace_owner(data.workspace_id, user_id)
    results = await retrieve_memories(
        data.query,
        user_id=user_id,
        semantic_k=data.top_k,
        workspace_id=data.workspace_id,
        main_category=data.main_category,
        sub_category=data.sub_category,
    )
    return results


@router.post("/hygiene", response_model=MemoryHygieneResponse)
async def run_memory_hygiene_now(
    data: MemoryHygieneRequest,
    user_id: str = Query(...),
    _user=Depends(require_user_self),
):
    await _verify_workspace_owner(data.workspace_id, user_id)
    workspace_id = data.workspace_id or await resolve_workspace_id(user_id=user_id)
    return await run_memory_hygiene(
        user_id=user_id,
        workspace_id=workspace_id,
        allow_llm=data.allow_llm,
        max_scopes=2,
        max_memories_per_scope=data.max_memories_per_scope,
    )


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    data: MemoryUpdateRequest,
    m=Depends(require_memory_owner),
):
    if m.isArchived:
        raise HTTPException(status_code=404, detail="Memory not found")

    update_data: dict = {}
    old_content = m.content
    if data.content is not None:
        # 超限的内容存进去也永远不会被检索到 —— 用户会看到"保存成功", 得到的却是一条
        # AI 再也想不起来的记忆。跟 knowledge 导入 (agent_template/knowledge.py) 同样的
        # 取舍: 明确拒绝并把原文留给用户自己拆, 好过系统擅自截断或静默收下。
        if exceeds_injection_limit(data.content):
            # detail 跟本文件其余错误一样用可读文案 (不是机器码): 这个接口的 detail
            # 会直接透到用户面前, 而"内容太长"恰好是用户自己能处理的问题。
            raise HTTPException(
                status_code=400,
                detail="Memory content is too long to be recalled; split it into shorter entries",
            )
        update_data["content"] = data.content
        await store_embedding(m.id, await generate_embedding(data.content))
    if not update_data:
        return _serialize_memory(m)

    await memory_repo.update(m.id, source=m.source, record=m, **update_data)
    await log_memory_changelog(
        m.userId,
        m.id,
        "user_edit",
        old_value=old_content,
        new_value=update_data.get("content", m.content),
        workspace_id=m.workspaceId,
    )
    updated = await memory_repo.find_unique(m.id)
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _serialize_memory(updated)


@router.post("/bulk-delete", response_model=MemoryBulkDeleteResponse)
async def bulk_delete_memories(
    data: MemoryBulkDeleteRequest,
    user_id: str = Query(...),
    _user=Depends(require_user_self),
):
    requested = list(dict.fromkeys(data.memory_ids))
    archived = 0
    missing_or_forbidden: list[str] = []
    for memory_id in requested:
        rec = await memory_repo.find_unique(memory_id)
        if not rec or rec.userId != user_id or rec.isArchived:
            missing_or_forbidden.append(memory_id)
            continue
        await memory_repo.update(
            rec.id,
            source=rec.source,
            record=rec,
            isArchived=True,
        )
        await log_memory_changelog(
            rec.userId,
            rec.id,
            "user_bulk_delete",
            old_value=rec.content,
            new_value=None,
            workspace_id=rec.workspaceId,
        )
        archived += 1
    return MemoryBulkDeleteResponse(
        requested=len(requested),
        archived=archived,
        missing_or_forbidden=missing_or_forbidden,
    )


@router.post("/workspace-wipe", response_model=WorkspaceMemoryWipeResponse)
async def wipe_workspace_memories(
    data: WorkspaceMemoryWipeRequest,
    user_id: str = Query(...),
    _user=Depends(require_user_self),
):
    workspace = await db.chatworkspace.find_unique(where={"id": data.workspace_id})
    if not workspace or workspace.userId != user_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if not data.include_user and not data.include_ai:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one memory source must be selected",
        )

    archived_user = 0
    archived_ai = 0
    where = {
        "userId": user_id,
        "workspaceId": data.workspace_id,
        "isArchived": False,
    }
    if data.include_user:
        records = await memory_repo.find_many(source="user", where=where)
        archived_user = await memory_repo.update_many(
            source="user",
            where=where,
            data={"isArchived": True},
        )
        for rec in records:
            await log_memory_changelog(
                rec.userId,
                rec.id,
                "workspace_wipe",
                old_value=rec.content,
                new_value=None,
                workspace_id=rec.workspaceId,
            )
    if data.include_ai:
        records = await memory_repo.find_many(source="ai", where=where)
        archived_ai = await memory_repo.update_many(
            source="ai",
            where=where,
            data={"isArchived": True},
        )
        for rec in records:
            await log_memory_changelog(
                rec.userId,
                rec.id,
                "workspace_wipe",
                old_value=rec.content,
                new_value=None,
                workspace_id=rec.workspaceId,
            )
    return WorkspaceMemoryWipeResponse(
        workspace_id=data.workspace_id,
        archived_user=archived_user,
        archived_ai=archived_ai,
    )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    include_quality: bool = False,
    m=Depends(require_memory_owner),
):
    qualities = await _quality_map([m], include_quality)
    return _serialize_memory(m, qualities.get(m.id))
