"""Unified memory repository layer.

Routes CRUD operations to memories_user or memories_ai tables.
Provides a consistent MemoryRecord interface regardless of source.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from app.db import db
from app.services.runtime.cache import bump_cache_version
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

Source = Literal["user", "ai"]


@dataclass
class MemoryRecord:
    id: str
    userId: str
    type: str | None
    source: Source
    level: int
    content: str
    summary: str | None
    importance: float
    mentionCount: int
    isArchived: bool
    occurTime: datetime | None
    createdAt: datetime
    updatedAt: datetime
    mainCategory: str | None = None
    subCategory: str | None = None
    workspaceId: str | None = None


def _to_record(row, source: Source) -> MemoryRecord:
    """Convert a Prisma model instance to MemoryRecord."""
    return MemoryRecord(
        id=row.id,
        userId=row.userId,
        type=row.type,
        mainCategory=getattr(row, "mainCategory", None),
        subCategory=getattr(row, "subCategory", None),
        source=source,
        level=row.level,
        content=row.content,
        summary=row.summary,
        importance=row.importance,
        mentionCount=row.mentionCount,
        isArchived=row.isArchived,
        occurTime=getattr(row, "occurTime", None),
        createdAt=row.createdAt,
        updatedAt=row.updatedAt,
        workspaceId=getattr(row, "workspaceId", None),
    )


def _table(source: Source):
    """Return the Prisma table accessor for the given source."""
    if source == "ai":
        return db.aimemory
    if source == "user":
        return db.usermemory
    raise ValueError(f"Invalid source: {source!r}, must be 'user' or 'ai'")


def _build_kwargs(
    where: dict | None = None,
    order: dict | None = None,
    take: int | None = None,
    skip: int | None = None,
) -> dict:
    kwargs: dict = {}
    if where is not None:
        kwargs["where"] = where
    if order is not None:
        kwargs["order"] = order
    if take is not None:
        kwargs["take"] = take
    if skip is not None:
        kwargs["skip"] = skip
    return kwargs


# --- CRUD ---


async def create(source: Source = "user", **data) -> MemoryRecord:
    """Create a memory in the appropriate table.

    Pass fields as keyword args: userId, content, summary, level, importance, type, etc.
    Do NOT pass 'source' in data — it's determined by the `source` parameter.
    """
    if data.get("userId") and not data.get("workspaceId"):
        data["workspaceId"] = await resolve_workspace_id(user_id=data["userId"])
    row = await _table(source).create(data=data)
    await _invalidate_caches(data.get("userId"), data.get("workspaceId"))
    return _to_record(row, source)


async def _invalidate_caches(user_id: str | None, workspace_id: str | None) -> None:
    """Bump per-(user, workspace) cache version so stale retrieval/graph
    results are never served. Best-effort: Redis hiccups must not fail the
    write path (caller has already persisted to Postgres)."""
    if not user_id:
        return
    try:
        await bump_cache_version(user_id, workspace_id)
    except Exception as e:
        logger.debug(f"cache bump failed for {user_id}/{workspace_id}: {e}")


async def _scope_where(where: dict | None, *, allow_cross_user: bool = False) -> dict | None:
    """Attach workspaceId filter so per-user queries never leak across workspaces.

    Callers MUST include `userId` unless they:
      - already pre-scoped via `workspaceId`
      - target memories by primary key (`id` / `id in [...]`)
      - pass `allow_cross_user=True` (batch/admin jobs like consolidation)

    Otherwise raise — a bare `where={"level": 1}` is a bug that would scan
    every workspace.
    """
    if not where:
        # Preserve legacy behaviour: empty/None means "no constraint".
        # Callers who want cross-user access must pass allow_cross_user=True
        # with a non-empty where (or accept that bare empty scans nothing
        # special beyond the table). Keep historical API surface intact.
        return where

    # If workspaceId is already specified, respect it and do not overwrite —
    # callers that pass both userId and workspaceId have done the scoping
    # themselves and may be targeting an explicit legacy/migration workspace.
    if "workspaceId" in where:
        return where

    if "userId" in where:
        scoped = dict(where)
        scoped["workspaceId"] = await resolve_workspace_id(user_id=scoped["userId"])
        return scoped

    # Queries keyed by primary id are inherently scoped.
    if "id" in where:
        return where

    if allow_cross_user:
        return where

    raise ValueError(
        "memory_repo queries must include userId / workspaceId / id, "
        "or pass allow_cross_user=True for batch jobs "
        "(prevents cross-workspace data leakage)"
    )


async def find_many(
    source: Source | None = None,
    where: dict | None = None,
    order: dict | None = None,
    take: int | None = None,
    skip: int | None = None,
    allow_cross_user: bool = False,
) -> list[MemoryRecord]:
    """Query memories. source=None queries both tables and merges results.

    Set allow_cross_user=True only for admin/batch jobs (consolidation,
    lifecycle) that intentionally span users.
    """
    where = await _scope_where(where, allow_cross_user=allow_cross_user)
    kwargs = _build_kwargs(where, order, take, skip)

    if source is not None:
        rows = await _table(source).find_many(**kwargs)
        return [_to_record(r, source) for r in rows]

    # Both tables — skip/take applied after merge, not per-table
    both_kwargs = _build_kwargs(where, order, take=((skip or 0) + take) if take is not None else None)
    user_rows, ai_rows = await asyncio.gather(
        db.usermemory.find_many(**both_kwargs),
        db.aimemory.find_many(**both_kwargs),
    )

    records = [_to_record(r, "user") for r in user_rows] + [
        _to_record(r, "ai") for r in ai_rows
    ]

    # Re-sort merged results
    if order:
        key = next(iter(order))
        reverse = order[key] == "desc"
        records.sort(key=lambda r: getattr(r, key), reverse=reverse)

    # Apply skip + take on merged results
    start = skip or 0
    if take is not None:
        records = records[start : start + take]
    elif start:
        records = records[start:]

    return records


async def find_unique(id: str) -> MemoryRecord | None:
    """Find a memory by ID, checking both tables concurrently."""
    user_row, ai_row = await asyncio.gather(
        db.usermemory.find_unique(where={"id": id}),
        db.aimemory.find_unique(where={"id": id}),
    )
    if user_row:
        return _to_record(user_row, "user")
    if ai_row:
        return _to_record(ai_row, "ai")
    return None


async def count(
    source: Source | None = None,
    where: dict | None = None,
    allow_cross_user: bool = False,
) -> int:
    """Count memories. source=None counts both tables."""
    kwargs: dict = {}
    if where is not None:
        kwargs["where"] = await _scope_where(where, allow_cross_user=allow_cross_user)

    if source is not None:
        return await _table(source).count(**kwargs)

    user_count, ai_count = await asyncio.gather(
        db.usermemory.count(**kwargs),
        db.aimemory.count(**kwargs),
    )
    return user_count + ai_count


async def update(
    id: str,
    source: Source | None = None,
    *,
    record: MemoryRecord | None = None,
    **data,
) -> None:
    """Update a memory by ID. If source unknown, auto-detect.

    Callers that already hold the pre-update `MemoryRecord` (e.g. a
    consolidation/conflict job that just fetched the row) can pass it via
    `record=` to skip the extra `find_unique` round-trip. Only `userId` /
    `workspaceId` / `source` are read from the record — all write fields
    come from **data.
    """
    if record is None:
        record = await find_unique(id)
    if record is None:
        logger.warning(f"Memory {id} not found for update")
        return
    if source is None:
        source = record.source

    await _table(source).update(where={"id": id}, data=data)
    # userId/workspaceId don't change on update; safe to bump from record
    await _invalidate_caches(record.userId, record.workspaceId)


async def update_many(
    source: Source | None = None,
    where: dict | None = None,
    data: dict | None = None,
    allow_cross_user: bool = False,
) -> int:
    """Batch update. source=None updates both tables.

    Cache invalidation: collect the distinct (userId, workspaceId) pairs
    that will be affected BEFORE the update. We only support cache bumping
    for the common case `where={"id": {"in": [...]}}` (what every caller
    in this codebase uses); for more exotic where clauses we fall back to
    no-bump rather than loading the whole table.
    """
    kwargs_where = await _scope_where(where or {}, allow_cross_user=allow_cross_user)
    kwargs_data = data or {}

    scopes = await _scopes_for_update_many(source, kwargs_where)

    if source is not None:
        total = await _table(source).update_many(where=kwargs_where, data=kwargs_data)
    else:
        user_count, ai_count = await asyncio.gather(
            db.usermemory.update_many(where=kwargs_where, data=kwargs_data),
            db.aimemory.update_many(where=kwargs_where, data=kwargs_data),
        )
        total = user_count + ai_count

    for user_id, workspace_id in scopes:
        await _invalidate_caches(user_id, workspace_id)
    return total


async def _scopes_for_update_many(
    source: Source | None, where: dict | None,
) -> set[tuple[str, str | None]]:
    """Return the distinct (userId, workspaceId) pairs that update_many will
    touch, for cache invalidation. Only handles the id-IN form used in this
    codebase; other shapes return empty (skipping the bump)."""
    if not where:
        return set()
    id_filter = where.get("id")
    if not isinstance(id_filter, dict):
        return set()
    ids = id_filter.get("in")
    if not ids:
        return set()

    sources: list[Source] = [source] if source is not None else ["user", "ai"]
    scopes: set[tuple[str, str | None]] = set()
    for s in sources:
        rows = await _table(s).find_many(
            where={"id": {"in": list(ids)}},
        )
        for r in rows:
            scopes.add((r.userId, getattr(r, "workspaceId", None)))
    return scopes


async def delete(id: str, source: Source | None = None) -> None:
    """Delete a memory by ID, cascading to its embedding.

    Order is intentional: memory row is deleted first so that even if the
    vector-row delete fails, we never leave a memory-less orphan embedding
    with the same id in a surprising visibility state (any reinsert of a
    new memory with the same uuid would collide with the stale vector).
    The embedding delete tolerates absence (WHERE matches 0 rows is OK).
    """
    rec = await find_unique(id)
    if not rec:
        return
    if source is None:
        source = rec.source

    await _table(source).delete(where={"id": id})
    try:
        await db.execute_raw(
            "DELETE FROM memory_embeddings WHERE memory_id = $1", id,
        )
    except Exception as e:
        logger.warning(f"Embedding cleanup failed for memory {id}: {e}")
    await _invalidate_caches(rec.userId, rec.workspaceId)
