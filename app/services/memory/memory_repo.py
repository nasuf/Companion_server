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


def _to_record(row, source: Source) -> MemoryRecord:
    """Convert a Prisma model instance to MemoryRecord."""
    return MemoryRecord(
        id=row.id,
        userId=row.userId,
        type=row.type,
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
    row = await _table(source).create(data=data)
    return _to_record(row, source)


async def find_many(
    source: Source | None = None,
    where: dict | None = None,
    order: dict | None = None,
    take: int | None = None,
    skip: int | None = None,
) -> list[MemoryRecord]:
    """Query memories. source=None queries both tables and merges results."""
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
) -> int:
    """Count memories. source=None counts both tables."""
    kwargs: dict = {}
    if where is not None:
        kwargs["where"] = where

    if source is not None:
        return await _table(source).count(**kwargs)

    user_count, ai_count = await asyncio.gather(
        db.usermemory.count(**kwargs),
        db.aimemory.count(**kwargs),
    )
    return user_count + ai_count


async def update(id: str, source: Source | None = None, **data) -> None:
    """Update a memory by ID. If source unknown, auto-detect."""
    if source is None:
        rec = await find_unique(id)
        if not rec:
            logger.warning(f"Memory {id} not found for update")
            return
        source = rec.source

    await _table(source).update(where={"id": id}, data=data)


async def update_many(
    source: Source | None = None,
    where: dict | None = None,
    data: dict | None = None,
) -> int:
    """Batch update. source=None updates both tables."""
    kwargs_where = where or {}
    kwargs_data = data or {}

    if source is not None:
        return await _table(source).update_many(where=kwargs_where, data=kwargs_data)

    user_count, ai_count = await asyncio.gather(
        db.usermemory.update_many(where=kwargs_where, data=kwargs_data),
        db.aimemory.update_many(where=kwargs_where, data=kwargs_data),
    )
    return user_count + ai_count


async def delete(id: str, source: Source | None = None) -> None:
    """Delete a memory by ID."""
    if source is None:
        rec = await find_unique(id)
        if not rec:
            return
        source = rec.source

    await _table(source).delete(where={"id": id})
