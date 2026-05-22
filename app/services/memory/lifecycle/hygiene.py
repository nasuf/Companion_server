"""Scheduled memory hygiene.

This job is the offline companion to write-time reconciliation.  The write path
keeps latency low and handles high-confidence duplicate/update decisions; this
job revisits existing rows in bounded batches and applies the same fact
evolution principle inside each top-level memory category.
"""

from __future__ import annotations

import logging
import json
import uuid
from collections import defaultdict
from typing import TypedDict, cast

from app.db import db
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import log_memory_changelog
from app.services.memory.storage.reconciliation import resolve_memory_write
from app.services.memory.storage.repo import MemoryRecord, Source

logger = logging.getLogger(__name__)


class HygieneStats(TypedDict):
    scopes: int
    checked: int
    archived: int
    merged: int
    updated: int
    errors: int
    changes: list["HygieneChange"]


class HygieneMemory(TypedDict):
    id: str
    source: Source
    level: int
    main_category: str | None
    sub_category: str | None
    content: str
    summary: str | None
    importance: float


class HygieneChange(TypedDict):
    action: str
    source: Source
    main_category: str | None
    sub_category: str | None
    kept: HygieneMemory | None
    removed: HygieneMemory | None
    before: str | None
    after: str | None
    reason: str


async def _active_scopes(
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    limit: int = 50,
) -> list[tuple[Source, str, str | None]]:
    rows = await db.query_raw(
        """
        SELECT source, user_id, workspace_id
        FROM (
            SELECT 'user' AS source, user_id, workspace_id
            FROM memories_user
            WHERE is_archived = false
              AND ($1::text IS NULL OR user_id = $1)
              AND ($2::text IS NULL OR workspace_id = $2)
            GROUP BY user_id, workspace_id
            UNION ALL
            SELECT 'ai' AS source, user_id, workspace_id
            FROM memories_ai
            WHERE is_archived = false
              AND ($1::text IS NULL OR user_id = $1)
              AND ($2::text IS NULL OR workspace_id = $2)
            GROUP BY user_id, workspace_id
        ) scopes
        LIMIT $3
        """,
        user_id,
        workspace_id,
        limit,
    )
    scopes: list[tuple[Source, str, str | None]] = []
    for row in rows:
        source = row.get("source")
        uid = row.get("user_id")
        if source not in {"user", "ai"} or not uid:
            continue
        scopes.append((cast(Source, source), str(uid), row.get("workspace_id")))
    return scopes


async def _scope_memories(
    *,
    source: Source,
    user_id: str,
    workspace_id: str | None,
    limit: int,
) -> list[MemoryRecord]:
    return await memory_repo.find_many(
        source=source,
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "isArchived": False,
        },
        order={"updatedAt": "desc"},
        take=limit,
    )


def _text_of(record: MemoryRecord) -> str:
    return record.summary or record.content or ""


def _snapshot(record: MemoryRecord | None) -> HygieneMemory | None:
    if record is None:
        return None
    return {
        "id": record.id,
        "source": record.source,
        "level": record.level,
        "main_category": record.mainCategory,
        "sub_category": record.subCategory,
        "content": record.content,
        "summary": record.summary,
        "importance": float(record.importance or 0),
    }


def _group_by_main(records: list[MemoryRecord]) -> dict[str, list[MemoryRecord]]:
    grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.mainCategory or ""].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda r: (len(_text_of(r)), r.updatedAt or r.createdAt))
    return grouped


async def run_memory_hygiene(
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    allow_llm: bool = True,
    max_scopes: int = 50,
    max_memories_per_scope: int = 200,
) -> HygieneStats:
    """Run bounded duplicate cleanup and fact evolution over active memories.

    Hard boundaries: source, user, workspace.  Fact evolution only compares
    memories under the same top-level category; sub-category is intentionally
    not a partition because extraction can classify the same fact differently.
    """
    stats: HygieneStats = {
        "scopes": 0,
        "checked": 0,
        "archived": 0,
        "merged": 0,
        "updated": 0,
        "errors": 0,
        "changes": [],
    }
    scopes = await _active_scopes(
        user_id=user_id,
        workspace_id=workspace_id,
        limit=max_scopes,
    )
    stats["scopes"] = len(scopes)

    for source, scope_user_id, scope_workspace_id in scopes:
        try:
            records = await _scope_memories(
                source=source,
                user_id=scope_user_id,
                workspace_id=scope_workspace_id,
                limit=max_memories_per_scope,
            )
        except Exception as e:
            stats["errors"] += 1
            logger.warning(f"Memory hygiene failed to load scope {source}/{scope_user_id}: {e}")
            continue

        archived_ids: set[str] = set()
        for group in _group_by_main(records).values():
            for record in group:
                if record.id in archived_ids:
                    continue
                stats["checked"] += 1
                try:
                    await _hygiene_one(
                        record=record,
                        source=source,
                        user_id=scope_user_id,
                        workspace_id=scope_workspace_id,
                        archived_ids=archived_ids,
                        allow_llm=allow_llm,
                        stats=stats,
                    )
                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(f"Memory hygiene failed for memory {record.id}: {e}")

    await _best_effort_record_consolidation_run(
        stats,
        user_id=user_id,
        workspace_id=workspace_id,
    )
    return stats


async def _hygiene_one(
    *,
    record: MemoryRecord,
    source: Source,
    user_id: str,
    workspace_id: str | None,
    archived_ids: set[str],
    allow_llm: bool,
    stats: HygieneStats,
) -> None:
    text = _text_of(record)
    if not text.strip():
        return
    embedding = await generate_embedding(text)
    decision = await resolve_memory_write(
        user_id=user_id,
        source=source,
        workspace_id=workspace_id or record.workspaceId,
        content=record.content,
        summary=record.summary,
        embedding=embedding,
        main_category=record.mainCategory,
        sub_category=record.subCategory,
        exclude_id=record.id,
        allow_llm=allow_llm,
    )
    if not decision.existing_id or decision.existing_id == record.id:
        return
    if decision.existing_id in archived_ids:
        return

    if decision.action == "drop_duplicate":
        await _archive_absorbed(
            record=record,
            source=source,
            user_id=user_id,
            workspace_id=workspace_id,
            operation="hygiene_archived_duplicate",
            new_value=f"covered_by:{decision.existing_id}",
        )
        archived_ids.add(record.id)
        stats["archived"] += 1
        stats["changes"].append({
            "action": "archived_duplicate",
            "source": source,
            "main_category": record.mainCategory,
            "sub_category": record.subCategory,
            "kept": _snapshot(decision.existing_record),
            "removed": _snapshot(record),
            "before": record.content,
            "after": None,
            "reason": decision.reason or "existing_covers_new",
        })
        return

    if decision.action not in {"update_existing", "merge_existing"} or not decision.existing_record:
        return

    existing = decision.existing_record
    merged_text = decision.merged_summary or decision.merged_content or text
    merged_embedding = await generate_embedding(merged_text)
    await store_embedding(existing.id, merged_embedding)
    await memory_repo.update(
        existing.id,
        source=source,
        record=existing,
        content=merged_text,
        summary=merged_text,
        level=min(existing.level, record.level),
        importance=max(float(existing.importance or 0), float(record.importance or 0)),
        type=existing.type or record.type,
        mainCategory=existing.mainCategory or record.mainCategory,
        subCategory=existing.subCategory or record.subCategory,
    )
    await log_memory_changelog(
        user_id,
        existing.id,
        "hygiene_merge" if decision.action == "merge_existing" else "hygiene_update",
        old_value=existing.content,
        new_value=merged_text,
        workspace_id=workspace_id,
    )
    await _archive_absorbed(
        record=record,
        source=source,
        user_id=user_id,
        workspace_id=workspace_id,
        operation="hygiene_absorbed",
        new_value=f"absorbed_by:{existing.id}",
    )
    archived_ids.add(record.id)
    stats["archived"] += 1
    if decision.action == "merge_existing":
        stats["merged"] += 1
    else:
        stats["updated"] += 1
    stats["changes"].append({
        "action": "merged" if decision.action == "merge_existing" else "updated",
        "source": source,
        "main_category": existing.mainCategory or record.mainCategory,
        "sub_category": existing.subCategory or record.subCategory,
        "kept": _snapshot(existing),
        "removed": _snapshot(record),
        "before": existing.content,
        "after": merged_text,
        "reason": decision.reason or decision.action,
    })


async def _archive_absorbed(
    *,
    record: MemoryRecord,
    source: Source,
    user_id: str,
    workspace_id: str | None,
    operation: str,
    new_value: str,
) -> None:
    await memory_repo.update(
        record.id,
        source=source,
        record=record,
        isArchived=True,
    )
    await log_memory_changelog(
        user_id,
        record.id,
        operation,
        old_value=record.content,
        new_value=new_value,
        workspace_id=workspace_id,
    )


async def _best_effort_record_consolidation_run(
    stats: HygieneStats,
    *,
    user_id: str | None,
    workspace_id: str | None,
) -> None:
    try:
        rows = await db.query_raw(
            """
            INSERT INTO memory_consolidation_runs (
                id, status, user_id, workspace_id, scopes, checked,
                archived, merged, updated, errors, changes
            )
            VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10, $11::jsonb
            )
            RETURNING id
            """,
            str(uuid.uuid4()),
            "succeeded" if int(stats.get("errors") or 0) == 0 else "completed_with_errors",
            user_id,
            workspace_id,
            int(stats.get("scopes") or 0),
            int(stats.get("checked") or 0),
            int(stats.get("archived") or 0),
            int(stats.get("merged") or 0),
            int(stats.get("updated") or 0),
            int(stats.get("errors") or 0),
            json.dumps(stats.get("changes") or [], ensure_ascii=False),
        )
        run_id = rows[0].get("id") if rows else None
        if run_id:
            stats["run_id"] = run_id  # type: ignore[typeddict-unknown-key]
    except Exception as e:
        logger.debug("memory consolidation run audit skipped: %s", e)
