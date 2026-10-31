"""Template knowledge memories: append + publish to cloned agents.

Feature (2026-07): admins extend an EXISTING template's memory bank with a
knowledge document (公司/产品/活动 facts — parsed by knowledge_import.py), then
publish the new rows to agents cloned from that template — either a single
hand-picked agent (canary test) or all of them.

Design invariants:

- Knowledge rows are ordinary AI L1 memories (生活/工作, importance 0.86,
  provenance='knowledge_seed') in the template's active workspace, so a NEW
  clone picks them up automatically through the normal clone copy — sync is
  only needed for agents cloned BEFORE the append.
- Sync is a verbatim row + pgvector copy (no LLM, no re-embedding), mirroring
  clone._clone_ai_memories.
- Idempotency is exact-content comparison scoped to provenance =
  'knowledge_seed' in the target workspace. Re-running a sync, or syncing an
  agent cloned after the append, is a no-op. The comparison deliberately
  includes archived rows so knowledge an admin archived inside one agent is
  never resurrected. (A PHYSICAL delete — e.g. the user asked the AI to
  forget a fact via the chat deletion flow — is re-published by the next
  sync: published knowledge is admin-governed by design.)
- The template-level "pending" badge derives from a watermark
  (``ai_agents.knowledge_synced_at``, accessed via raw SQL for pre-migration
  tolerance): it only advances when a FULL sync — all related agents — ends
  with zero failures. Canary syncs never advance it.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import db
from app.redis_client import get_redis
from app.services.agent_template.clone import _MEMORY_COPY_FIELDS
from app.services.agent_template.knowledge_import import KnowledgeItem, validate_knowledge_snapshot
from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME, count_active_clones
from app.services.memory.evidence import EvidenceSource, bind_memory_evidence, content_version
from app.services.memory.provenance import KNOWLEDGE_SEED
from app.services.memory.retrieval.knowledge_hits import invalidate_knowledge_rows
from app.services.memory.input_evidence import InputOrigin, persist_input_origin
from app.services.memory.storage import repo as memory_repo
from app.services.memory.storage.embedding import generate_embedding, store_embedding
from app.services.memory.storage.persistence import after_memory_changelog
from app.services.time_domain.parser import resolve_occur_time
from app.services.runtime.distributed_lock import (
    DistributedLockNotAcquired,
    distributed_lock,
)
from app.services.runtime.tasks import fire_background
from app.services.workspace.workspaces import get_active_workspace

logger = logging.getLogger(__name__)

# Knowledge facts are work/world context for the persona (the current use case
# is 公司/产品/活动 facts for a company-employee persona). 生活/工作 is an
# AI-L1-legal, non-singleton sub-category, so any number of rows may coexist
# without touching the provisioning coverage/singleton rules.
KNOWLEDGE_MAIN_CATEGORY = "生活"
KNOWLEDGE_SUB_CATEGORY = "工作"
# L1 (≥0.85) so the rows never decay via L2 dynamics, but below the 0.95 of
# core persona facts so they never outrank identity in importance-ordered use.
KNOWLEDGE_IMPORTANCE = 0.86

_PROGRESS_TTL_S = 24 * 3600
_SYNC_LOCK_TTL_S = 1800
# A "running" progress older than this is treated as crashed (lock TTL has
# expired by then too) and no longer blocks a new sync.
_STALE_RUNNING_S = 1800
_MAX_FAILURE_SAMPLES = 20
MAX_KNOWLEDGE_SYNC_MEMORIES = 1000
KNOWLEDGE_SYNC_EXTRACTOR = "knowledge-sync-v1"


class KnowledgeSyncBusy(RuntimeError):
    """A sync for this template is already running."""


# ── Append ─────────────────────────────────────────────────────────────


async def append_knowledge_to_template(
    *, template_agent_id: str, template_user_id: str,
    items: list[KnowledgeItem], input_origin: InputOrigin,
) -> dict[str, Any]:
    """Append one bounded document atomically, including its original input.

    Embedding calls run outside the transaction. A dedicated PostgreSQL advisory lock
    serializes exact-content dedup even without Redis. The persona lock and
    scope recheck prevent an import prepared for an old workspace from landing
    in a newly generated persona. Archived knowledge still blocks duplicates.
    """
    validate_knowledge_snapshot(items, input_origin)
    workspace = await get_active_workspace(agent_id=template_agent_id)
    if workspace is None:
        raise ValueError("模板没有可用的活跃 workspace")
    # Bounded by the uploaded batch, not by the template's lifetime row count.
    summaries = list(dict.fromkeys(item.summary for item in items))
    existing_rows = await db.query_raw(
        """SELECT DISTINCT content FROM memories_ai WHERE user_id=$1 AND workspace_id=$2
           AND provenance='knowledge_seed' AND content=ANY($3::text[])""",
        template_user_id, workspace.id, summaries,
    )
    existing = {row["content"] for row in existing_rows}
    embeddings = {}
    for item in items:
        if not item.is_oversized and item.summary not in existing and item.summary not in embeddings:
            embeddings[item.summary] = await generate_embedding(item.summary)

    new_ids: list[str] = []
    skipped = 0
    oversized = []
    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        # Only appenders contend on this lock. Avoid workspace UPDATE here:
        # cloning holds workspace SHARE before acquiring its persona lock.
        await tx.query_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1,0))::text AS held",
            "knowledge-append:" + workspace.id,
        )
        await tx.query_raw(
            "SELECT pg_advisory_xact_lock_shared(hashtextextended($1,0))::text AS held",
            "persona-init:" + workspace.id,
        )
        scope = await tx.query_raw(
            """SELECT w.id FROM chat_workspaces w
               JOIN ai_agents a ON a.id=w.agent_id AND a.user_id=w.user_id
               JOIN users u ON u.id=a.user_id
               WHERE w.id=$1 AND w.agent_id=$2 AND w.user_id=$3
                 AND u.username=$4 AND u.status='active' AND u.archived_at IS NULL
                 AND a.status='active' AND a.archived_at IS NULL
                 AND w.status='active' AND w.archived_at IS NULL
               FOR SHARE OF w,a,u""",
            workspace.id, template_agent_id, template_user_id, TEMPLATE_SYSTEM_USERNAME,
        )
        if len(scope) != 1:
            raise ValueError("knowledge_template_scope_changed")
        rows = await tx.query_raw(
            """SELECT DISTINCT content FROM memories_ai WHERE user_id=$1 AND workspace_id=$2
               AND provenance='knowledge_seed' AND content=ANY($3::text[])""", template_user_id, workspace.id, summaries,
        )
        existing = {row["content"] for row in rows}
        source = None
        for item in items:
            if item.summary in existing:
                skipped += 1
                continue
            if item.is_oversized:
                oversized.append(item.summary[:120])
                continue
            if item.summary not in embeddings:
                # It existed before the network call and was then deleted or
                # edited. A retry must prepare its vector; never insert half a row.
                raise ValueError("knowledge_existing_content_changed_retry")
            if source is None:
                source = await persist_input_origin(
                    tx, user_id=template_user_id, workspace_id=workspace.id,
                    agent_id=template_agent_id, origin=input_origin,
                )
            now = datetime.now(UTC)
            occur_time = resolve_occur_time(
                item.summary, statement_time=None, main_category=KNOWLEDGE_MAIN_CATEGORY,
                sub_category=KNOWLEDGE_SUB_CATEGORY, provenance=KNOWLEDGE_SEED,
            )
            data = dict(userId=template_user_id, workspaceId=workspace.id,
                        content=item.summary, type="life", level=1,
                        importance=KNOWLEDGE_IMPORTANCE, mainCategory=KNOWLEDGE_MAIN_CATEGORY,
                        subCategory=KNOWLEDGE_SUB_CATEGORY, provenance=KNOWLEDGE_SEED,
                        valueUpdatedAt=now, statementTime=now)
            if occur_time is not None:
                data["occurTime"] = occur_time
            memory = await memory_repo.create(
                "ai", database=tx, evidence_sources=(source,),
                extractor_version="knowledge-document-v1", **data,
            )
            await store_embedding(memory.id, embeddings[item.summary], database=tx)
            await tx.memorychangelog.create(data={
                "userId": template_user_id, "memoryId": memory.id, "operation": "insert",
                "newValue": item.summary, "workspaceId": workspace.id,
            })
            existing.add(item.summary)
            new_ids.append(memory.id)
    if new_ids:
        await memory_repo.invalidate_scope(template_user_id, workspace.id)
        await _bust_knowledge_rows_cache(workspace.id)
        for memory_id in new_ids:
            await after_memory_changelog(user_id=template_user_id, memory_id=memory_id,
                                         operation="insert", workspace_id=workspace.id)
    logger.info("[TEMPLATE-KNOWLEDGE] appended %d rows (%d dup, %d oversized) to template %s",
                len(new_ids), skipped, len(oversized), template_agent_id[:8])
    return {"parsed": len(items), "stored": len(new_ids),
            "skipped_duplicates": skipped, "skipped_oversized": oversized}


async def _bust_knowledge_rows_cache(workspace_id: str) -> None:
    """Keep existing callers on the shared post-commit invalidation path."""
    await invalidate_knowledge_rows(workspace_id)


# ── Status / overview ──────────────────────────────────────────────────


async def get_knowledge_status(template_agent_id: str) -> dict[str, Any]:
    """Badge-level status: row count, last append time, watermark, pending.

    ``pending_update`` is a watermark proxy, not an exact per-agent diff (the
    exact diff lives in get_related_agents_with_pending, one click deeper):
    it turns on when knowledge rows are newer than the last successful FULL
    sync AND at least one cloned agent exists. A false positive (e.g. every
    current clone was created after the append and already carries the rows)
    is cleared by running a full sync, which no-ops and advances the
    watermark.
    """
    empty = {
        "knowledge_count": 0,
        "last_appended_at": None,
        "knowledge_synced_at": None,
        "related_agent_count": 0,
        "pending_update": False,
    }
    workspace = await get_active_workspace(agent_id=template_agent_id)
    if workspace is None:
        return empty

    n = 0
    latest: str | None = None
    unsynced: int | None = None
    synced_at: str | None = None
    try:
        # One aggregate query: row count, newest append, watermark value and
        # how many rows are newer than it. The LEFT JOIN binds the single
        # template agent row to every memory row (constant join).
        rows = await db.query_raw(
            """
            SELECT count(*)::int AS n,
                   max(m.created_at)::text AS latest,
                   max(a.knowledge_synced_at)::text AS synced_at,
                   count(*) FILTER (
                       WHERE a.knowledge_synced_at IS NULL
                          OR m.created_at > a.knowledge_synced_at
                   )::int AS unsynced
            FROM memories_ai m
            LEFT JOIN ai_agents a ON a.id = $2
            WHERE m.workspace_id = $1
              AND m.provenance = 'knowledge_seed'
              AND m.is_archived = FALSE
            """,
            workspace.id,
            template_agent_id,
        )
        if rows:
            n = int(rows[0].get("n") or 0)
            latest = rows[0].get("latest")
            synced_at = rows[0].get("synced_at")
            unsynced = rows[0].get("unsynced")
    except Exception as exc:
        # Pre-migration tolerance: knowledge_synced_at may not exist yet —
        # fall back to counting only and treat everything as unsynced.
        logger.warning("[TEMPLATE-KNOWLEDGE] status query fallback: %s", exc)
        rows = await db.query_raw(
            """
            SELECT count(*)::int AS n, max(created_at)::text AS latest
            FROM memories_ai
            WHERE workspace_id = $1
              AND provenance = 'knowledge_seed'
              AND is_archived = FALSE
            """,
            workspace.id,
        )
        if rows:
            n = int(rows[0].get("n") or 0)
            latest = rows[0].get("latest")
        unsynced = n

    pending = int(unsynced if unsynced is not None else n)
    related = await count_active_clones(template_agent_id)
    return {
        "knowledge_count": n,
        "last_appended_at": latest,
        "knowledge_synced_at": synced_at,
        "related_agent_count": related,
        # No clones → nothing can be out of sync (new clones copy knowledge
        # rows at clone time), so the badge must stay off.
        "pending_update": n > 0 and pending > 0 and related > 0,
    }


async def list_knowledge_items(template_agent_id: str, take: int = 200) -> list[dict[str, Any]]:
    """Non-archived knowledge rows of the template, newest first."""
    workspace = await get_active_workspace(agent_id=template_agent_id)
    if workspace is None:
        return []
    rows = await db.aimemory.find_many(
        where={
            "workspaceId": workspace.id,
            "provenance": KNOWLEDGE_SEED,
            "isArchived": False,
        },
        order={"createdAt": "desc"},
        take=max(1, min(int(take), 200)),
    )
    return [
        {"id": row.id, "summary": row.content, "created_at": str(row.createdAt),
         "evidence_scope": {"user_id": row.userId, "workspace_id": workspace.id, "source": "ai"}}
        for row in rows
    ]


async def get_related_agents_with_pending(
    template_agent_id: str, limit: int = 500
) -> list[dict[str, Any]]:
    """Active agents cloned from this template + how many knowledge rows each
    is still missing (exact-content diff, one aggregate query for all agents).
    """
    # limit is a server-side constant (never user input) — inlined because
    # numeric LIMIT parameter binding is not worth the driver-compat risk.
    agents = await db.query_raw(
        f"""
        SELECT a.id, a.user_id, u.username, a.created_at::text AS created_at
        FROM ai_agents a
        JOIN users u ON u.id = a.user_id
        WHERE a.source_template_id = $1
          AND a.status = 'active'
          AND a.archived_at IS NULL
        ORDER BY a.created_at DESC
        LIMIT {int(limit)}
        """,
        template_agent_id,
    )
    missing_by_agent: dict[str, int] = {}
    workspace = await get_active_workspace(agent_id=template_agent_id)
    if workspace is not None and agents:
        rows = await db.query_raw(
            """
            SELECT a.id AS agent_id, count(DISTINCT t.id)::int AS missing
            FROM ai_agents a
            JOIN chat_workspaces w ON w.agent_id = a.id AND w.status = 'active'
            CROSS JOIN (
                SELECT id, content
                FROM memories_ai
                WHERE workspace_id = $2
                  AND provenance = 'knowledge_seed'
                  AND is_archived = FALSE
            ) t
            WHERE a.source_template_id = $1
              AND a.status = 'active'
              AND a.archived_at IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM memories_ai m
                  WHERE m.workspace_id = w.id
                    AND m.provenance = 'knowledge_seed'
                    AND m.content = t.content
              )
            GROUP BY a.id
            """,
            template_agent_id,
            workspace.id,
        )
        missing_by_agent = {r["agent_id"]: int(r["missing"] or 0) for r in rows}
    return [
        {
            "agent_id": a["id"],
            "user_id": a["user_id"],
            "username": a.get("username"),
            "created_at": a["created_at"],
            "pending_count": missing_by_agent.get(a["id"], 0),
        }
        for a in agents
    ]


# ── Sync (publish to clones) ───────────────────────────────────────────


async def start_knowledge_sync(
    *, template_agent_id: str, agent_ids: list[str] | None
) -> dict[str, Any]:
    """Validate and launch a background sync job.

    ``agent_ids=None`` publishes to ALL related agents (advances the badge
    watermark on full success); a non-empty list is a canary sync that never
    advances the watermark.
    """
    progress = await get_sync_progress(template_agent_id)
    if progress and progress.get("status") == "running":
        started_ts = float(progress.get("started_at_ts") or 0)
        if time.time() - started_ts < _STALE_RUNNING_S:
            raise KnowledgeSyncBusy("该模板的记忆同步正在进行中，请稍后再试")

    workspace = await get_active_workspace(agent_id=template_agent_id)
    if workspace is None:
        raise ValueError("模板没有可用的活跃 workspace")
    template_rows = await db.aimemory.find_many(
        where={
            "workspaceId": workspace.id,
            "provenance": KNOWLEDGE_SEED,
            "isArchived": False,
        },
        order={"createdAt": "asc"},
        take=MAX_KNOWLEDGE_SYNC_MEMORIES + 1,
    )
    if not template_rows:
        raise ValueError("模板还没有知识记忆，请先上传知识文档")
    if len(template_rows) > MAX_KNOWLEDGE_SYNC_MEMORIES:
        raise ValueError("template_knowledge_over_limit")

    full_mode = not agent_ids
    targets = await _resolve_target_agents(template_agent_id, agent_ids)
    if not targets:
        raise ValueError("没有可同步的关联 agent")

    # Write the "running" placeholder BEFORE firing the job so a UI poll
    # landing between this response and the job's first own write can never
    # observe a stale "done" record from a previous run.
    await _write_progress(
        template_agent_id,
        {
            "status": "running",
            "mode": "all" if full_mode else "selected",
            "total_agents": len(targets),
            "processed_agents": 0,
            "copied_memories": 0,
            "started_at_ts": time.time(),
            "started_at": _now_iso(),
        },
    )
    coro = _run_sync(
        template_agent_id=template_agent_id,
        template_rows=template_rows,
        targets=targets,
        full_mode=full_mode,
    )
    try:
        fire_background(coro)
    except Exception as exc:
        coro.close()
        # Do not leave the placeholder blocking retries for the stale window.
        await _write_progress(
            template_agent_id,
            {"status": "error", "error": str(exc)[:200], "finished_at": _now_iso()},
        )
        raise
    return {
        "started": True,
        "mode": "all" if full_mode else "selected",
        "total_agents": len(targets),
    }


async def _resolve_target_agents(
    template_agent_id: str, agent_ids: list[str] | None
) -> list[dict[str, Any]]:
    """Freeze each live target's owner, workspace and execution generations.

    sourceTemplateId is read via raw SQL everywhere (consistent with
    registry.count_active_clones) so this works even when the generated Prisma
    client predates the column.
    """
    if agent_ids:
        results: list[dict[str, Any]] = []
        for agent_id in dict.fromkeys(agent_ids):
            rows = await db.query_raw(
                """
                SELECT a.id,a.user_id,w.id AS workspace_id,
                       a.execution_generation AS agent_generation,
                       u.execution_generation AS user_generation,
                       w.execution_generation AS workspace_generation
                FROM ai_agents a JOIN users u ON u.id=a.user_id
                JOIN chat_workspaces w ON w.agent_id=a.id AND w.user_id=a.user_id
                WHERE a.id=$1 AND a.source_template_id=$2
                  AND a.status='active' AND a.archived_at IS NULL
                  AND u.status='active' AND u.archived_at IS NULL
                  AND w.status='active' AND w.archived_at IS NULL
                """,
                agent_id,
                template_agent_id,
            )
            if not rows:
                raise ValueError(f"agent {agent_id} 不是该模板的有效关联 agent")
            results.append(rows[0])
        return results
    return await db.query_raw(
        """
        SELECT a.id,a.user_id,w.id AS workspace_id,
               a.execution_generation AS agent_generation,
               u.execution_generation AS user_generation,
               w.execution_generation AS workspace_generation
        FROM ai_agents a JOIN users u ON u.id=a.user_id
        JOIN chat_workspaces w ON w.agent_id=a.id AND w.user_id=a.user_id
        WHERE a.source_template_id=$1 AND a.status='active' AND a.archived_at IS NULL
          AND u.status='active' AND u.archived_at IS NULL
          AND w.status='active' AND w.archived_at IS NULL
        ORDER BY a.created_at ASC
        """,
        template_agent_id,
    )


async def _run_sync(
    *,
    template_agent_id: str,
    template_rows: list[Any],
    targets: list[dict[str, Any]],
    full_mode: bool,
) -> None:
    """Background job wrapper: lock, run, and record terminal state."""
    try:
        async with distributed_lock(
            f"template_knowledge_sync:{template_agent_id}",
            ttl_s=_SYNC_LOCK_TTL_S,
            wait_timeout_s=0.0,
            fail_open=False,
        ):
            await _run_sync_locked(
                template_agent_id=template_agent_id,
                template_rows=template_rows,
                targets=targets,
                full_mode=full_mode,
            )
    except DistributedLockNotAcquired:
        # Another worker won the race; it owns the progress record — do not
        # touch it from the losing side.
        logger.warning(
            "[TEMPLATE-KSYNC] sync already running for template %s; skipped",
            template_agent_id[:8],
        )
    except Exception as exc:
        logger.exception(
            "[TEMPLATE-KSYNC] sync crashed for template %s", template_agent_id[:8]
        )
        await _write_progress(
            template_agent_id,
            {
                "status": "error",
                "error": str(exc)[:200],
                "finished_at": _now_iso(),
            },
        )


async def _run_sync_locked(
    *,
    template_agent_id: str,
    template_rows: list[Any],
    targets: list[dict[str, Any]],
    full_mode: bool,
) -> None:
    progress: dict[str, Any] = {
        "status": "running",
        "mode": "all" if full_mode else "selected",
        "total_agents": len(targets),
        "processed_agents": 0,
        "synced_agents": 0,
        "skipped_agents": 0,
        "failed_agents": 0,
        "copied_memories": 0,
        "failures": [],
        "started_at_ts": time.time(),
        "started_at": _now_iso(),
        "finished_at": None,
        "error": None,
        "watermark_advanced": False,
    }
    await _write_progress(template_agent_id, progress)

    watermark = max(
        (row.createdAt for row in template_rows if getattr(row, "createdAt", None)),
        default=None,
    )
    for target in targets:
        agent_id = target["id"]
        user_id = target["user_id"]
        try:
            copied = await _sync_agent(
                template_rows=template_rows, template_agent_id=template_agent_id,
                agent_id=agent_id, user_id=user_id, target_scope=target,
            )
            if copied > 0:
                progress["synced_agents"] += 1
                progress["copied_memories"] += copied
            else:
                progress["skipped_agents"] += 1
        except Exception as exc:
            logger.warning(
                "[TEMPLATE-KSYNC] agent %s sync failed: %s", agent_id[:8], exc
            )
            progress["failed_agents"] += 1
            if len(progress["failures"]) < _MAX_FAILURE_SAMPLES:
                progress["failures"].append(
                    {"agent_id": agent_id, "reason": str(exc)[:120]}
                )
        progress["processed_agents"] += 1
        await _write_progress(template_agent_id, progress)

    progress["status"] = "done"
    if full_mode and progress["failed_agents"] == 0 and watermark is not None:
        try:
            await _advance_watermark(template_agent_id, watermark)
            progress["watermark_advanced"] = True
        except Exception:
            # Copies have committed independently. Keep their counts while
            # exposing a failed finalization, so the UI never claims that a
            # failed watermark write cleared the pending badge.
            logger.exception("[TEMPLATE-KSYNC] watermark finalization failed")
            progress["status"] = "error"
            progress["error"] = "watermark_write_failed"

    progress["finished_at"] = _now_iso()
    await _write_progress(template_agent_id, progress)
    logger.info(
        "[TEMPLATE-KSYNC] template %s: %d synced / %d skipped / %d failed "
        "(+%d rows, mode=%s)",
        template_agent_id[:8],
        progress["synced_agents"],
        progress["skipped_agents"],
        progress["failed_agents"],
        progress["copied_memories"],
        progress["mode"],
    )


def _build_copy_rows(
    template_rows: list[Any],
    *,
    existing_contents: set[str],
    user_id: str,
    workspace_id: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Pure diff: template rows missing from the target workspace → create
    payloads + ``(template_memory_id, new_memory_id)`` pairs for the pgvector
    copy. Mirrors clone._clone_ai_memories field-for-field."""
    new_rows: list[dict[str, Any]] = []
    id_pairs: list[tuple[str, str]] = []
    seen = set(existing_contents)
    for row in template_rows:
        content = getattr(row, "content", None)
        if not content or content in seen:
            continue
        seen.add(content)
        new_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "id": new_id,
            "userId": user_id,
            "workspaceId": workspace_id,
            "valueUpdatedAt": datetime.now(UTC),
        }
        for field in _MEMORY_COPY_FIELDS:
            payload[field] = getattr(row, field, None)
        new_rows.append(payload)
        id_pairs.append((row.id, new_id))
    return new_rows, id_pairs


async def _sync_agent(
    *, template_rows: list[Any], template_agent_id: str, agent_id: str, user_id: str,
    target_scope: dict[str, Any],
) -> int:
    """Publish one bounded snapshot atomically, rechecking both live scopes.

    The caller's snapshot only supplies IDs and expected text versions. Copy
    fields come from locked source rows; model/caller supplied fields cannot
    acquire template-copy authority. A failed vector, audit or source write
    rolls back the entire target batch, so a retry cannot skip partial rows.
    """
    if not 1 <= len(template_rows) <= MAX_KNOWLEDGE_SYNC_MEMORIES:
        raise ValueError("invalid_knowledge_sync_batch")
    expected = {row.id: content_version(row.content) for row in template_rows}
    if len(expected) != len(template_rows):
        raise ValueError("duplicate_knowledge_source_ids")

    async with db.tx(timeout=timedelta(seconds=30)) as tx:
        template_query = """SELECT w.id,w.user_id FROM ai_agents a
               JOIN users u ON u.id=a.user_id
               JOIN chat_workspaces w ON w.agent_id=a.id AND w.user_id=a.user_id
               WHERE a.id=$1 AND u.username=$2 AND u.status='active'
                 AND a.status='active' AND a.archived_at IS NULL AND w.status='active'
                 AND w.archived_at IS NULL AND u.archived_at IS NULL
               """
        templates = await tx.query_raw(template_query, template_agent_id, TEMPLATE_SYSTEM_USERNAME)
        if len(templates) != 1:
            raise ValueError("knowledge_template_scope_invalid")
        template = templates[0]
        # Coordinate with force persona replacement without blocking other
        # publishers' reads of the same template.
        await tx.query_raw(
            "SELECT pg_advisory_xact_lock_shared(hashtextextended($1,0))::text AS held",
            "persona-init:" + template["id"],
        )
        locked = await tx.query_raw(
            template_query + " FOR SHARE OF a,u,w", template_agent_id, TEMPLATE_SYSTEM_USERNAME,
        )
        if locked != templates:
            raise ValueError("knowledge_template_scope_changed")
        targets = await tx.query_raw(
            """SELECT w.id AS workspace_id,a.execution_generation AS agent_generation,
                      u.execution_generation AS user_generation,w.execution_generation AS workspace_generation
               FROM ai_agents a
               JOIN users u ON u.id=a.user_id
               JOIN chat_workspaces w ON w.agent_id=a.id AND w.user_id=a.user_id
               WHERE a.id=$1 AND a.user_id=$2 AND a.source_template_id=$3
                 AND a.status='active' AND a.archived_at IS NULL AND w.status='active'
                 AND u.status='active' AND u.username<>$4
                 AND u.archived_at IS NULL AND w.archived_at IS NULL
               FOR UPDATE OF w FOR SHARE OF a,u""",
            agent_id, user_id, template_agent_id, TEMPLATE_SYSTEM_USERNAME,
        )
        if len(targets) != 1:
            raise ValueError("knowledge_target_scope_invalid")
        if any(target_scope.get(key) != value for key,value in targets[0].items()):
            raise ValueError("knowledge_target_scope_expired")
        workspace_id = targets[0]["workspace_id"]
        snapshot = await tx.query_raw(
            """SELECT id FROM memories_ai WHERE id=ANY($1::text[]) AND workspace_id=$2
               AND user_id=$3 AND provenance='knowledge_seed' AND NOT is_archived
               ORDER BY id LIMIT $4 FOR SHARE""",
            list(expected), template["id"], template["user_id"], MAX_KNOWLEDGE_SYNC_MEMORIES + 1,
        )
        if len(snapshot) != len(expected):
            raise ValueError("knowledge_source_scope_or_state_changed")
        rows = await tx.aimemory.find_many(
            where={"id": {"in": [r["id"] for r in snapshot]}}, order={"id": "asc"},
        )
        if any(content_version(row.content) != expected[row.id] for row in rows):
            raise ValueError("knowledge_source_version_changed")
        existing = await tx.query_raw(
            """SELECT content FROM memories_ai WHERE user_id=$1 AND workspace_id=$2
               AND provenance='knowledge_seed' AND content=ANY($3::text[])""",
            user_id, workspace_id, [row.content for row in rows],
        )
        new_rows, id_pairs = _build_copy_rows(
            rows, existing_contents={r["content"] for r in existing},
            user_id=user_id, workspace_id=workspace_id,
        )
        if not new_rows:
            return 0
        await tx.aimemory.create_many(data=new_rows)
        values = ",".join(f"(${i*2+1},${i*2+2})" for i in range(len(id_pairs)))
        await tx.execute_raw(
            f"""INSERT INTO memory_embeddings (memory_id,embedding)
                SELECT v.new_id::text,e.embedding FROM (VALUES {values}) v(new_id,template_id)
                JOIN memory_embeddings e ON e.memory_id=v.template_id::text
                ON CONFLICT (memory_id) DO NOTHING""",
            *[v for source_id,new_id in id_pairs for v in (new_id,source_id)],
        )
        await tx.memorychangelog.create_many(data=[{
            "userId": user_id, "memoryId": row["id"], "operation": "knowledge_sync",
            "newValue": row["content"], "workspaceId": workspace_id,
        } for row in new_rows])
        for source_id, new_id in id_pairs:
            await bind_memory_evidence(
                memory_id=new_id, side="ai", user_id=user_id, workspace_id=workspace_id,
                sources=(EvidenceSource("memory", source_id, side="ai", relation="template_copy",
                                        expected_version=expected[source_id]),),
                extractor_version=KNOWLEDGE_SYNC_EXTRACTOR, database=tx,
            )
    await _bust_knowledge_rows_cache(workspace_id)
    return len(new_rows)


async def _advance_watermark(template_agent_id: str, watermark: datetime) -> None:
    updated = await db.execute_raw(
        """UPDATE ai_agents SET knowledge_synced_at =
           GREATEST(knowledge_synced_at, $2::timestamptz) WHERE id = $1""",
        template_agent_id,
        watermark.isoformat(),
    )
    if updated != 1:
        raise ValueError("knowledge_watermark_target_unavailable")


# ── Progress (Redis) ───────────────────────────────────────────────────


def _progress_key(template_agent_id: str) -> str:
    return f"template_knowledge_sync:{template_agent_id}"


async def _write_progress(template_agent_id: str, progress: dict[str, Any]) -> None:
    try:
        redis = await get_redis()
        await redis.set(
            _progress_key(template_agent_id),
            json.dumps(progress, ensure_ascii=False),
            ex=_PROGRESS_TTL_S,
        )
    except Exception as exc:
        logger.warning("[TEMPLATE-KSYNC] progress write failed: %s", exc)


async def get_sync_progress(template_agent_id: str) -> dict[str, Any] | None:
    try:
        redis = await get_redis()
        raw = await redis.get(_progress_key(template_agent_id))
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
