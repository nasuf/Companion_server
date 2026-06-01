"""Admin API for agent instance management — viewing agent data, memories, conversations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.memory.lifecycle.quality import (
    derive_memory_quality_from_changelog_rows,
    serialize_quality,
)
from app.services.achievements.service import list_achievements
from app.services.runtime.data_reset import hard_delete_agent_data

router = APIRouter(prefix="/admin-api/agents", tags=["admin-agents"])


# ── helpers ──

async def _resolve_workspace_id(agent_id: str) -> str | None:
    """Resolve the active workspace ID for an agent. Returns None if not found."""
    rows = await db.query_raw(
        "SELECT id FROM chat_workspaces WHERE agent_id = $1 AND status = 'active' LIMIT 1",
        agent_id,
    )
    return str(rows[0]["id"]) if rows else None


async def _resolve_workspace_context(agent_id: str) -> tuple[str, str] | None:
    """Resolve active workspace and owner user for an agent."""
    rows = await db.query_raw(
        """
        SELECT w.id AS workspace_id, w.user_id AS user_id
        FROM chat_workspaces w
        JOIN ai_agents a ON a.id = w.agent_id
        WHERE w.agent_id = $1
          AND w.status = 'active'
          AND a.user_id = w.user_id
        LIMIT 1
        """,
        agent_id,
    )
    if not rows:
        return None
    return str(rows[0]["workspace_id"]), str(rows[0]["user_id"])


def _memory_row(r: dict, source: str = "") -> dict:
    """Convert a raw SQL memory row to API response dict."""
    d = {
        "id": str(r.get("id", "")),
        "content": str(r.get("content", "")),
        "summary": str(r.get("summary", "")),
        "level": int(r.get("level", 3)),
        "importance": round(float(r.get("importance", 0)), 2),
        "main_category": str(r.get("main_category", "")),
        "sub_category": str(r.get("sub_category", "")),
        "type": str(r.get("type", "")),
        "created_at": str(r.get("created_at", "")),
    }
    if source:
        d["source"] = source
    if "mention_count" in r:
        d["mention_count"] = int(r.get("mention_count", 0))
    return d


async def _admin_quality_map(rows: list[dict], include_quality: bool) -> dict[str, dict]:
    if not include_quality or not rows:
        return {}
    row_by_id = {str(row.get("id", "")): row for row in rows if row.get("id")}
    changelog_rows = await db.query_raw(
        """
        SELECT memory_id, operation, old_value, new_value, created_at
        FROM memory_changelogs
        WHERE memory_id = ANY($1::text[])
        ORDER BY created_at ASC
        """,
        [str(row.get("id", "")) for row in rows],
    )
    grouped: dict[str, list[dict]] = {}
    for row in changelog_rows:
        grouped.setdefault(str(row["memory_id"]), []).append(row)
    return {
        memory_id: serialize_quality(
            derive_memory_quality_from_changelog_rows(
                memory_id=memory_id,
                importance=float(row_by_id.get(memory_id, {}).get("importance", 0.5)),
                source=str(row_by_id.get(memory_id, {}).get("source", "user")),
                rows=records,
            )
        )
        for memory_id, records in grouped.items()
    }


# ── endpoints ──

@router.get("")
async def list_agents(
    search: str = "",
    status: str = "",
    _: str = Depends(require_admin_jwt),
):
    """List all agents with basic info."""
    where: dict = {}
    if status:
        where["status"] = status

    agents = await db.aiagent.find_many(
        where=where,
        order={"createdAt": "desc"},
        include={"user": True},
        take=200,
    )

    result = []
    for a in agents:
        if search:
            haystack = f"{a.name} {a.user.username if a.user else ''}".lower()
            if search.lower() not in haystack:
                continue
        result.append({
            "id": a.id,
            "name": a.name,
            "status": a.status,
            "gender": a.gender,
            "age": a.age,
            "occupation": a.occupation,
            "city": a.city,
            "user_id": a.userId,
            "username": a.user.username if a.user else None,
            "created_at": str(a.createdAt),
        })
    return result


@router.get("/{agent_id}/life-story")
async def get_life_story(
    agent_id: str,
    _: str = Depends(require_admin_jwt),
):
    """Get the agent's life story (L1 AI memories scoped by workspace)."""
    workspace_id = await _resolve_workspace_id(agent_id)
    if not workspace_id:
        return []

    rows = await db.query_raw(
        """
        SELECT id, content, summary, level, importance,
               main_category, sub_category, type, created_at
        FROM memories_ai
        WHERE workspace_id = $1 AND level = 1 AND is_archived = FALSE
        ORDER BY created_at ASC
        """,
        workspace_id,
    )
    return [_memory_row(r) for r in rows]


@router.get("/{agent_id}/memory-stats")
async def get_memory_stats(
    agent_id: str,
    source: str = "",
    _: str = Depends(require_admin_jwt),
):
    """Return raw grouped counts for frontend cross-filter computation."""
    from app.api.public.memories import _compute_stats
    workspace_id = await _resolve_workspace_id(agent_id)
    return await _compute_stats(workspace_id, source or None)


@router.get("/{agent_id}/achievements")
async def get_agent_achievements(
    agent_id: str,
    _: str = Depends(require_admin_jwt),
):
    """Return achievement completion analytics for one agent."""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    data = await list_achievements(user_id=agent.userId, agent_id=agent_id)
    items = data["items"]
    category_stats: dict[str, dict] = {}
    level_stats: dict[str, dict] = {}
    recent = []
    for item in items:
        category = str(item.get("category") or "未分类")
        level = str(item.get("level_name") or "未分级")
        unlocked = bool(item.get("unlocked"))
        score = int(item.get("score") or 0)
        category_row = category_stats.setdefault(category, {"category": category, "total": 0, "unlocked": 0, "score": 0})
        level_row = level_stats.setdefault(level, {"level_name": level, "total": 0, "unlocked": 0, "score": 0})
        category_row["total"] += 1
        level_row["total"] += 1
        if unlocked:
            category_row["unlocked"] += 1
            category_row["score"] += score
            level_row["unlocked"] += 1
            level_row["score"] += score
            recent.append(item)
    recent.sort(key=lambda item: str(item.get("unlocked_at") or ""), reverse=True)
    return {
        **data,
        "agent_id": agent_id,
        "agent_name": agent.name,
        "category_stats": sorted(category_stats.values(), key=lambda row: (-row["unlocked"], row["category"])),
        "level_stats": sorted(level_stats.values(), key=lambda row: (-row["unlocked"], row["level_name"])),
        "recent_unlocks": recent[:8],
    }


@router.get("/{agent_id}/memories")
async def get_memories(
    agent_id: str,
    source: str = "",
    main_category: str = "",
    sub_category: str = "",
    level: int | None = None,
    search: str = "",
    limit: int = 50,
    offset: int = 0,
    include_quality: bool = False,
    _: str = Depends(require_admin_jwt),
):
    """Get agent memories with server-side filtering and pagination."""
    workspace_context = await _resolve_workspace_context(agent_id)
    if not workspace_context:
        return []
    workspace_id, user_id = workspace_context

    conditions = ["is_archived = FALSE", "workspace_id = $1", "user_id = $2"]
    params: list = [workspace_id, user_id]
    idx = 3

    if main_category:
        conditions.append(f"main_category = ${idx}")
        params.append(main_category)
        idx += 1
    if sub_category:
        conditions.append(f"sub_category = ${idx}")
        params.append(sub_category)
        idx += 1
    if level is not None:
        conditions.append(f"level = ${idx}")
        params.append(level)
        idx += 1
    if search:
        conditions.append(f"(content ILIKE ${idx} OR summary ILIKE ${idx})")
        params.append(f"%{search}%")
        idx += 1

    where_clause = " AND ".join(conditions)
    limit_idx = idx
    offset_idx = idx + 1

    tables: list[tuple[str, str]] = []
    if source in ("", "user"):
        tables.append(("memories_user", "user"))
    if source in ("", "ai"):
        tables.append(("memories_ai", "ai"))

    all_rows = []
    for table, src_label in tables:
        rows = await db.query_raw(
            f"""
            SELECT id, content, summary, level, importance,
                   main_category, sub_category, type, mention_count, created_at
            FROM {table}
            WHERE {where_clause}
            ORDER BY importance DESC, created_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
            """,
            *params, limit, offset,
        )
        for r in rows:
            all_rows.append(_memory_row(r, source=src_label))

    all_rows.sort(key=lambda x: -x["importance"])
    qualities = await _admin_quality_map(all_rows, include_quality)
    for row in all_rows:
        if row["id"] in qualities:
            row["quality"] = qualities[row["id"]]
    return all_rows


@router.get("/{agent_id}/conversations")
async def get_conversations(
    agent_id: str,
    _: str = Depends(require_admin_jwt),
):
    """Get all conversations for an agent with message counts (single query)."""
    rows = await db.query_raw(
        """
        SELECT c.id, c.title, c.workspace_id, c.created_at, c.updated_at,
               COUNT(m.id)::int AS message_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.agent_id = $1 AND c.is_deleted = FALSE
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        """,
        agent_id,
    )
    return [
        {
            "id": str(r["id"]),
            "title": r.get("title"),
            "workspace_id": str(r["workspace_id"]) if r.get("workspace_id") else None,
            "message_count": int(r.get("message_count", 0)),
            "created_at": str(r.get("created_at", "")),
            "updated_at": str(r.get("updated_at", "")),
        }
        for r in rows
    ]


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    _: str = Depends(require_admin_jwt),
):
    """Delete an agent and ALL related data (conversations, memories, embeddings, graph, Redis).

    Uses the same hard_delete_agent_data as the user management tab.
    Scoped by workspace — other agents' data is not affected.
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    stats = await hard_delete_agent_data(agent_id, agent.userId)
    return {"ok": True, "stats": stats}
