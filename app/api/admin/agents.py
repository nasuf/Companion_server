"""Admin API for agent instance management — viewing agent data, memories, conversations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_admin_jwt
from app.db import db
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
            "character_profile_id": getattr(a, "characterProfileId", None),
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


@router.get("/{agent_id}/memories")
async def get_memories(
    agent_id: str,
    limit: int = 500,
    _: str = Depends(require_admin_jwt),
):
    """Get all agent memories (both user + ai), scoped by workspace.

    Filtering is done client-side for consistent counts with the chat inspector.
    """
    workspace_id = await _resolve_workspace_id(agent_id)
    if not workspace_id:
        return {"items": []}

    all_rows = []
    for table, src_label in [("memories_user", "user"), ("memories_ai", "ai")]:
        rows = await db.query_raw(
            f"""
            SELECT id, content, summary, level, importance,
                   main_category, sub_category, type, mention_count, created_at
            FROM {table}
            WHERE is_archived = FALSE AND workspace_id = $1
            ORDER BY importance DESC, created_at DESC
            LIMIT $2
            """,
            workspace_id, limit,
        )
        for r in rows:
            all_rows.append(_memory_row(r, source=src_label))

    all_rows.sort(key=lambda x: -x["importance"])
    return {"items": all_rows}


@router.get("/{agent_id}/conversations")
async def get_conversations(
    agent_id: str,
    _: str = Depends(require_admin_jwt),
):
    """Get all conversations for an agent with message counts (single query)."""
    rows = await db.query_raw(
        """
        SELECT c.id, c.title, c.created_at, c.updated_at,
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
            "message_count": int(r.get("message_count", 0)),
            "created_at": str(r.get("created_at", "")),
            "updated_at": str(r.get("updated_at", "")),
        }
        for r in rows
    ]


@router.get("/{agent_id}/conversations/{conv_id}/messages")
async def get_messages(
    agent_id: str,
    conv_id: str,
    limit: int = 200,
    _: str = Depends(require_admin_jwt),
):
    """Get messages for a specific conversation."""
    messages = await db.message.find_many(
        where={"conversationId": conv_id},
        order={"createdAt": "asc"},
        take=limit,
    )
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "metadata": m.metadata if isinstance(m.metadata, dict) else None,
            "created_at": str(m.createdAt),
        }
        for m in messages
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
