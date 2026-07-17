"""Clone a pre-provisioned template agent into a per-user agent.

Why this exists
---------------
The normal agent-creation pipeline (``POST /agents``) runs an expensive LLM +
embedding flow (~90s) to synthesize the AI persona and its L1 self-memory. That
is far too slow for a "WeChat Mini Program user logs in and starts chatting
immediately" experience.

Instead we provision ONE fully-built *template* agent once (via the normal flow,
owned by the template system user), then cheaply **clone** it per user:

* copy the persona row (name / MBTI / background / values ...) and assign one
  gender-matched avatar for the new user,
* copy its L1 ``memories_ai`` rows into the new user's workspace,
* copy the corresponding pgvector embeddings (pure SQL copy — no Ollama call),
* create the user's workspace + first conversation.

Isolation guarantee
--------------------
Every clone is an independent ``AiAgent`` owned by the user. All downstream
dynamic state is keyed by (agent, user) / workspace / conversation:

* conversations / messages      -> per conversation (userId)
* memories_user / memories_ai   -> per workspace (+ userId)
* intimacy / patience           -> unique(agentId, userId)
* proactive_state               -> unique(workspaceId)
* ai_daily_schedule             -> per cloned agent (each clone has its own)

So chat history, memory, mood, intimacy and boundary are fully independent
between users, while the *starting persona* is identical (cloned from template).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from prisma import Json

from app.db import db
from app.services.agent_avatars import pick_agent_avatar
from app.services.agent_template.registry import get_default_template_agent_id
from app.services.workspace.workspaces import (
    create_workspace,
    finalize_archived_workspaces,
    get_active_workspace,
    restore_staged_workspaces,
    stage_active_workspaces_for_user,
)

logger = logging.getLogger(__name__)

# Scalar columns copied verbatim from a template AiMemory row into the clone.
_MEMORY_COPY_FIELDS = (
    "type",
    "mainCategory",
    "subCategory",
    "level",
    "content",
    "summary",
    "importance",
    "mentionCount",
    "isArchived",
    "occurTime",
    "statementTime",
    "recurrence",
)


def _clone_persona_data(template, user_id: str) -> dict[str, Any]:
    """Build the create() payload for a persona-identical clone."""
    avatar = pick_agent_avatar(template.gender)
    data: dict[str, Any] = {
        "name": template.name,
        "user": {"connect": {"id": user_id}},
        "status": "active",
        "archivedAt": None,
        "background": template.background,
        "lifeOverview": template.lifeOverview,
        "age": template.age,
        "occupation": template.occupation,
        "city": template.city,
        "gender": template.gender,
        "avatarKey": avatar.key,
        "avatarUrl": avatar.url,
    }
    if template.mbti is not None:
        data["mbti"] = Json(template.mbti)
    if template.currentMbti is not None:
        data["currentMbti"] = Json(template.currentMbti)
    if template.values is not None:
        data["values"] = Json(template.values)
    return data


async def _clone_ai_memories(
    *, template_workspace_id: str, new_workspace_id: str, user_id: str
) -> int:
    """Copy the template's L1 self-memory rows + their embeddings.

    Uses client-generated UUIDs so the vector rows can be copied in the same
    pass without a round-trip to read back generated ids.
    """
    template_rows = await db.aimemory.find_many(
        where={"workspaceId": template_workspace_id, "isArchived": False}
    )
    if not template_rows:
        return 0

    new_rows: list[dict[str, Any]] = []
    id_pairs: list[tuple[str, str]] = []  # (template_id, new_id)
    for row in template_rows:
        new_id = str(uuid.uuid4())
        id_pairs.append((row.id, new_id))
        payload: dict[str, Any] = {
            "id": new_id,
            "userId": user_id,
            "workspaceId": new_workspace_id,
        }
        for field in _MEMORY_COPY_FIELDS:
            payload[field] = getattr(row, field, None)
        new_rows.append(payload)

    await db.aimemory.create_many(data=new_rows)

    # Copy embeddings row-by-row (dozens of L1 rows; no LLM involved).
    copied = 0
    for template_id, new_id in id_pairs:
        try:
            await db.execute_raw(
                """
                INSERT INTO memory_embeddings (memory_id, embedding)
                SELECT $1, embedding FROM memory_embeddings WHERE memory_id = $2
                ON CONFLICT (memory_id) DO NOTHING
                """,
                new_id,
                template_id,
            )
            copied += 1
        except Exception as exc:
            # A missing embedding must not fail the whole clone; the row still
            # exists and can be re-embedded lazily later.
            logger.warning(
                "[AGENT-CLONE] embedding copy failed for memory %s: %s",
                new_id[:8],
                exc,
            )
    logger.info(
        "[AGENT-CLONE] copied %d memories (%d embeddings) into workspace %s",
        len(new_rows),
        copied,
        new_workspace_id[:8],
    )
    return len(new_rows)


async def clone_template_agent_for_user(user_id: str, template_agent_id: str):
    """Clone the template agent into a new per-user agent + workspace + conversation.

    Returns ``(agent, workspace, conversation)``. Raises ``ValueError`` if the
    template is missing/archived or has no active workspace to copy memory from.
    """
    template = await db.aiagent.find_unique(where={"id": template_agent_id})
    # Only clone a fully-provisioned template; a "provisioning" one may still have
    # incomplete / empty L1 memory, which would produce a broken clone.
    if not template or getattr(template, "status", "") != "active":
        raise ValueError("template agent not available")

    template_ws = await get_active_workspace(agent_id=template_agent_id)
    if not template_ws:
        raise ValueError("template agent has no active workspace")

    agent = await db.aiagent.create(data=_clone_persona_data(template, user_id))
    # Record provenance via raw SQL so this works even before a Prisma client
    # regen picks up the new column. Failure here must not break the clone.
    try:
        await db.execute_raw(
            "UPDATE ai_agents SET source_template_id = $1 WHERE id = $2",
            template_agent_id,
            agent.id,
        )
    except Exception as exc:
        logger.warning("[AGENT-CLONE] source_template_id write failed: %s", exc)

    staged: list[dict[str, Any]] = []
    workspace = None
    try:
        # Make the clone the user's single active workspace (mirrors create_agent).
        staged = await stage_active_workspaces_for_user(user_id)
        workspace = await create_workspace(user_id, agent.id)
        await finalize_archived_workspaces(staged)

        await _clone_ai_memories(
            template_workspace_id=template_ws.id,
            new_workspace_id=workspace.id,
            user_id=user_id,
        )

        conversation = await db.conversation.create(
            data={
                "user": {"connect": {"id": user_id}},
                "agent": {"connect": {"id": agent.id}},
                "workspace": {"connect": {"id": workspace.id}},
            }
        )
    except Exception:
        # Roll back best-effort so a failed clone never leaves a half-active user.
        if workspace is not None:
            try:
                await db.chatworkspace.update(
                    where={"id": workspace.id},
                    data={"status": "archived", "archivedAt": datetime.now(UTC)},
                )
            except Exception:
                pass
        if staged:
            await restore_staged_workspaces(staged)
        try:
            await db.aiagent.update(
                where={"id": agent.id},
                data={"status": "archived", "archivedAt": datetime.now(UTC)},
            )
        except Exception:
            pass
        raise

    logger.info(
        "[AGENT-CLONE] provisioned agent %s for user %s from template %s",
        agent.id[:8],
        user_id[:8],
        template_agent_id[:8],
    )
    return agent, workspace, conversation


async def _has_agent_or_pending(user_id: str) -> bool:
    """True when the user already has an active workspace or an in-flight provision."""
    if await get_active_workspace(user_id=user_id) is not None:
        return True
    pending = await db.aiagent.find_first(
        where={"userId": user_id, "status": "provisioning"}
    )
    return pending is not None


async def ensure_default_agent_for_user(user_id: str):
    """Idempotently give a user the default cloned agent, if configured.

    * No-op (returns None) when the user already has an active workspace, or when
      no default template is configured, or on any failure (login must not break).
    * Otherwise clones the configured default template and returns the new agent.

    Concurrency: a short-lived Redis lock serializes simultaneous first-logins of
    the same user so two requests can't each create a clone (which would leave an
    orphaned archived workspace). The lock is best-effort — if Redis is down we
    fall back to the DB checks, which still prevent the common (sequential) case.
    """
    template_id = await get_default_template_agent_id()
    if not template_id:
        return None

    if await _has_agent_or_pending(user_id):
        return None

    lock_key = f"lock:ensure_default_agent:{user_id}"
    redis = None
    have_lock = False
    try:
        from app.redis_client import get_redis

        redis = await get_redis()
        # NX + short TTL: whoever wins clones; the TTL auto-releases on crash.
        have_lock = bool(await redis.set(lock_key, "1", nx=True, ex=60))
        if not have_lock:
            return None
    except Exception:
        # Redis unavailable — proceed without the lock (DB checks still apply).
        redis = None

    try:
        # Re-check under the lock (another request may have just finished).
        if await _has_agent_or_pending(user_id):
            return None
        agent, _workspace, _conversation = await clone_template_agent_for_user(
            user_id, template_id
        )
        return agent
    except Exception as exc:
        logger.warning(
            "[AGENT-CLONE] ensure_default_agent_for_user failed for %s: %s",
            user_id[:8],
            exc,
        )
        return None
    finally:
        if redis is not None and have_lock:
            try:
                await redis.delete(lock_key)
            except Exception:
                pass
