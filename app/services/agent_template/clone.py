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
from app.services.memory.retrieval.context_selector import exceeds_injection_limit
from app.services.speech_output.voices import assign_random_voice
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
    "importance",
    "mentionCount",
    "isArchived",
    "occurTime",
    "statementTime",
    "recurrence",
    "provenance",
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

    # 克隆是逐字复制, 所以它同时也是个放大器: 模板里的任何一条问题记忆, 每克隆
    # 一次就多一条。2026-08 实测这个放大倍数是 48 (两个模板 agent 各被克隆 38/10
    # 次), 一条模板记忆的影响面因此是它自己的 49 倍。
    #
    # 注意这里查的是**注入上限**(>180 token, 检索时会被整条跳过), 不是"偏长"。
    # 同一次排查发现的 2276 条 135-180 token 的记忆不在此列 —— 那些能正常注入,
    # 只是粒度粗, 属于另一个问题, 不该用同一个阈值混在一起报。
    #
    # 这里刻意**不**改成"跳过超限条"或"就地拆分":
    #   跳过  → 克隆与模板不再等价, 而且原文丢了, 以后修好模板也没法回填这些克隆;
    #   拆分  → 拆出来的新文本跟模板的 embedding 不再对应, 得在注册热路径上重新
    #          嵌入 (正是下面那段批量 copy 特意要避开的 N+1)。
    # 真正的闸门在模板侧 (registry.set_default_template_agent_id), 这里只负责让
    # "模板脏了"这件事不再无声无息。
    oversized = [
        row for row in template_rows
        if exceeds_injection_limit(getattr(row, "content", "") or "")
    ]
    if oversized:
        logger.warning(
            "[AGENT-CLONE] template workspace %s has %d/%d memories over the "
            "injection limit; cloning them verbatim into %s (they will never be "
            "retrievable — fix the template, then backfill clones)",
            template_workspace_id[:8],
            len(oversized),
            len(template_rows),
            new_workspace_id[:8],
        )

    await db.aimemory.create_many(data=new_rows)

    # Copy embeddings in a single batched INSERT ... SELECT. A per-row loop here
    # is an N+1 round-trip storm (one query per memory) that holds a DB
    # connection for the whole clone and starves the small pool under signup
    # bursts. The VALUES join maps each new id to its template embedding in one
    # statement; a template row without an embedding is simply skipped by the
    # JOIN, and ON CONFLICT keeps the copy idempotent.
    copied = 0
    try:
        values_clause = ",".join(
            f"(${i * 2 + 1},${i * 2 + 2})" for i in range(len(id_pairs))
        )
        flat_args: list[str] = []
        for template_id, new_id in id_pairs:
            flat_args.extend((new_id, template_id))
        copied = await db.execute_raw(
            f"""
            INSERT INTO memory_embeddings (memory_id, embedding)
            SELECT v.new_id::text, e.embedding
            FROM (VALUES {values_clause}) AS v(new_id, template_id)
            JOIN memory_embeddings e ON e.memory_id = v.template_id::text
            ON CONFLICT (memory_id) DO NOTHING
            """,
            *flat_args,
        )
    except Exception as exc:
        # A batch embedding-copy failure must not fail the whole clone; the rows
        # still exist and can be re-embedded lazily later.
        logger.warning(
            "[AGENT-CLONE] batch embedding copy failed for workspace %s: %s",
            new_workspace_id[:8],
            exc,
        )
    logger.info(
        "[AGENT-CLONE] copied %d memories (%d embeddings) into workspace %s",
        len(new_rows),
        copied,
        new_workspace_id[:8],
    )

    # Phase 2-6: carry the entity graph over. Entities are scoped per
    # (user, workspace), so re-upsert them in the clone's scope and link the
    # mapped memory ids. Persona entity volume is small (pet/family/friends),
    # so per-row upserts are fine; failures never abort the clone.
    try:
        await _clone_memory_entities(
            template_workspace_id=template_workspace_id,
            id_pairs=id_pairs,
            user_id=user_id,
            new_workspace_id=new_workspace_id,
        )
    except Exception as exc:
        logger.warning("[AGENT-CLONE] entity graph copy failed: %s", exc)

    return len(new_rows)


async def _clone_memory_entities(
    *,
    template_workspace_id: str,
    id_pairs: list[tuple[str, str]],
    user_id: str,
    new_workspace_id: str,
) -> int:
    """Re-create the template's memory↔entity links in the clone's scope."""
    from app.services.memory.storage.entity_repo import record_entities_for_memory

    rows = await db.query_raw(
        """
        SELECT mm.memory_id, me.canonical_name, me.entity_type, me.role, me.aliases
        FROM memory_mentions mm
        JOIN memory_entities me ON me.id = mm.entity_id
        WHERE mm.workspace_id = $1 AND me.is_archived = false
        """,
        template_workspace_id,
    )
    if not rows:
        return 0

    entities_by_template_mid: dict[str, list[dict]] = {}
    for row in rows:
        mid = str(row.get("memory_id") or "")
        name = (row.get("canonical_name") or "").strip()
        if not mid or not name:
            continue
        entities_by_template_mid.setdefault(mid, []).append({
            "name": name,
            "type": row.get("entity_type") or "other",
            "role": row.get("role"),
            "aliases": row.get("aliases") or None,
        })

    linked = 0
    for template_id, new_id in id_pairs:
        entities = entities_by_template_mid.get(template_id)
        if not entities:
            continue
        linked += await record_entities_for_memory(
            memory_id=new_id,
            memory_source="ai",
            user_id=user_id,
            workspace_id=new_workspace_id,
            entities=entities,
        )
    if linked:
        logger.info("[AGENT-CLONE] linked %d entity edges in clone workspace", linked)
    return linked


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
    try:
        await assign_random_voice(
            agent_id=agent.id,
            gender=agent.gender,
            agent=agent,
        )
    except Exception as exc:
        logger.warning("[AGENT-CLONE] TTS voice assignment failed: %s", exc)
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

    _dispatch_day_one_schedule(agent, user_id)

    logger.info(
        "[AGENT-CLONE] provisioned agent %s for user %s from template %s",
        agent.id[:8],
        user_id[:8],
        template_agent_id[:8],
    )
    return agent, workspace, conversation


def _dispatch_day_one_schedule(agent, user_id: str) -> None:
    """Generate the clone's first daily schedule in the background.

    The daily-schedule cron runs pre-dawn, so an afternoon signup would leave
    the clone with ai_status=None (no delay profile / 隐性状态约束 / 忙闲语义)
    until the next cron run. lifeOverview + MBTI were copied from the template,
    so one background LLM call fills the gap. Best-effort: a failure only means
    the degraded no-schedule day-one behavior we had before.
    """
    try:
        from app.services.mbti import get_mbti
        from app.services.runtime.tasks import fire_background
        from app.services.schedule_domain.schedule import (
            generate_daily_schedule,
            get_life_overview,
        )
    except Exception as exc:
        logger.warning("[AGENT-CLONE] day-one schedule imports failed: %s", exc)
        return

    async def _gen() -> None:
        try:
            overview = await get_life_overview(agent.id)
            await generate_daily_schedule(
                agent.id,
                agent.name,
                get_mbti(agent),
                life_overview=overview,
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning(
                "[AGENT-CLONE] day-one schedule generation failed for %s: %s",
                agent.id[:8],
                exc,
            )

    coro = _gen()
    try:
        fire_background(coro)
    except Exception as exc:
        coro.close()
        logger.warning("[AGENT-CLONE] day-one schedule dispatch failed: %s", exc)


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
