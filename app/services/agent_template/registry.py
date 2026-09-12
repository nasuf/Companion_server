"""Template registry: system owner + enrollment pool for new-user cloning.

Templates are ordinary, fully-provisioned agents owned by a reserved *system
user* (they never appear in a real user's agent list). Many templates can be
open at once; a new signup randomly clones one from the open pool
(``status='active' AND template_enabled=TRUE``). Stopping a template flips
only ``template_enabled`` — already-cloned users keep their independent agent.

The legacy ``system_config.default_template_agent_id`` pointer is still
readable (env fallback / old admin clients) but is no longer the matching
source of truth.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)

# Reserved account that owns all template agents. Chosen to be unreachable via
# normal registration (username validator forbids these characters anyway).
TEMPLATE_SYSTEM_USERNAME = "__companion_template_system__"


async def get_or_create_template_user():
    """Return the reserved system user that owns template agents (create if absent)."""
    user = await db.user.find_unique(where={"username": TEMPLATE_SYSTEM_USERNAME})
    if user:
        return user
    return await db.user.create(
        data={
            "username": TEMPLATE_SYSTEM_USERNAME,
            "hashedPassword": None,
            "role": "user",
            "status": "active",
        }
    )


# Cached template owner id — the system user's row never changes, so per-agent
# crons (which call this every tick) reuse it instead of hitting the DB.
_template_owner_id_cache: str | None = None


async def get_template_owner_id() -> str | None:
    """Return the template system user's id **read-only** (no create), or None.

    Cron enumerations (daily schedule/summary, proactive scan, special dates)
    use this to exclude template agents: the template must stay a frozen clone
    source and never accumulate its own runtime state (schedules, self-memory,
    proactive sends). Returns None when no template user exists yet — callers
    then apply no exclusion, which is correct because there are no templates.
    """
    global _template_owner_id_cache
    if _template_owner_id_cache is not None:
        return _template_owner_id_cache
    user = await db.user.find_unique(where={"username": TEMPLATE_SYSTEM_USERNAME})
    if user:
        _template_owner_id_cache = user.id
        return user.id
    return None


async def list_template_agents() -> list[Any]:
    """All template agents, newest first — including archived ones.

    Archived templates are still listed so an admin can see and delete legacy
    rows. Historically a new template archived the template system user's other
    templates (single-active-agent staging), leaving old templates invisible and
    thus undeletable from the admin UI. That staging is now disabled on the
    template path, but pre-existing archived templates must remain manageable.
    """
    owner = await get_or_create_template_user()
    return await db.aiagent.find_many(
        where={"userId": owner.id},
        order={"createdAt": "desc"},
    )


async def is_template_agent(agent_id: str) -> bool:
    """True when ``agent_id`` belongs to the template system user."""
    owner = await get_or_create_template_user()
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    return bool(agent and agent.userId == owner.id)


def is_enrolling(agent: Any) -> bool:
    """True when this template is in the new-user matching pool."""
    if getattr(agent, "status", "") != "active":
        return False
    flag = getattr(agent, "templateEnabled", None)
    # Column missing from an old Prisma client → treat provisioned templates
    # as open (matches the SQL DEFAULT TRUE).
    if flag is None:
        return True
    return bool(flag)


async def list_enrolling_template_ids() -> list[str]:
    """Ids of fully-provisioned templates currently open for cloning."""
    owner_id = await get_template_owner_id()
    if not owner_id:
        return []
    try:
        rows = await db.query_raw(
            """
            SELECT id FROM ai_agents
            WHERE user_id = $1
              AND status = 'active'
              AND template_enabled = TRUE
            ORDER BY created_at ASC
            """,
            owner_id,
        )
    except Exception as exc:
        logger.warning("[TEMPLATE] list_enrolling_template_ids failed: %s", exc)
        return []
    return [str(row["id"]) for row in rows if row.get("id")]


def pick_enrolling_template_id(ids: list[str]) -> str | None:
    """Uniform random choice; None when the pool is empty."""
    if not ids:
        return None
    return secrets.choice(ids)


async def _restore_template_runtime(agent_id: str) -> None:
    """Bring a legacy-archived template back to status=active.

    Only this template's agent row + its own workspace are touched. Cloned
    user agents (source_template_id = this id) are independent and stay as-is.
    """
    from app.services.workspace.workspaces import reactivate_workspace

    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if agent and getattr(agent, "status", "") != "active":
        await db.aiagent.update(
            where={"id": agent_id},
            data={"status": "active", "archivedAt": None},
        )
    workspace = await db.chatworkspace.find_first(
        where={"agentId": agent_id},
        order={"createdAt": "desc"},
    )
    if workspace is None:
        raise ValueError("模板没有工作区，无法开放给新用户")
    if getattr(workspace, "status", "") != "active":
        await reactivate_workspace(workspace.id)


async def set_template_enabled(agent_id: str, enabled: bool) -> None:
    """Open or close a template for new-user matching.

    Enable restores a wrongly-archived template (workspace + agent row only)
    *before* the oversized-memory check, so a dirty archived template becomes
    editable instead of staying stuck. Disable only flips the flag.
    """
    if enabled:
        # Restore first so a legacy-archived template becomes editable even if
        # the oversized check then refuses to open it for new users.
        await _restore_template_runtime(agent_id)
        oversized = await count_oversized_memories(agent_id)
        if oversized:
            raise ValueError(
                f"该 agent 有 {oversized} 条记忆超过检索单条上限, 不能开放给新用户 —— "
                f"克隆会逐字复制, 每个新用户都会继承这些永远检索不到的记忆。"
                f"请先用 scripts/split_oversized_memories.py 拆分后重试。"
            )
    await db.execute_raw(
        "UPDATE ai_agents SET template_enabled = $1, updated_at = now() WHERE id = $2",
        enabled,
        agent_id,
    )
    logger.info(
        "[TEMPLATE] enrollment %s for %s",
        "open" if enabled else "closed",
        agent_id[:8],
    )


async def count_active_clones(template_agent_id: str) -> int:
    """How many in-use (active) agents were cloned from this template."""
    try:
        rows = await db.query_raw(
            "SELECT count(*)::int AS n FROM ai_agents "
            "WHERE source_template_id = $1 AND status = 'active'",
            template_agent_id,
        )
        return int(rows[0]["n"]) if rows else 0
    except Exception as exc:
        logger.warning("[TEMPLATE] count_active_clones failed: %s", exc)
        return 0


async def get_default_template_agent_id() -> str | None:
    """Resolve the default template agent id: DB pointer first, then env."""
    try:
        rows = await db.query_raw(
            "SELECT default_template_agent_id FROM system_config WHERE id = 1"
        )
        if rows:
            value = rows[0].get("default_template_agent_id")
            if value:
                return str(value)
    except Exception as exc:
        logger.warning("[TEMPLATE] read default_template_agent_id failed: %s", exc)

    env_value = (settings.default_template_agent_id or "").strip()
    return env_value or None


async def count_oversized_memories(agent_id: str) -> int:
    """How many of this agent's persona memories can never be injected.

    Only memories_ai: a template's user-side table is empty by construction
    (nobody has chatted with it), and clone.py copies only the AI rows anyway,
    so the user table cannot contribute to what a clone inherits.

    "Oversized" here means over the *injection* limit — a row that
    select_context skips whole. Merely long-but-usable rows are a separate
    (granularity) concern and deliberately do not block promotion.

    Cheap enough for admin paths: one indexed query plus a token estimate per
    row over a single agent's few hundred rows.
    """
    from app.services.memory.retrieval.context_selector import exceeds_injection_limit

    rows = await db.query_raw(
        """
        SELECT m.content
        FROM memories_ai m
        JOIN chat_workspaces w ON w.id = m.workspace_id
        WHERE w.agent_id = $1 AND m.is_archived = false
        """,
        agent_id,
    )
    return sum(1 for r in rows if exceeds_injection_limit(r.get("content") or ""))


async def set_default_template_agent_id(agent_id: str | None) -> None:
    """Persist the default template pointer on the singleton system_config row.

    Refuses to promote an agent whose persona contains memories over the
    injection limit. This is the choke point worth guarding: cloning copies
    memory rows verbatim, so a dirty template does not stay one bad agent —
    it becomes one bad agent per signup, forever (2026-08: 2 such templates
    accounted for ~2000 unusable rows across 48 clones). Failing loudly here
    costs an admin one retry; failing to check costs every future user.

    Deliberately not a warning: an admin who sees "default template set" has
    no reason to go looking at a log line, which is exactly how the previous
    round went unnoticed for a month.
    """
    if agent_id:
        oversized = await count_oversized_memories(agent_id)
        if oversized:
            raise ValueError(
                f"该 agent 有 {oversized} 条记忆超过检索单条上限, 不能设为默认模板 —— "
                f"克隆会逐字复制, 每个新用户都会继承这些永远检索不到的记忆。"
                f"请先用 scripts/split_oversized_memories.py 拆分后重试。"
            )
    await db.execute_raw(
        """
        INSERT INTO system_config (id, default_template_agent_id, updated_at)
        VALUES (1, $1, now())
        ON CONFLICT (id)
        DO UPDATE SET default_template_agent_id = $1, updated_at = now()
        """,
        agent_id,
    )
    logger.info(
        "[TEMPLATE] default template set to %s",
        (agent_id[:8] if agent_id else "<none>"),
    )
