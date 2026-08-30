"""Template registry: the system owner + the default-template pointer.

Templates are ordinary, fully-provisioned agents that happen to be owned by a
reserved *system user* (so they never appear in any real user's agent list and
are never chatted with by end users). The "which template is the default for new
users" pointer lives in ``system_config.default_template_agent_id`` and is
managed from the web admin (Agent管理 → 模板管理).

The pointer is read/written with raw SQL so it works regardless of whether the
Prisma client has been regenerated for the new column.
"""

from __future__ import annotations

import logging
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
