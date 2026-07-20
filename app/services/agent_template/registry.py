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


async def set_default_template_agent_id(agent_id: str | None) -> None:
    """Persist the default template pointer on the singleton system_config row."""
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
