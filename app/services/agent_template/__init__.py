"""Agent template subsystem.

Splits into two cohesive modules:

* ``registry`` — the reserved system user that owns template agents, plus the
  enrollment pool (many templates can be open; new users pick one at random).
* ``clone``    — cheaply instantiating a per-user agent from a template (copying
  persona + L1 memory + embeddings, no LLM), so a new user can chat instantly
  while every downstream state stays isolated per user.

Public API is re-exported here so callers use ``app.services.agent_template``.
"""

from app.services.agent_template.clone import (
    clone_template_agent_for_user,
    ensure_default_agent_for_user,
)
from app.services.agent_template.registry import (
    count_active_clones,
    get_default_template_agent_id,
    get_or_create_template_user,
    is_enrolling,
    is_template_agent,
    list_enrolling_template_ids,
    list_template_agents,
    pick_enrolling_template_id,
    set_default_template_agent_id,
    set_template_enabled,
)

__all__ = [
    "clone_template_agent_for_user",
    "count_active_clones",
    "ensure_default_agent_for_user",
    "get_default_template_agent_id",
    "get_or_create_template_user",
    "is_enrolling",
    "is_template_agent",
    "list_enrolling_template_ids",
    "list_template_agents",
    "pick_enrolling_template_id",
    "set_default_template_agent_id",
    "set_template_enabled",
]
