"""Agent template subsystem.

Splits into two cohesive modules:

* ``registry`` — the "template" concept: a dedicated system user that owns
  fully-provisioned template agents, plus the default-template pointer stored in
  ``system_config`` (falling back to the env setting).
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
    is_template_agent,
    list_template_agents,
    set_default_template_agent_id,
)

__all__ = [
    "clone_template_agent_for_user",
    "count_active_clones",
    "ensure_default_agent_for_user",
    "get_default_template_agent_id",
    "get_or_create_template_user",
    "is_template_agent",
    "list_template_agents",
    "set_default_template_agent_id",
]
