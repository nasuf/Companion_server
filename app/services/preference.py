"""Preference learning service.

Thin wrapper around the Postgres entity layer — preferences are stored as
rows in `memory_entities` with entity_type='preference' (category goes in
`role`, value goes in `canonical_name`).
"""

import logging

from app.services.memory.entity_repo import get_user_preferences, upsert_entity
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)


async def update_user_preference(
    user_id: str,
    category: str,
    value: str,
    is_explicit: bool = False,
    workspace_id: str | None = None,
) -> None:
    """Record an expressed preference. Idempotent; repeated calls only
    merge aliases/metadata (mention_count is bumped by the memory pipeline
    when the preference is also linked to a concrete memory)."""
    ws = workspace_id or await resolve_workspace_id(user_id=user_id)
    await upsert_entity(
        user_id=user_id,
        workspace_id=ws,
        canonical_name=value,
        entity_type="preference",
        role=category or None,
        metadata={"explicit": bool(is_explicit)},
    )
    logger.info(
        f"Updated preference for user {user_id}: {category}={value} "
        f"(explicit={is_explicit})"
    )


async def get_preferences(user_id: str, workspace_id: str | None = None) -> list[dict]:
    """Get all preferences for a user in the active (or given) workspace.
    Shape: [{"category": ..., "value": ..., "count": ...}, ...]"""
    ws = workspace_id or await resolve_workspace_id(user_id=user_id)
    return await get_user_preferences(user_id=user_id, workspace_id=ws)
