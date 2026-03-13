"""Preference learning service.

score = 0.7 * explicit + 0.3 * inferred
"""

import logging

from app.services.graph_service import link_user_preference
from app.services.graph.queries import get_user_preferences

logger = logging.getLogger(__name__)


async def update_user_preference(
    user_id: str,
    category: str,
    value: str,
    is_explicit: bool = False,
) -> None:
    """Update a user preference in the graph."""
    await link_user_preference(user_id, category, value)
    logger.info(
        f"Updated preference for user {user_id}: "
        f"{category}={value} (explicit={is_explicit})"
    )


async def get_preferences(user_id: str) -> list[dict]:
    """Get all preferences for a user."""
    return await get_user_preferences(user_id)
