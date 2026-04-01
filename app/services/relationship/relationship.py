"""Relationship evolution tracking.

Tracks trust score, interaction frequency/depth.
"""

import logging

from app.neo4j_client import run_write, run_query

logger = logging.getLogger(__name__)


async def update_relationship(
    user_id: str,
    agent_id: str,
    workspace_id: str,
    interaction_quality: float = 0.5,
) -> None:
    """Update user-agent relationship metrics."""
    await run_write(
        """
        MATCH (u:User {id: $user_id, workspace_id: $workspace_id})
        MATCH (a:AI {id: $agent_id, workspace_id: $workspace_id})
        MERGE (u)-[r:INTERACTS_WITH]->(a)
        ON CREATE SET
            r.trust = $quality,
            r.interaction_count = 1,
            r.created_at = datetime()
        ON MATCH SET
            r.trust = r.trust * 0.9 + $quality * 0.1,
            r.interaction_count = r.interaction_count + 1
        """,
        {
            "user_id": user_id,
            "agent_id": agent_id,
            "workspace_id": workspace_id,
            "quality": interaction_quality,
        },
    )


async def get_relationship(user_id: str, agent_id: str, workspace_id: str) -> dict | None:
    """Get relationship metrics between user and agent."""
    results = await run_query(
        """
        MATCH (u:User {id: $user_id, workspace_id: $workspace_id})-[r:INTERACTS_WITH]->(a:AI {id: $agent_id, workspace_id: $workspace_id})
        RETURN r.trust AS trust, r.interaction_count AS interaction_count
        """,
        {"user_id": user_id, "agent_id": agent_id, "workspace_id": workspace_id},
    )
    return results[0] if results else None
