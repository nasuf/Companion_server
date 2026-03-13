"""Graph query service.

Queries Neo4j for relationship context, preferences, entities.
"""

import asyncio
import logging

from app.neo4j_client import run_query

logger = logging.getLogger(__name__)


async def get_related_memories(user_id: str, limit: int = 10) -> list[dict]:
    """Get user's most important memories from graph."""
    return await run_query(
        """
        MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
        RETURN m.id AS id, m.summary AS summary, m.importance AS importance
        ORDER BY m.importance DESC
        LIMIT $limit
        """,
        {"user_id": user_id, "limit": limit},
    )


async def get_user_preferences(user_id: str) -> list[dict]:
    """Get user's preferences from graph."""
    return await run_query(
        """
        MATCH (u:User {id: $user_id})-[r:LIKES]->(p:Preference)
        RETURN p.category AS category, p.value AS value, r.strength AS strength
        ORDER BY r.strength DESC
        LIMIT 20
        """,
        {"user_id": user_id},
    )


async def get_entity_context(entity_name: str) -> list[dict]:
    """Get memories related to a specific entity."""
    return await run_query(
        """
        MATCH (e:Entity {name: $name})<-[:MENTIONS]-(m:Memory)
        RETURN m.id AS id, m.summary AS summary, m.importance AS importance
        ORDER BY m.importance DESC
        LIMIT 10
        """,
        {"name": entity_name},
    )


async def get_relationship_context(user_id: str) -> dict:
    """Get full relationship context for prompt building.

    Returns dict with topics and entities.
    """
    # Run both queries concurrently
    topics_result, entities_result = await asyncio.gather(
        run_query(
            """
            MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)-[:RELATED_TO]->(t:Topic)
            RETURN t.name AS name, count(m) AS count
            ORDER BY count DESC
            LIMIT 10
            """,
            {"user_id": user_id},
        ),
        run_query(
            """
            MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)-[:MENTIONS]->(e:Entity)
            RETURN e.name AS name, e.type AS type, count(m) AS count
            ORDER BY count DESC
            LIMIT 10
            """,
            {"user_id": user_id},
        ),
    )

    return {
        "topics": [r["name"] for r in topics_result],
        "entities": [f"{r['name']} ({r.get('type', '')})" for r in entities_result],
    }
