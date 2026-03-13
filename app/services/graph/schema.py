"""Neo4j schema initialization.

Creates constraints and indexes for the graph database.
"""

import logging

from app.neo4j_client import run_write

logger = logging.getLogger(__name__)

SCHEMA_QUERIES = [
    # Constraints
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT ai_id IF NOT EXISTS FOR (a:AI) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
    # Indexes
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
    "CREATE INDEX preference_category IF NOT EXISTS FOR (p:Preference) ON (p.category)",
]


async def init_graph_schema():
    """Initialize Neo4j schema with constraints and indexes."""
    for query in SCHEMA_QUERIES:
        try:
            await run_write(query)
        except Exception as e:
            logger.warning(f"Schema query skipped (may already exist): {e}")
    logger.info("Neo4j schema initialized")
