"""Neo4j graph service.

Node and edge operations for the relationship graph.
"""

import logging

from app.neo4j_client import run_write, run_query

logger = logging.getLogger(__name__)


# --- Node Operations ---

async def create_user_node(user_id: str, workspace_id: str, name: str) -> None:
    await run_write(
        "MERGE (u:User {id: $id, workspace_id: $workspace_id}) SET u.name = $name, u.status = 'active'",
        {"id": user_id, "workspace_id": workspace_id, "name": name},
    )


async def create_ai_node(agent_id: str, workspace_id: str, name: str) -> None:
    await run_write(
        "MERGE (a:AI {id: $id, workspace_id: $workspace_id}) SET a.name = $name, a.status = 'active'",
        {"id": agent_id, "workspace_id": workspace_id, "name": name},
    )


async def create_memory_node(
    memory_id: str,
    workspace_id: str,
    summary: str,
    level: int,
    importance: float,
    main_category: str | None = None,
    sub_category: str | None = None,
) -> None:
    await run_write(
        """
        MERGE (m:Memory {id: $id, workspace_id: $workspace_id})
        SET m.summary = $summary,
            m.level = $level,
            m.importance = $importance,
            m.main_category = $main_category,
            m.sub_category = $sub_category,
            m.status = 'active'
        """,
        {
            "id": memory_id,
            "workspace_id": workspace_id,
            "summary": summary,
            "level": level,
            "importance": importance,
            "main_category": main_category,
            "sub_category": sub_category,
        },
    )


async def merge_entity_node(name: str, entity_type: str) -> None:
    await run_write(
        "MERGE (e:Entity {name: $name}) SET e.type = $type",
        {"name": name, "type": entity_type},
    )


async def merge_topic_node(name: str) -> None:
    await run_write(
        "MERGE (t:Topic {name: $name})",
        {"name": name},
    )


async def merge_preference_node(category: str, value: str) -> None:
    await run_write(
        "MERGE (p:Preference {category: $category, value: $value})",
        {"category": category, "value": value},
    )


async def merge_taxonomy_node(main_category: str, sub_category: str) -> None:
    await run_write(
        """
        MERGE (c:MemoryCategory {main_category: $main_category, sub_category: $sub_category})
        """,
        {"main_category": main_category, "sub_category": sub_category},
    )


# --- Edge Operations ---

async def link_user_memory(user_id: str, workspace_id: str, memory_id: str) -> None:
    await run_write(
        """
        MATCH (u:User {id: $user_id, workspace_id: $workspace_id})
        MATCH (m:Memory {id: $memory_id, workspace_id: $workspace_id})
        MERGE (u)-[:HAS_MEMORY]->(m)
        """,
        {"user_id": user_id, "workspace_id": workspace_id, "memory_id": memory_id},
    )


async def link_memory_entity(memory_id: str, workspace_id: str, entity_name: str) -> None:
    await run_write(
        """
        MATCH (m:Memory {id: $memory_id, workspace_id: $workspace_id})
        MERGE (e:Entity {name: $entity_name})
        MERGE (m)-[:MENTIONS]->(e)
        """,
        {"memory_id": memory_id, "workspace_id": workspace_id, "entity_name": entity_name},
    )


async def link_memory_topic(memory_id: str, workspace_id: str, topic_name: str) -> None:
    await run_write(
        """
        MATCH (m:Memory {id: $memory_id, workspace_id: $workspace_id})
        MERGE (t:Topic {name: $topic_name})
        MERGE (m)-[:RELATED_TO]->(t)
        """,
        {"memory_id": memory_id, "workspace_id": workspace_id, "topic_name": topic_name},
    )


async def link_user_preference(user_id: str, workspace_id: str, category: str, value: str) -> None:
    await run_write(
        """
        MATCH (u:User {id: $user_id, workspace_id: $workspace_id})
        MERGE (p:Preference {category: $category, value: $value})
        MERGE (u)-[r:LIKES]->(p)
        ON CREATE SET r.strength = 0.5
        ON MATCH SET r.strength = r.strength * 0.9 + 0.1
        """,
        {"user_id": user_id, "workspace_id": workspace_id, "category": category, "value": value},
    )


async def link_memory_taxonomy(
    memory_id: str,
    workspace_id: str,
    main_category: str,
    sub_category: str,
) -> None:
    await run_write(
        """
        MATCH (m:Memory {id: $memory_id, workspace_id: $workspace_id})
        MERGE (c:MemoryCategory {main_category: $main_category, sub_category: $sub_category})
        MERGE (m)-[:IN_CATEGORY]->(c)
        """,
        {
            "memory_id": memory_id,
            "workspace_id": workspace_id,
            "main_category": main_category,
            "sub_category": sub_category,
        },
    )


async def link_memory_emotion(
    memory_id: str, workspace_id: str, pleasure: float, arousal: float, dominance: float,
) -> None:
    await run_write(
        """
        MATCH (m:Memory {id: $memory_id, workspace_id: $workspace_id})
        MERGE (e:Emotion {memory_id: $memory_id, workspace_id: $workspace_id})
        SET e.pleasure = $pleasure, e.arousal = $arousal, e.dominance = $dominance
        MERGE (m)-[:HAS_EMOTION]->(e)
        """,
        {
            "memory_id": memory_id,
            "workspace_id": workspace_id,
            "pleasure": pleasure,
            "arousal": arousal,
            "dominance": dominance,
        },
    )


# --- Graph Update from Extraction ---

async def update_graph_from_extraction(
    user_id: str,
    memory_id: str,
    extraction: dict,
    workspace_id: str,
    agent_id: str | None = None,
    user_name: str | None = None,
    agent_name: str | None = None,
) -> None:
    """Update the graph after memory extraction.

    Creates nodes and edges for entities, topics, preferences.
    """
    memory_data = None
    for mem in extraction.get("memories", []):
        if mem.get("summary"):
            memory_data = mem
            break

    if not memory_data:
        return

    # Create memory node
    if user_name:
        await create_user_node(user_id, workspace_id, user_name)
    if agent_id and agent_name:
        await create_ai_node(agent_id, workspace_id, agent_name)

    await create_memory_node(
        memory_id,
        workspace_id,
        memory_data.get("summary", ""),
        memory_data.get("level", 3),
        memory_data.get("importance", 0.5),
        main_category=memory_data.get("main_category"),
        sub_category=memory_data.get("sub_category"),
    )
    await link_user_memory(user_id, workspace_id, memory_id)

    main_category = memory_data.get("main_category")
    sub_category = memory_data.get("sub_category")
    if main_category and sub_category:
        await merge_taxonomy_node(main_category, sub_category)
        await link_memory_taxonomy(memory_id, workspace_id, main_category, sub_category)

    # Link entities
    for entity in extraction.get("entities", []):
        name = entity.get("name", "")
        etype = entity.get("type", "unknown")
        if name:
            await merge_entity_node(name, etype)
            await link_memory_entity(memory_id, workspace_id, name)

    # Link topics
    for topic in extraction.get("topics", []):
        if topic:
            await merge_topic_node(topic)
            await link_memory_topic(memory_id, workspace_id, topic)

    # Link preferences
    for pref in extraction.get("preferences", []):
        cat = pref.get("category", "")
        val = pref.get("value", "")
        if cat and val:
            await merge_preference_node(cat, val)
            await link_user_preference(user_id, workspace_id, cat, val)

    # Link emotion
    emotion = memory_data.get("emotion")
    if emotion:
        await link_memory_emotion(
            memory_id,
            workspace_id,
            emotion.get("pleasure", 0.0),
            emotion.get("arousal", 0.0),
            emotion.get("dominance", 0.0),
        )
