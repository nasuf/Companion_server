"""User portrait generation service.

Generates and updates 200-300 character user portraits from L1/L2 memories.
Runs weekly as a scheduled job.
"""

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.memory.storage import repo as memory_repo
from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.defaults import (
    PORTRAIT_GENERATION_PROMPT,
    PORTRAIT_UPDATE_PROMPT,
)
from app.services.prompting.store import get_prompt_text
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)


async def check_portrait_preconditions(user_id: str, agent_id: str) -> bool:
    """检查首次画像生成前置条件。

    条件:
    - 注册≥24h
    - L2记忆≥20条
    - L1记忆≥5条
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return False

    hours_since_creation = (datetime.now(UTC) - agent.createdAt.replace(
        tzinfo=UTC if agent.createdAt.tzinfo is None else agent.createdAt.tzinfo
    )).total_seconds() / 3600
    if hours_since_creation < 24:
        logger.info(f"Portrait precondition: agent {agent_id} created <24h ago")
        return False

    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    l2_count = await memory_repo.count(
        source="user",
        where={"userId": user_id, "workspaceId": workspace_id, "level": 2, "isArchived": False},
    )
    if l2_count < 20:
        logger.info(f"Portrait precondition: only {l2_count} L2 memories (need 20)")
        return False

    l1_count = await memory_repo.count(
        source="user",
        where={"userId": user_id, "workspaceId": workspace_id, "level": 1, "isArchived": False},
    )
    if l1_count < 5:
        logger.info(f"Portrait precondition: only {l1_count} L1 memories (need 5)")
        return False

    return True


async def generate_portrait(user_id: str, agent_id: str) -> str | None:
    """Generate a user portrait from L1/L2 memories."""
    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    # Check preconditions for first-time generation
    existing = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
    )
    if not existing:
        if not await check_portrait_preconditions(user_id, agent_id):
            return None

    memories = await memory_repo.find_many(
        source="user",
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "level": {"in": [1, 2]},
            "isArchived": False,
        },
        order={"importance": "desc"},
        take=30,
    )

    if not memories:
        logger.info(f"No L1/L2 memories for user {user_id}, skipping portrait")
        return None

    memories_text = "\n".join(
        f"- [L{m.level}] [{m.mainCategory or '未分类'}/{m.subCategory or '其他'}] {m.summary or m.content}"
        for m in memories
    )

    prompt = (await get_prompt_text("portrait.generation")).format(memories=memories_text)

    try:
        portrait = await invoke_text(get_utility_model(), prompt)
    except Exception as e:
        logger.error(f"Portrait generation failed: {e}")
        return None

    # Store portrait
    existing = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
        order={"version": "desc"},
    )
    version = (existing.version + 1) if existing else 1

    await db.userportrait.create(
        data={
            "user": {"connect": {"id": user_id}},
            "agentId": agent_id,
            "version": version,
            "content": portrait,
        }
    )

    logger.info(f"Generated portrait v{version} for user {user_id}")
    return portrait


async def update_portrait_weekly(user_id: str, agent_id: str) -> str | None:
    """Update user portrait based on weekly memory changes."""
    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    # Get previous portrait
    previous = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
        order={"version": "desc"},
    )

    if not previous:
        return await generate_portrait(user_id, agent_id)

    # Get this week's changelog
    one_week_ago = datetime.now(UTC) - timedelta(days=7)

    changelogs = await db.memorychangelog.find_many(
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "createdAt": {"gte": one_week_ago},
        },
        order={"createdAt": "desc"},
        take=50,
    )

    if not changelogs:
        logger.info(f"No changes this week for user {user_id}, keeping portrait")
        return previous.content

    changes_text = "\n".join(
        f"- [{cl.operation}] {cl.newValue or cl.oldValue or ''}"
        for cl in changelogs
    )

    prompt = (await get_prompt_text("portrait.update")).format(
        previous_portrait=previous.content,
        weekly_changes=changes_text,
    )

    try:
        portrait = await invoke_text(get_utility_model(), prompt)
    except Exception as e:
        logger.error(f"Portrait update failed: {e}")
        return previous.content

    await db.userportrait.create(
        data={
            "user": {"connect": {"id": user_id}},
            "agentId": agent_id,
            "version": previous.version + 1,
            "content": portrait,
        }
    )

    # 清理已消费的变更日志
    try:
        deleted = await db.memorychangelog.delete_many(
            where={
                "userId": user_id,
                "workspaceId": workspace_id,
                "createdAt": {"lte": one_week_ago},
            },
        )
        if deleted:
            logger.info(f"Cleaned up {deleted} changelog entries for user {user_id}")
    except Exception as e:
        logger.warning(f"Failed to clean changelog for user {user_id}: {e}")

    logger.info(f"Updated portrait v{previous.version + 1} for user {user_id}")
    return portrait


async def get_latest_portrait(user_id: str, agent_id: str) -> str | None:
    """Get the latest portrait for a user-agent pair."""
    portrait = await db.userportrait.find_first(
        where={"userId": user_id, "agentId": agent_id},
        order={"version": "desc"},
    )
    return portrait.content if portrait else None
