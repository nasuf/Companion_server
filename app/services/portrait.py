"""User portrait generation service.

Generates and updates 200-300 character user portraits from L1/L2 memories.
Runs weekly as a scheduled job.
"""

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.memory.storage import repo as memory_repo
from app.services.llm.models import get_utility_model, invoke_text
from app.services.profile_tags import has_active_profile_tags, refresh_profile_tags
from app.services.prompting.store import get_prompt_text
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)


async def _refresh_tags_best_effort(
    user_id: str,
    agent_id: str,
    *,
    workspace_id: str | None,
    portrait: str | None,
) -> None:
    try:
        await refresh_profile_tags(
            user_id,
            agent_id,
            workspace_id=workspace_id,
            portrait=portrait,
        )
    except Exception as exc:
        logger.warning("Profile tag refresh failed for user=%s agent=%s: %s", user_id, agent_id, exc)


# 首次画像的前置条件。
#
# 旧条件是 `L2 ≥ 20 AND L1 ≥ 5`, 生产上从未被满足过一次 —— 15 个有记忆的用户
# 里 0 个达标, user_portraits 表一直是空的。原因是这两个数在真实数据里**此消彼长**:
# 层级由 importance 推导, 说身份事实的用户攒 L1 (实测有人 L1=23/L2=5), 聊日常的
# 用户攒 L2 (L1=0/L2=23), 两者做 AND 就成了不可达的门。
#
# 画像真正需要的是"有没有足够素材写 200 字", 跟素材落在哪一层无关。所以改看总量,
# 并补一个对话量下限 —— 只有记忆没有对话时, 画像会写得像档案摘要而不是人的印象。
MIN_MEMORIES_FOR_PORTRAIT = 15   # L1 + L2 合计
MIN_USER_MESSAGES_FOR_PORTRAIT = 30
MIN_AGENT_AGE_HOURS = 24


async def check_portrait_preconditions(user_id: str, agent_id: str) -> bool:
    """检查首次画像生成前置条件 (见上方常量的说明)。"""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return False

    hours_since_creation = (datetime.now(UTC) - agent.createdAt.replace(
        tzinfo=UTC if agent.createdAt.tzinfo is None else agent.createdAt.tzinfo
    )).total_seconds() / 3600
    if hours_since_creation < MIN_AGENT_AGE_HOURS:
        logger.info(f"Portrait precondition: agent {agent_id} created <24h ago")
        return False

    workspace_id = await resolve_workspace_id(user_id=user_id, agent_id=agent_id)
    recallable = await memory_repo.count(
        source="user",
        where={
            "userId": user_id,
            "workspaceId": workspace_id,
            "level": {"in": [1, 2]},
            "isArchived": False,
        },
    )
    if recallable < MIN_MEMORIES_FOR_PORTRAIT:
        logger.info(
            f"Portrait precondition: only {recallable} L1+L2 memories "
            f"(need {MIN_MEMORIES_FOR_PORTRAIT})"
        )
        return False

    # 与上面的记忆统计保持同一个作用域: 都按 workspace 收口, 且排除用户已删除的
    # 会话 —— 删掉的对话不该继续把人推过画像门槛。
    user_messages = await db.message.count(
        where={
            "role": "user",
            "conversation": {
                "is": {
                    "userId": user_id,
                    "agentId": agent_id,
                    "workspaceId": workspace_id,
                    "isDeleted": False,
                },
            },
        },
    )
    if user_messages < MIN_USER_MESSAGES_FOR_PORTRAIT:
        logger.info(
            f"Portrait precondition: only {user_messages} user messages "
            f"(need {MIN_USER_MESSAGES_FOR_PORTRAIT})"
        )
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
        f"- [L{m.level}] [{m.mainCategory or '未分类'}/{m.subCategory or '其他'}] {m.content}"
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
    await _refresh_tags_best_effort(
        user_id,
        agent_id,
        workspace_id=workspace_id,
        portrait=portrait,
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
        if not await has_active_profile_tags(
            user_id,
            workspace_id,
            agent_id=agent_id,
        ):
            await _refresh_tags_best_effort(
                user_id,
                agent_id,
                workspace_id=workspace_id,
                portrait=previous.content,
            )
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
    await _refresh_tags_best_effort(
        user_id,
        agent_id,
        workspace_id=workspace_id,
        portrait=portrait,
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
