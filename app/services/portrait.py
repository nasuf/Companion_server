"""User portrait generation service.

Generates and updates 200-300 character user portraits from L1/L2 memories.
Runs weekly as a scheduled job.
"""

import logging
from datetime import UTC, datetime, timedelta

from app.db import db
from app.services.memory.behaviour_signals import collect_behavioural_facts
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


def _warn_if_behaviour_dropped(key: str, template: str) -> None:
    """提示词里没有 {behaviour} 占位符时告警。

    `str.format` 对多余的 kwarg 是静默忽略的 —— 后台把占位符删掉之后, 行为观察会
    无声地从画像里消失, 没有异常、没有日志, 画像看起来照常生成。这类"功能还在但
    不再生效"的状态在这个项目里出过好几次 (夜间 cron、聚类阈值、画像门槛本身),
    共同点都是失效时太安静。
    """
    if "{behaviour}" not in template:
        logger.error(
            "%s 缺少 {behaviour} 占位符 —— 互动观察不会进入画像。"
            "若是后台改动所致, 补回占位符; 若是刚改过代码默认值, 等下次部署同步。",
            key,
        )


async def _collect_behaviour_facts(
    user_id: str, agent_id: str, workspace_id: str | None,
) -> list:
    """取行为事实; 失败或无数据都返回空列表。

    单独拆出来是因为周更要先知道"有没有新观察"才能决定要不要重写画像 —— 只看
    memory changelog 的话, 一个天天来聊但没说出新事实的用户永远触发不了更新。
    """
    try:
        return await collect_behavioural_facts(
            user_id=user_id, agent_id=agent_id, workspace_id=workspace_id,
        )
    except Exception as e:
        logger.warning("Behavioural facts unavailable for portrait: %s", e)
        return []


def _render_behaviour(facts: list) -> str:
    if not facts:
        return "（互动数据还不够，暂无可靠观察）"
    return "\n".join(f"- {fact.statement}" for fact in facts)


async def _behaviour_section(
    user_id: str, agent_id: str, workspace_id: str | None,
) -> str:
    """互动行为观察, 作为画像的第二类输入。

    记忆记录的是用户**说过**的话。但有一类信息用户从不会说出口, 只体现在行为里 ——
    什么时候来找 AI、来的时候情绪如何、习惯长句还是短句、主动搭话理不理。画像本来
    就是"我对这个人的整体了解", 缺了这一半是不完整的。

    这些观察曾经被做成独立的记忆条目写进检索池, 实测行不通: 72 条真实消息里只有
    7% 能召回它们, 且召回时多是误配 ("你平时什么时候工作啊" 召回 "用户习惯在 21 点
    聊天" —— 只是"时间"这个词撞上了)。原因是它们是**特质**不是事实: 向量检索按话题
    相似度建索引, 而"他习惯用短句"不关于任何话题, 它关于怎么回应所有话题。
    画像是必然注入的, 正好是这类信息该待的地方。

    取不到就返回占位符 —— 画像不该因为统计失败而生成不出来。
    """
    return _render_behaviour(
        await _collect_behaviour_facts(user_id, agent_id, workspace_id)
    )


async def generate_portrait(user_id: str, agent_id: str) -> str | None:
    """Generate a user portrait from L1/L2 memories plus interaction behaviour."""
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

    template = await get_prompt_text("portrait.generation")
    behaviour = await _behaviour_section(user_id, agent_id, workspace_id)
    _warn_if_behaviour_dropped("portrait.generation", template)
    prompt = template.format(memories=memories_text, behaviour=behaviour)

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

    # 记忆没变不代表没有新信息: 互动方式 (什么时候来、情绪基调、消息长短) 会先于
    # 记忆变化。只看 changelog 的话, 一个天天来聊但没说出任何新事实的用户, 画像里
    # 的相处方式会一直停在几周前。
    behaviour_facts = await _collect_behaviour_facts(user_id, agent_id, workspace_id)
    if not changelogs and not behaviour_facts:
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

    # 现在可能是"记忆没变但互动方式变了"才走到这里, 所以要显式说明本周无记忆变化 ——
    # 留空的话模板里会出现一段空白, 模型容易当成"记忆被清空了"。
    changes_text = "\n".join(
        f"- [{cl.operation}] {cl.newValue or cl.oldValue or ''}"
        for cl in changelogs
    ) or "（本周没有新的记忆变化）"

    # 周更也要带上 —— 只在首次生成时注入的话, 画像会随着每周重写慢慢把行为观察
    # 冲掉, 几周后又退回"只知道他说过什么"。
    template = await get_prompt_text("portrait.update")
    _warn_if_behaviour_dropped("portrait.update", template)
    prompt = template.format(
        previous_portrait=previous.content,
        weekly_changes=changes_text,
        behaviour=_render_behaviour(behaviour_facts),
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
