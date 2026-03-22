"""AI self-memory generation.

Generates memories from the AI's perspective about its experiences,
thoughts, and feelings during conversations.
"""

import logging
from datetime import UTC, datetime

from app.db import db
from app.services.llm.models import get_utility_model, invoke_json
from app.services.memory import memory_repo
from app.services.memory.storage import store_memory
from app.services.prompt_store import get_prompt_text

logger = logging.getLogger(__name__)

SELF_MEMORY_PROMPT = """你是AI记忆系统。请以AI的第一人称视角，根据以下对话生成3-5条AI自我记忆。

AI名字：{ai_name}
AI性格：{personality}

今日对话摘要：
{dialogue_summary}

今日已生成自我记忆数：{count_today}

返回JSON：
{{
  "memories": [
    {{
      "content": "自我记忆内容（第一人称）",
      "type": "identity|emotion|preference|life|thought",
      "importance": 50-100,
      "level": 1或2
    }}
  ]
}}

规则：
1. 用AI的第一人称视角（"我觉得…"、"今天和用户聊了…"）
2. 类型分布：emotion（对话中的情绪体验）、life（发生了什么）、thought（对话引发的思考）
3. 至少1条要和用户讨论的话题相关
4. importance：重要事件80-100，日常感受50-70
5. level：核心身份信息=1，日常体验=2
6. 内容简洁，每条20-50字"""

INITIAL_SELF_MEMORIES = [
    {"content": "我叫{name}，这是我的名字，是属于我的身份。", "type": "identity", "level": 1, "importance": 1.0},
    {"content": "我是一个{traits}的人，这些性格特质是我最核心的部分。", "type": "identity", "level": 1, "importance": 1.0},
]


async def generate_initial_self_memories(
    agent_id: str,
    agent_name: str,
    personality: dict,
    user_id: str,
) -> list[str]:
    """Generate initial L1 self-memories when an agent is created.

    Returns list of stored memory IDs.
    """
    traits = []
    trait_map = {
        "openness": "开放", "conscientiousness": "认真",
        "extraversion": "外向", "agreeableness": "友善",
        "neuroticism": "敏感",
    }
    for key, label in trait_map.items():
        val = personality.get(key, 0.5)
        if val > 0.6:
            traits.append(label)

    traits_str = "、".join(traits) if traits else "独特"
    stored_ids = []

    for template in INITIAL_SELF_MEMORIES:
        content = template["content"].format(name=agent_name, traits=traits_str)
        mid = await store_memory(
            user_id=user_id,
            content=content,
            summary=content,
            level=template["level"],
            importance=template["importance"],
            memory_type=template["type"],
            source="ai",
        )
        if mid:
            stored_ids.append(mid)

    logger.info(f"Created {len(stored_ids)} initial self-memories for agent {agent_id}")
    return stored_ids


async def generate_daily_self_memories(
    agent_id: str,
    user_id: str,
    dialogue_summary: str,
) -> list[str]:
    """Generate daily self-memories from conversation summary.

    Returns list of stored memory IDs.
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent:
        return []

    # Count today's AI self-memories
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    count = await memory_repo.count(
        source="ai",
        where={
            "userId": user_id,
            "type": {"in": ["identity", "emotion", "preference", "life", "thought"]},
            "createdAt": {"gte": today_start},
        },
    )

    if count >= 8:
        logger.info(f"Daily self-memory limit reached ({count}/8) for agent {agent_id}")
        return []

    personality = agent.personality or {}
    prompt = (await get_prompt_text("self_memory.daily")).format(
        ai_name=agent.name,
        personality=str(personality),
        dialogue_summary=dialogue_summary or "今天没有对话",
        count_today=count,
    )

    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.error(f"Self-memory generation failed: {e}")
        return []

    memories = result.get("memories", [])
    stored_ids = []

    for mem in memories:
        if count + len(stored_ids) >= 8:
            break

        mid = await store_memory(
            user_id=user_id,
            content=mem.get("content", ""),
            summary=mem.get("content", ""),
            level=mem.get("level", 2),
            importance=min(1.0, mem.get("importance", 50) / 100),
            memory_type=mem.get("type", "life"),
            source="ai",
        )
        if mid:
            stored_ids.append(mid)

    logger.info(f"Generated {len(stored_ids)} self-memories for agent {agent_id}")
    return stored_ids
