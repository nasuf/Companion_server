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
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


async def generate_initial_self_memories(
    agent_id: str,
    agent_name: str,
    personality: dict,
    user_id: str,
    *,
    background: str = "",
    values: dict | None = None,
    gender: str = "",
    age: int = 22,
    occupation: str = "",
    city: str = "",
) -> list[str]:
    """DEPRECATED: Replaced by life_story.convert_profile_to_memories + extract_chapter_memories.

    Kept for backward compatibility with daily self-memory generation module.
    """
    prompt = (await get_prompt_text("self_memory.initial")).format(
        name=agent_name,
        gender=gender or "未设定",
        age=age,
        occupation=occupation or "未设定",
        city=city or "未设定",
        personality=str(personality),
        background=background or "暂无",
        values=str(values) if values else "暂无",
    )

    try:
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.error(f"Initial self-memory LLM generation failed: {e}")
        return []

    memories = result.get("memories", [])
    stored_ids = []

    for mem in memories[:5]:
        content = mem.get("content", "")
        if not content or not mem.get("type") or not mem.get("main_category"):
            continue
        mid = await store_memory(
            user_id=user_id,
            content=content,
            summary=content,
            level=1,
            importance=min(1.0, mem.get("importance", 90) / 100),
            memory_type=mem["type"],
            main_category=mem["main_category"],
            sub_category=mem.get("sub_category", "自我认知"),
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

    from app.services.mbti import format_mbti_for_prompt, get_mbti
    prompt = (await get_prompt_text("self_memory.daily")).format(
        ai_name=agent.name,
        personality=format_mbti_for_prompt(get_mbti(agent)) or "中性",
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

        # Per spec §2.3: 0-100 importance score → level
        #   ≥ 85   → L1 核心   |   50-84 → L2 重要
        #   10-49 → L3 模糊    |   0-9   → 不存
        raw_score = float(mem.get("importance", 50))
        if raw_score < 10:
            continue
        elif raw_score >= 85:
            level = 1
        elif raw_score >= 50:
            level = 2
        else:
            level = 3
        importance_unit = min(1.0, raw_score / 100)

        mid = await store_memory(
            user_id=user_id,
            content=mem.get("content", ""),
            summary=mem.get("content", ""),
            level=level,
            importance=importance_unit,
            memory_type=mem.get("type", "life"),
            main_category=mem.get("main_category"),
            sub_category=mem.get("sub_category"),
            source="ai",
        )
        if mid:
            stored_ids.append(mid)

    logger.info(f"Generated {len(stored_ids)} self-memories for agent {agent_id}")
    return stored_ids
