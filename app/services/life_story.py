"""AI 人生经历生成服务。

流程:
  1. 选择匹配的 CharacterProfile (性别匹配, published 状态)
  2. 生成人生大纲 (12-15 个章节标题+关键事件)
  3. 逐章提取结构化记忆 (每章 5-10 条, 与聊天记忆格式一致)
  4. 写入 AI L1 记忆
  5. 通过 Redis 实时更新进度, 供前端轮询
"""

from __future__ import annotations

import json
import logging
import random

from app.db import db
from app.redis_client import get_redis
from app.services.llm.models import get_chat_model, invoke_json
from app.services.memory.storage import store_memory
from app.services.memory.taxonomy import TAXONOMY

logger = logging.getLogger(__name__)

# Redis key for provisioning progress
PROGRESS_KEY_PREFIX = "provision_progress:"
PROGRESS_TTL = 3600  # 1 hour


def _taxonomy_description() -> str:
    """将完整 taxonomy 格式化为 LLM 可读文本，供大纲 prompt 使用。"""
    lines = []
    for main_cat, sub_cats in TAXONOMY.items():
        subs = ", ".join(sub_cats)
        lines.append(f"  {main_cat}: [{subs}]")
    return "\n".join(lines)


async def set_progress(agent_id: str, stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
    """Update provisioning progress in Redis."""
    redis = await get_redis()
    percent = 0
    if stage == "selecting_profile":
        percent = 2
    elif stage == "generating_outline":
        percent = 8
    elif stage == "generating_chapter":
        percent = 10 + int(75 * current / max(total, 1))
    elif stage == "storing_memories":
        percent = 88 + int(10 * current / max(total, 1))
    elif stage == "complete":
        percent = 100
    elif stage == "failed":
        percent = -1

    data = json.dumps({
        "stage": stage,
        "current": current,
        "total": total,
        "percent": percent,
        "message": message,
    }, ensure_ascii=False)
    await redis.set(f"{PROGRESS_KEY_PREFIX}{agent_id}", data, ex=PROGRESS_TTL)


async def get_progress(agent_id: str) -> dict | None:
    """Get provisioning progress from Redis."""
    redis = await get_redis()
    raw = await redis.get(f"{PROGRESS_KEY_PREFIX}{agent_id}")
    if not raw:
        return None
    return json.loads(raw)


async def select_character_profile(gender: str | None) -> dict | None:
    """Select a matching published CharacterProfile by gender.

    Returns dict with profile id + data, or None if no match.
    """
    where: dict = {"status": "published"}
    # Find profiles matching gender in JSON data
    profiles = await db.characterprofile.find_many(
        where=where,
        include={"career": True},
    )

    # Filter by gender (stored in data.identity.gender)
    matching = []
    for p in profiles:
        data = p.data if isinstance(p.data, dict) else {}
        identity = data.get("identity", {})
        if isinstance(identity, dict):
            p_gender = identity.get("gender")
            if gender and p_gender and p_gender != gender:
                continue
        matching.append(p)

    if not matching:
        # Fallback: any published profile
        matching = profiles

    if not matching:
        return None

    selected = random.choice(matching)
    return {
        "id": selected.id,
        "data": selected.data if isinstance(selected.data, dict) else {},
        "career_id": getattr(selected, "careerId", None),
        "career": {
            "title": selected.career.title if selected.career else None,
            "duties": selected.career.duties if selected.career else None,
        } if selected.career else None,
    }


async def generate_outline(
    profile_data: dict,
    name: str,
    gender: str | None,
    personality: dict | None,
    seven_dim: dict | None,
) -> list[dict]:
    """Generate life story outline: 12-15 chapters with titles and key events."""
    identity = profile_data.get("identity", {})
    career = profile_data.get("career", {})
    education = profile_data.get("education_knowledge", {})
    values = profile_data.get("values", {})
    likes = profile_data.get("likes", {})
    abilities = profile_data.get("abilities", {})

    prompt = f"""你是一个人物传记作家。请为以下角色生成一份人生经历大纲。

角色基本信息：
- 姓名: {name}
- 性别: {gender or '未设定'}
- 年龄: {identity.get('age', 22)}
- 出生日期: {identity.get('birthday', '未知')}
- 城市: {identity.get('location', '未知')}
- 教育背景: {education.get('degree', '未知')}
- 职业: {career.get('title', '未知')}
- 工作内容: {career.get('duties', '未知')}
- 家庭: {identity.get('family', '未知')}
- 性格特点: {json.dumps(seven_dim, ensure_ascii=False) if seven_dim else '未设定'}
- 喜好: {json.dumps(likes, ensure_ascii=False) if likes else '未知'}
- 价值观: {values.get('motto', '未知')}
- 人生目标: {values.get('goal', '未知')}
- 擅长: {json.dumps(abilities.get('good_at', []), ensure_ascii=False)}

请生成12-15个人生阶段的大纲。每个阶段包含：
1. title: 阶段标题（如"童年时光"）
2. age_range: 年龄范围（如"0-6岁"）
3. key_events: 2-3个关键事件描述（简短）
4. main_category: 该章节最匹配的记忆主分类（必须从下方分类表中选择）
5. sub_category: 该章节最匹配的记忆子分类（必须从对应主分类的子分类中选择）

记忆分类表（必须严格从中选择）：
{_taxonomy_description()}

要求：
- 从出生到现在的完整人生轨迹
- 事件要与角色的职业、性格、喜好逻辑自洽
- 以第一人称视角（"我"）
- 包含成长转折点、关键人物、情感体验
- 12-15个章节应覆盖尽可能多的不同分类（身份/偏好/生活/情绪/思维），而非集中在某一类

返回JSON数组: [{{"title": "...", "age_range": "...", "key_events": ["...", "..."], "main_category": "身份", "sub_category": "成长地"}}]"""

    model = get_chat_model()
    result = await invoke_json(model, prompt)
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "chapters" in result:
        return result["chapters"]
    return []


async def extract_chapter_memories(
    chapter: dict,
    outline_summary: str,
    name: str,
    chapter_index: int,
    total_chapters: int,
) -> list[dict]:
    """Extract 5-10 structured memories from a single chapter."""
    prompt = f"""你正在为角色「{name}」梳理人生经历第 {chapter_index + 1}/{total_chapters} 阶段的记忆。

完整人生大纲：
{outline_summary}

当前阶段：
- 标题: {chapter.get('title', '')}
- 年龄阶段: {chapter.get('age_range', '')}
- 关键事件: {json.dumps(chapter.get('key_events', []), ensure_ascii=False)}

请为这个阶段生成 5-10 条结构化记忆。每条记忆是一个独立的事实/经历/感受，格式与聊天记忆一致。

记忆分类表（必须严格选择）：
{_taxonomy_description()}

返回 JSON：
{{
  "memories": [
    {{
      "summary": "一句话描述（20-80字），以'我'开头",
      "level": 1,
      "importance": 0.85,
      "type": "identity|preference|life|emotion|thought",
      "main_category": "身份|偏好|生活|情绪|思维",
      "sub_category": "从上方分类表中选择"
    }}
  ]
}}

要求：
1. 每条记忆只描述一个具体事实，如"我5岁时养了一只叫小白的猫"
2. 不要写成大段叙事，每条 summary 控制在 20-80 字
3. 以第一人称「我」开头
4. 涵盖具体人物、地点、事件、情感、决定等细节
5. 不同记忆应覆盖不同的分类（身份/偏好/生活/情绪/思维）
6. importance 范围: 核心身份事实 0.9-0.95, 重要经历 0.8-0.9, 情感/偏好 0.7-0.85
7. sub_category 必须从分类表中原样复制，禁止自造"""

    model = get_chat_model()
    result = await invoke_json(model, prompt)
    if isinstance(result, dict) and "memories" in result:
        return result["memories"]
    if isinstance(result, list):
        return result
    return []


async def store_chapter_memories(
    agent_id: str,
    user_id: str,
    all_memories: list[dict],
    workspace_id: str | None = None,
) -> list[str]:
    """Store extracted chapter memories as L1 AI memories."""
    stored_ids = []
    total = len(all_memories)
    for i, mem in enumerate(all_memories):
        summary = mem.get("summary", "")
        if not summary:
            continue
        await set_progress(
            agent_id, "storing_memories",
            current=i + 1, total=total,
            message=f"正在写入记忆 ({i+1}/{total})...",
        )
        memory_id = await store_memory(
            user_id=user_id,
            content=summary,
            summary=summary,
            level=1,
            importance=float(mem.get("importance", 0.85)),
            memory_type=mem.get("type", "life"),
            main_category=mem.get("main_category", "生活"),
            sub_category=mem.get("sub_category", "其他"),
            source="ai",
            workspace_id=workspace_id,
        )
        if memory_id:
            stored_ids.append(memory_id)
    return stored_ids


async def generate_full_life_story(
    agent_id: str,
    user_id: str,
    name: str,
    gender: str | None,
    personality: dict | None,
    seven_dim: dict | None,
    workspace_id: str | None = None,
) -> None:
    """Main entry point: select profile → outline → extract memories → store L1.

    Updates Redis progress throughout. Sets agent status to 'active' when done.
    """
    try:
        # Step 1: Select matching CharacterProfile
        await set_progress(agent_id, "selecting_profile", message="正在匹配角色背景...")
        profile = await select_character_profile(gender)
        if not profile:
            logger.warning(f"No matching CharacterProfile for agent {agent_id}, skipping life story")
            await set_progress(agent_id, "complete", message="无可用背景模板，已跳过")
            await _activate_agent(agent_id)
            return

        # Bind profile to agent + 用 profile 数据覆盖 agent 的 occupation/age/city
        # 防止 _init_agent_background 的 LLM 自由生成与 profile 冲突
        profile_data = profile["data"]
        identity = profile_data.get("identity", {})
        career = profile_data.get("career", {})
        update_data: dict = {"characterProfileId": profile["id"]}
        if career.get("title"):
            update_data["occupation"] = career["title"]
        if identity.get("location"):
            update_data["city"] = str(identity["location"])
        if identity.get("age"):
            update_data["age"] = int(identity["age"])
        await db.aiagent.update(where={"id": agent_id}, data=update_data)

        # Step 2: Generate outline
        await set_progress(agent_id, "generating_outline", message="正在构思人生大纲...")
        chapters = await generate_outline(profile_data, name, gender, personality, seven_dim)
        if not chapters:
            logger.warning(f"Outline generation returned empty for agent {agent_id}")
            await set_progress(agent_id, "complete", message="大纲生成失败，已跳过")
            await _activate_agent(agent_id)
            return

        total = len(chapters)
        outline_summary = "\n".join(
            f"{i+1}. {ch.get('title', '')} ({ch.get('age_range', '')}): {', '.join(ch.get('key_events', []))}"
            for i, ch in enumerate(chapters)
        )

        # Step 3: Extract structured memories from each chapter
        all_memories: list[dict] = []
        for i, chapter in enumerate(chapters):
            await set_progress(
                agent_id, "generating_chapter",
                current=i + 1, total=total,
                message=f"正在提取「{chapter.get('title', '')}」的记忆 ({i+1}/{total})...",
            )
            try:
                memories = await extract_chapter_memories(
                    chapter, outline_summary, name, i, total,
                )
                all_memories.extend(memories)
            except Exception as e:
                logger.warning(f"Chapter {i+1} memory extraction failed: {e}")

        # Step 4: Store as L1 memories (progress updated inside store function)
        stored_ids = await store_chapter_memories(
            agent_id, user_id, all_memories,
            workspace_id=workspace_id,
        )

        logger.info(
            f"Life story complete for agent {agent_id}: "
            f"{len(stored_ids)} memories stored ({len(all_memories)} extracted)"
        )

        # Step 5: Mark complete
        await set_progress(
            agent_id, "complete",
            message=f"人生经历生成完成: {len(stored_ids)} 条记忆",
        )
        await _activate_agent(agent_id)

    except Exception as e:
        logger.error(f"Life story generation failed for agent {agent_id}: {e}", exc_info=True)
        await set_progress(agent_id, "failed", message=f"生成失败: {str(e)[:200]}")
        # Still activate agent so user isn't permanently stuck
        await _activate_agent(agent_id)


async def _activate_agent(agent_id: str) -> None:
    """Set agent status to active."""
    try:
        await db.aiagent.update(
            where={"id": agent_id},
            data={"status": "active"},
        )
    except Exception as e:
        logger.error(f"Failed to activate agent {agent_id}: {e}")
