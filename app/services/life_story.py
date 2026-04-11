"""AI 万字人生经历生成服务。

流程:
  1. 选择匹配的 CharacterProfile (性别匹配, published 状态)
  2. 生成人生大纲 (12-15 个章节标题+关键事件)
  3. 逐章扩写 (~800 字/章, 传入大纲+前章摘要保证连贯)
  4. 写入 AI L1 记忆
  5. 通过 Redis 实时更新进度, 供前端轮询
"""

from __future__ import annotations

import json
import logging
import random

from app.db import db
from app.redis_client import get_redis
from app.services.llm.models import get_chat_model, invoke_json, invoke_text
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


async def expand_chapter(
    chapter: dict,
    outline_summary: str,
    profile_data: dict,
    name: str,
    prev_summary: str | None,
    chapter_index: int,
    total_chapters: int,
) -> str:
    """Expand a single chapter outline into ~800 chars of narrative."""
    prompt = f"""你是一个人物传记作家，正在为角色「{name}」撰写人生经历的第 {chapter_index + 1}/{total_chapters} 章。

完整大纲：
{outline_summary}

{"上一章摘要：" + prev_summary if prev_summary else "这是第一章。"}

当前章节：
- 标题: {chapter.get('title', '')}
- 年龄阶段: {chapter.get('age_range', '')}
- 关键事件: {json.dumps(chapter.get('key_events', []), ensure_ascii=False)}

要求：
1. 以第一人称「我」叙述，语气自然真实，像在跟朋友讲述自己的经历
2. 篇幅控制在600-1000字
3. 要有具体的场景描写、情感体验、心理变化
4. 与上一章自然衔接，与大纲逻辑一致
5. 不要写标题或章节号，直接开始叙述
6. 不要使用"我是一个AI"等打破沉浸感的表述"""

    model = get_chat_model()
    text = await invoke_text(model, prompt)
    return text.strip()


def _resolve_category(chapter: dict) -> tuple[str, str]:
    """Use LLM-assigned main_category + sub_category from outline, validated against TAXONOMY."""
    main = chapter.get("main_category", "")
    sub = chapter.get("sub_category", "")
    # Validate against TAXONOMY
    if main in TAXONOMY and sub in TAXONOMY[main]:
        return main, sub
    # main valid but sub invalid → fallback to "其他"
    if main in TAXONOMY:
        return main, "其他"
    # Both invalid → fallback
    return "生活", "其他"


async def store_chapters_as_memories(
    agent_id: str,
    user_id: str,
    chapters: list[dict],
    expanded_texts: list[str],
    workspace_id: str | None = None,
) -> list[str]:
    """Store expanded chapters as L1 AI memories, updating progress per item."""
    stored_ids = []
    total = len(chapters)
    for i, (chapter, text) in enumerate(zip(chapters, expanded_texts)):
        if not text:
            continue
        await set_progress(
            agent_id, "storing_memories",
            current=i + 1, total=total,
            message=f"正在写入记忆 ({i+1}/{total})...",
        )
        main_cat, sub_cat = _resolve_category(chapter)
        title = chapter.get("title", f"人生经历第{i+1}章")
        summary = f"{title} ({chapter.get('age_range', '')})"

        memory_id = await store_memory(
            user_id=user_id,
            content=text,
            summary=summary,
            level=1,
            importance=0.95,
            memory_type="life",
            main_category=main_cat,
            sub_category=sub_cat,
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
    """Main entry point: select profile → generate outline → expand chapters → store L1 memories.

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

        # Step 3: Expand each chapter
        expanded: list[str] = []
        prev_summary: str | None = None
        for i, chapter in enumerate(chapters):
            await set_progress(
                agent_id, "generating_chapter",
                current=i + 1, total=total,
                message=f"正在撰写「{chapter.get('title', '')}」({i+1}/{total})...",
            )
            try:
                text = await expand_chapter(
                    chapter, outline_summary, profile_data,
                    name, prev_summary, i, total,
                )
                expanded.append(text)
                # Use first 100 chars as summary for next chapter's context
                prev_summary = text[:100] + "..." if len(text) > 100 else text
            except Exception as e:
                logger.warning(f"Chapter {i+1} expansion failed: {e}")
                expanded.append("")

        # Step 4: Store as L1 memories (progress updated inside store function)
        valid_chapters = [(ch, txt) for ch, txt in zip(chapters, expanded) if txt]
        stored_ids = await store_chapters_as_memories(
            agent_id, user_id,
            [ch for ch, _ in valid_chapters],
            [txt for _, txt in valid_chapters],
            workspace_id=workspace_id,
        )

        total_chars = sum(len(t) for t in expanded if t)
        logger.info(
            f"Life story complete for agent {agent_id}: "
            f"{len(stored_ids)} memories, {total_chars} chars"
        )

        # Step 5: Mark complete
        await set_progress(
            agent_id, "complete",
            message=f"人生经历生成完成: {total_chars} 字, {len(stored_ids)} 段记忆",
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
