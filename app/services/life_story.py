"""AI 人生经历生成服务。

流程:
  1. 选择匹配的 CharacterProfile (性别匹配, published 状态)
  2. Phase 1: 将 Profile 结构化字段直接转换为 L1 记忆 (无需LLM)
  3. Phase 2: 基于完整 Profile + 性格生成人生大纲 → 逐章提取经历型记忆
  4. 通过 Redis 实时更新进度, 供前端轮询
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid

from app.db import db, ensure_connected
from app.redis_client import get_redis
from app.services.llm.models import get_chat_model, get_embedding_model, invoke_json
from app.services.memory.storage import normalize_memory_type
from app.services.memory.taxonomy import TAXONOMY_MATRIX, resolve_taxonomy
from app.services.memory.vector_search import format_vector
from app.services.runtime.cache import bump_cache_version
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

# Redis key for provisioning progress
PROGRESS_KEY_PREFIX = "provision_progress:"
PROGRESS_TTL = 3600  # 1 hour

# 章节并发上限：避免对 LLM provider 的 burst，但充分并行
_CHAPTER_CONCURRENCY = 8

# 进度刷新粒度：每完成 N 章（或最后一章）才写一次 Redis，避免锁/RTT 抖动
_PROGRESS_EVERY = 3


def _taxonomy_description() -> str:
    """格式化 AI L1 taxonomy 给 LLM 看。

    Life-story 生成的全部都是 AI-source L1 记忆，所以这里只罗列 ai/L1 集合
    （比 user/L1 多一个生活/交互），避免 LLM 输出 L2/L3 才允许的"变化"
    /"闲聊"等 sub-cat 又被 resolve 时强行归到"其他"。
    """
    lines = []
    for main_cat, sub_cats in TAXONOMY_MATRIX["ai"][1].items():
        subs = ", ".join(sub_cats)
        lines.append(f"  {main_cat}: [{subs}]")
    return "\n".join(lines)


async def set_progress(agent_id: str, stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
    """Update provisioning progress in Redis."""
    redis = await get_redis()
    percent = 0
    if stage == "selecting_profile":
        percent = 2
    elif stage == "converting_profile":
        percent = 5
    elif stage == "generating_outline":
        percent = 10
    elif stage == "generating_chapter":
        percent = 12 + int(75 * current / max(total, 1))
    elif stage == "storing_memories":
        percent = 90 + int(8 * current / max(total, 1))
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
    """Select a matching published CharacterProfile by gender."""
    profiles = await db.characterprofile.find_many(
        where={"status": "published"},
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
        matching = profiles
    if not matching:
        return None

    selected = random.choice(matching)
    return {
        "id": selected.id,
        "data": selected.data if isinstance(selected.data, dict) else {},
        "career_template": {
            "title": selected.career.title,
            "duties": selected.career.duties,
            "outputs": selected.career.outputs,
            "social_value": selected.career.socialValue,
            "clients": selected.career.clients,
        } if selected.career else None,
    }


# ── Phase 1: Profile 结构化字段 → L1 记忆 (无需LLM) ──

# (category_key, field_key, main_category, sub_category, memory_type, importance)
_PROFILE_FIELD_MAP: list[tuple[str, str, str, str, str, float]] = [
    # identity
    ("identity", "gender", "身份", "性别", "identity", 0.95),
    ("identity", "age", "身份", "年龄", "identity", 0.95),
    ("identity", "birthday", "身份", "生日", "identity", 0.90),
    # identity.education 与 education_knowledge.degree 重复, 只保留后者
    ("identity", "location", "身份", "现居地", "identity", 0.90),
    ("identity", "family", "身份", "亲属关系", "identity", 0.90),
    # appearance
    ("appearance", "height", "身份", "外貌特征", "identity", 0.80),
    ("appearance", "weight", "身份", "外貌特征", "identity", 0.75),
    ("appearance", "features", "身份", "外貌特征", "identity", 0.80),
    ("appearance", "style", "身份", "外貌特征", "identity", 0.75),
    ("appearance", "voice", "身份", "外貌特征", "identity", 0.75),
    # education_knowledge
    ("education_knowledge", "degree", "身份", "教育背景", "identity", 0.85),
    ("education_knowledge", "strengths", "生活", "技能", "life", 0.80),
    ("education_knowledge", "self_taught", "生活", "技能", "life", 0.80),
    # values
    ("values", "motto", "思维", "价值观", "thought", 0.85),
    ("values", "believes", "思维", "价值观", "thought", 0.80),
    ("values", "opposes", "思维", "价值观", "thought", 0.80),
    ("values", "goal", "思维", "理想与目标", "thought", 0.85),
    # abilities
    ("abilities", "good_at", "生活", "技能", "life", 0.80),
    ("abilities", "never_do", "偏好", "禁忌/雷区", "preference", 0.85),
    ("abilities", "limits", "思维", "自我认知", "thought", 0.75),
    # fears
    ("fears", "animals", "偏好", "禁忌/雷区", "preference", 0.85),
    ("fears", "objects", "偏好", "禁忌/雷区", "preference", 0.80),
    ("fears", "atmospheres", "偏好", "禁忌/雷区", "preference", 0.80),
    # dislikes
    ("dislikes", "foods", "偏好", "饮食厌恶", "preference", 0.80),
    ("dislikes", "sounds", "偏好", "审美厌恶", "preference", 0.75),
    ("dislikes", "smells", "偏好", "审美厌恶", "preference", 0.75),
    ("dislikes", "habits", "偏好", "人际厌恶", "preference", 0.75),
]

# likes 字段太多, 单独定义映射
_LIKES_FIELD_MAP: list[tuple[str, str, str, float]] = [
    # field_key, sub_category, memory_type, importance
    ("colors", "审美爱好", "preference", 0.70),
    ("foods", "饮食喜好", "preference", 0.80),
    ("fruits", "饮食喜好", "preference", 0.75),
    ("season", "审美爱好", "preference", 0.70),
    ("weather", "审美爱好", "preference", 0.70),
    ("plants", "审美爱好", "preference", 0.70),
    ("animals", "审美爱好", "preference", 0.75),
    ("music", "审美爱好", "preference", 0.75),
    ("songs", "审美爱好", "preference", 0.70),
    ("sounds", "审美爱好", "preference", 0.70),
    ("scents", "审美爱好", "preference", 0.70),
    ("books", "审美爱好", "preference", 0.70),
    ("movies", "审美爱好", "preference", 0.70),
    ("sports", "生活习惯", "preference", 0.75),
    ("quirks", "生活习惯", "preference", 0.75),
]

# (field_key, prefix, main_category, sub_category, memory_type, importance)
_CAREER_FIELD_MAP: list[tuple[str, str, str, str, str, float]] = [
    ("title", "我的职业是", "身份", "职业/与经济", "identity", 0.95),
    ("duties", "我的工作内容: ", "生活", "工作", "life", 0.90),
    ("outputs", "我的主要产出物: ", "生活", "工作", "life", 0.85),
    ("social_value", "我的工作的社会价值: ", "生活", "工作", "life", 0.80),
    ("clients", "我的服务对象: ", "生活", "工作", "life", 0.80),
]

# 字段名 → 中文描述 (用于生成 "我..." 句式)
_FIELD_LABELS: dict[str, str] = {
    "gender": "性别是",
    "age": "今年",
    "birthday": "生日是",
    "location": "住在",
    "family": "家庭情况:",
    "height": "身高",
    "weight": "体型",
    "features": "外貌特征:",
    "style": "穿搭风格:",
    "voice": "声音特点:",
    "degree": "学历:",
    "strengths": "擅长的知识领域:",
    "self_taught": "自学过的技能:",
    "motto": "人生信条:",
    "believes": "相信",
    "opposes": "反对",
    "goal": "人生目标:",
    "good_at": "擅长",
    "never_do": "绝对不会做的事:",
    "limits": "能力上限:",
    "colors": "喜欢的颜色:",
    "foods": "喜欢吃",
    "fruits": "喜欢的水果:",
    "season": "喜欢的季节:",
    "weather": "喜欢的天气:",
    "plants": "喜欢的植物:",
    "animals": "喜欢的动物:",
    "music": "喜欢的音乐:",
    "songs": "喜欢的歌曲:",
    "sounds": "喜欢的声音:",
    "scents": "喜欢的气味:",
    "books": "喜欢看的书:",
    "movies": "喜欢的电影:",
    "sports": "喜欢的运动:",
    "quirks": "小癖好:",
}

_DISLIKE_LABELS: dict[str, str] = {
    "foods": "讨厌吃",
    "sounds": "讨厌的声音:",
    "smells": "讨厌的气味:",
    "habits": "讨厌的习惯:",
}

_FEAR_LABELS: dict[str, str] = {
    "animals": "害怕的动物:",
    "objects": "害怕的东西:",
    "atmospheres": "害怕的氛围:",
}


def _format_value(value: object) -> str | None:
    """将字段值格式化为字符串, 过滤空值。"""
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(v).strip() for v in value if v]
        return "、".join(items) if items else None
    s = str(value).strip()
    return s if s else None


def convert_profile_to_memories(profile_data: dict, career_template: dict | None) -> list[dict]:
    """将 CharacterProfile 结构化字段直接转换为 L1 记忆列表 (无需LLM)。"""
    memories: list[dict] = []

    # 通用字段
    for cat_key, field_key, main_cat, sub_cat, mem_type, importance in _PROFILE_FIELD_MAP:
        cat_data = profile_data.get(cat_key, {})
        if not isinstance(cat_data, dict):
            continue
        val = _format_value(cat_data.get(field_key))
        if not val:
            continue
        label = _FIELD_LABELS.get(field_key, f"{field_key}:")
        # 特殊处理 dislikes/fears 的标签
        if cat_key == "dislikes":
            label = _DISLIKE_LABELS.get(field_key, f"讨厌{field_key}:")
        elif cat_key == "fears":
            label = _FEAR_LABELS.get(field_key, f"害怕{field_key}:")

        # age 特殊: "我今年22岁"
        if field_key == "age":
            summary = f"我今年{val}岁"
        else:
            summary = f"我{label}{val}"

        memories.append({
            "summary": summary,
            "main_category": main_cat,
            "sub_category": sub_cat,
            "type": mem_type,
            "importance": importance,
        })

    # likes 字段 (归属偏好)
    likes = profile_data.get("likes", {})
    if isinstance(likes, dict):
        for field_key, sub_cat, mem_type, importance in _LIKES_FIELD_MAP:
            val = _format_value(likes.get(field_key))
            if not val:
                continue
            label = _FIELD_LABELS.get(field_key, f"喜欢{field_key}:")
            memories.append({
                "summary": f"我{label}{val}",
                "main_category": "偏好",
                "sub_category": sub_cat,
                "type": mem_type,
                "importance": importance,
            })

    # 职业: 优先 career_template (CareerTemplate 表), 回退 profile_data.career
    ct = career_template
    if not ct:
        profile_career = profile_data.get("career", {})
        if isinstance(profile_career, dict) and profile_career.get("title"):
            ct = profile_career

    if ct:
        for key, prefix, main_cat, sub_cat, mem_type, importance in _CAREER_FIELD_MAP:
            val = _format_value(ct.get(key))
            if not val:
                continue
            memories.append({
                "summary": f"{prefix}{val}",
                "main_category": main_cat, "sub_category": sub_cat,
                "type": mem_type, "importance": importance,
            })

    return memories


# ── Phase 2: LLM 生成经历型记忆 ──

def _format_career(career_template: dict | None) -> str:
    """Format career template data for prompt injection."""
    if not career_template:
        return "未知"
    parts = [f"职业: {career_template.get('title', '未知')}"]
    if career_template.get("duties"):
        parts.append(f"工作内容: {career_template['duties']}")
    if career_template.get("outputs"):
        parts.append(f"主要产出物: {career_template['outputs']}")
    if career_template.get("social_value"):
        parts.append(f"社会价值: {career_template['social_value']}")
    if career_template.get("clients"):
        parts.append(f"服务对象: {career_template['clients']}")
    return "\n".join(parts)


def _format_profile_for_prompt(profile_data: dict, career_template: dict | None) -> str:
    """将完整 Profile 数据格式化为 LLM prompt 文本, 不遗漏任何分类。"""
    sections = []

    identity = profile_data.get("identity", {})
    if identity:
        sections.append(f"""基础身份:
- 年龄: {identity.get('age', '未知')}
- 出生日期: {identity.get('birthday', '未知')}
- 城市: {identity.get('location', '未知')}
- 教育: {identity.get('education', '未知')}
- 家庭: {identity.get('family', '未知')}""")

    appearance = profile_data.get("appearance", {})
    if appearance:
        labels = {"height": "身高", "weight": "体型", "features": "外貌特征",
                  "style": "穿搭风格", "voice": "声音特点"}
        parts = [f"- {labels[k]}: {v}" for k in labels if (v := appearance.get(k))]
        if parts:
            sections.append("外貌与形象:\n" + "\n".join(parts))

    edu = profile_data.get("education_knowledge", {})
    if edu:
        parts = []
        if edu.get("degree"):
            parts.append(f"- 学历: {edu['degree']}")
        if edu.get("strengths"):
            parts.append(f"- 知识擅长: {json.dumps(edu['strengths'], ensure_ascii=False)}")
        if edu.get("self_taught"):
            parts.append(f"- 自学技能: {json.dumps(edu['self_taught'], ensure_ascii=False)}")
        if parts:
            sections.append("教育与知识:\n" + "\n".join(parts))

    sections.append(f"职业详情:\n{_format_career(career_template)}")

    likes = profile_data.get("likes", {})
    if likes:
        sections.append(f"喜好: {json.dumps(likes, ensure_ascii=False)}")

    dislikes = profile_data.get("dislikes", {})
    if dislikes:
        sections.append(f"讨厌: {json.dumps(dislikes, ensure_ascii=False)}")

    fears = profile_data.get("fears", {})
    if fears:
        sections.append(f"害怕: {json.dumps(fears, ensure_ascii=False)}")

    values = profile_data.get("values", {})
    if values:
        parts = []
        if values.get("motto"):
            parts.append(f"- 人生信条: {values['motto']}")
        if values.get("believes"):
            parts.append(f"- 相信: {json.dumps(values['believes'], ensure_ascii=False)}")
        if values.get("opposes"):
            parts.append(f"- 反对: {json.dumps(values['opposes'], ensure_ascii=False)}")
        if values.get("goal"):
            parts.append(f"- 人生目标: {values['goal']}")
        if parts:
            sections.append("价值观:\n" + "\n".join(parts))

    abilities = profile_data.get("abilities", {})
    if abilities:
        parts = []
        if abilities.get("good_at"):
            parts.append(f"- 擅长: {json.dumps(abilities['good_at'], ensure_ascii=False)}")
        if abilities.get("never_do"):
            parts.append(f"- 绝不做: {json.dumps(abilities['never_do'], ensure_ascii=False)}")
        if abilities.get("limits"):
            parts.append(f"- 能力上限: {json.dumps(abilities['limits'], ensure_ascii=False)}")
        if parts:
            sections.append("能力与边界:\n" + "\n".join(parts))

    return "\n\n".join(sections)


async def generate_outline(
    profile_data: dict,
    name: str,
    gender: str | None,
    seven_dim: dict | None,
    career_template: dict | None = None,
) -> list[dict]:
    """Generate life story outline: 12-15 chapters with titles and key events."""
    profile_text = _format_profile_for_prompt(profile_data, career_template)

    prompt = f"""你是一个人物传记作家。请为以下角色生成一份人生经历大纲。

角色: {name} ({gender or '未设定'})
性格特点: {json.dumps(seven_dim, ensure_ascii=False) if seven_dim else '未设定'}

{profile_text}

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
- 事件要与角色的职业、性格、喜好、害怕的事物、价值观等逻辑自洽
- 以第一人称视角（"我"）
- 包含成长转折点、关键人物、情感体验
- 12-15个章节应覆盖尽可能多的不同分类（身份/偏好/生活/情绪/思维），而非集中在某一类
- 注意: 基本事实（身份/喜好/职业等）已通过其他方式记录，这里重点生成「经历和故事」

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
    seven_dim: dict | None,
    chapter_index: int,
    total_chapters: int,
    profile_text: str,
) -> list[dict]:
    """Extract 5-10 experiential memories from a single chapter."""
    prompt = f"""你正在为角色「{name}」梳理人生经历第 {chapter_index + 1}/{total_chapters} 阶段的记忆。

性格特点: {json.dumps(seven_dim, ensure_ascii=False) if seven_dim else '未设定'}

角色完整背景:
{profile_text}

完整人生大纲：
{outline_summary}

当前阶段：
- 标题: {chapter.get('title', '')}
- 年龄阶段: {chapter.get('age_range', '')}
- 关键事件: {json.dumps(chapter.get('key_events', []), ensure_ascii=False)}

请为这个阶段生成 5-10 条经历型记忆。每条记忆描述一个具体的事件/经历/感受/决定。

⚠️ 注意: 角色的基本事实（身份、喜好、职业、外貌等）已另外存储，请不要重复生成这些事实性记忆。
这里只生成「经历和故事」——发生了什么事、有什么感受、做了什么决定。

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
1. 每条记忆只描述一个具体经历，如"我5岁时在乡下第一次见到蛇，吓得哭了一下午"
2. 不要写成大段叙事，每条 summary 控制在 20-80 字
3. 以第一人称「我」开头
4. 经历要与角色的性格、喜好、职业、害怕的事物等逻辑自洽
5. importance 范围: 重大转折 0.9-0.95, 重要经历 0.8-0.9, 一般经历 0.7-0.85
6. sub_category 必须从分类表中原样复制，禁止自造"""

    model = get_chat_model()
    result = await invoke_json(model, prompt)
    if isinstance(result, dict) and "memories" in result:
        return result["memories"]
    if isinstance(result, list):
        return result
    return []


async def store_memories_batch(
    agent_id: str,
    user_id: str,
    all_memories: list[dict],
    workspace_id: str | None = None,
) -> list[str]:
    """Store memories with batch optimizations for initial provisioning.

    Optimizations vs per-item store_memory():
    - Batch embedding: aembed_documents() instead of N × aembed_query()
    - Dedup guard: refuse to run if the target workspace already has AI
      memories (re-running provisioning would otherwise duplicate records)
    - Bulk insert: client-side UUIDs + create_many for memories/changelog,
      one multi-VALUES SQL for embeddings
    - Best-effort rollback: if the embedding insert fails we delete the
      freshly-inserted memory rows so retrieval never sees a half-state
    """
    # Filter empty summaries
    valid = [m for m in all_memories if m.get("summary")]
    if not valid:
        return []

    total = len(valid)

    # Resolve workspace once (shared by all memories + changelogs)
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)

    # Dedup guard: bulk path intentionally skips per-row similarity checks
    # (it's hot, and on first provisioning DB is empty). But re-running
    # provisioning for a non-empty workspace would silently duplicate every
    # profile memory. Refuse rather than corrupt.
    existing = await db.aimemory.count(
        where={"userId": user_id, "workspaceId": workspace_id}
    )
    if existing > 0:
        logger.warning(
            f"store_memories_batch skipped for agent {agent_id}: workspace "
            f"already has {existing} AI memories (re-run would duplicate)"
        )
        return []

    texts = [m["summary"] for m in valid]

    await set_progress(agent_id, "storing_memories", current=0, total=total,
                       message=f"正在生成向量 ({total} 条)...")
    embed_model = get_embedding_model()
    embeddings = await embed_model.aembed_documents(texts)

    # Refresh DB connection (may have gone stale during embedding)
    await ensure_connected()

    # Pre-generate IDs so we can bulk-insert memories AND link embeddings
    # without a per-row RETURNING round-trip.
    ids = [str(uuid.uuid4()) for _ in valid]

    # Resolve taxonomy + normalize types in pure Python (no I/O)
    memory_rows: list[dict] = []
    changelog_rows: list[dict] = []
    for mid, mem in zip(ids, valid):
        summary = mem["summary"]
        mem_type = normalize_memory_type(mem.get("type", "life"))
        # Provisioning always writes AI-source L1 memories.
        taxonomy = resolve_taxonomy(
            main_category=mem.get("main_category", "生活"),
            sub_category=mem.get("sub_category", "其他"),
            legacy_type=mem_type,
            source="ai",
            level=1,
        )
        memory_rows.append({
            "id": mid,
            "userId": user_id,
            "content": summary,
            "summary": summary,
            "level": 1,
            "importance": float(mem.get("importance", 0.85)),
            "type": mem_type,
            "mainCategory": taxonomy.main_category,
            "subCategory": taxonomy.sub_category,
            "workspaceId": workspace_id,
        })
        changelog_rows.append({
            "userId": user_id,
            "memoryId": mid,
            "operation": "insert",
            "newValue": summary,
            "workspaceId": workspace_id,
        })

    # ── Bulk insert memories ──
    await set_progress(agent_id, "storing_memories",
                       current=int(total * 0.3), total=total,
                       message=f"正在写入记忆 ({total} 条)...")
    await db.aimemory.create_many(data=memory_rows)

    # ── Bulk insert embeddings: single multi-VALUES SQL ──
    # pgvector lacks Prisma binding, so we hand-build the placeholder list:
    # ($1,$2::extensions.vector),($3,$4::extensions.vector),...
    # Fresh client-side UUIDs guarantee no collisions, so no ON CONFLICT.
    await set_progress(agent_id, "storing_memories",
                       current=int(total * 0.6), total=total,
                       message=f"正在写入向量 ({total} 条)...")
    placeholders = ",".join(
        f"(${i * 2 + 1},${i * 2 + 2}::extensions.vector)" for i in range(total)
    )
    args: list = []
    for mid, emb in zip(ids, embeddings):
        args.append(mid)
        args.append(format_vector(emb))
    try:
        await db.execute_raw(
            f"INSERT INTO memory_embeddings (memory_id, embedding) VALUES {placeholders}",
            *args,
        )
    except Exception:
        # Embedding write failed — roll back the memories we just inserted
        # so vector search never sees records without vectors.
        logger.error(
            f"Embedding batch insert failed for agent {agent_id}; "
            f"rolling back {len(ids)} orphan memory rows"
        )
        try:
            await db.aimemory.delete_many(where={"id": {"in": ids}})
        except Exception as cleanup_err:
            logger.error(f"Rollback failed for agent {agent_id}: {cleanup_err}")
        raise

    # ── Bulk insert changelog (advisory, never abort for it) ──
    try:
        await db.memorychangelog.create_many(data=changelog_rows)
    except Exception as e:
        logger.warning(f"Bulk changelog insert failed for agent {agent_id}: {e}")

    # Invalidate any stale per-user retrieval/graph caches so the first
    # message after provisioning actually sees these memories.
    try:
        await bump_cache_version(user_id, workspace_id)
    except Exception as e:
        logger.debug(f"cache bump failed for {user_id}: {e}")

    await set_progress(agent_id, "storing_memories",
                       current=total, total=total,
                       message=f"已写入 {total} 条记忆")
    return ids


# ── Main Entry Point ──

async def prepare_profile_for_agent(
    agent_id: str,
    gender: str | None,
) -> tuple[dict, dict] | None:
    """Step A: 选 profile + 写回 agent 的 age/occupation/city/profileId。

    返回 (profile, applied_updates)，或 None (无可用模板)。applied_updates 是
    本次写入 agent 表的字段子集，调用方可直接把它 patch 到本地 agent 对象上，
    免去一次 find_unique 往返。
    """
    await set_progress(agent_id, "selecting_profile", message="正在匹配角色背景...")
    profile = await select_character_profile(gender)
    if not profile:
        logger.warning(f"No matching CharacterProfile for agent {agent_id}")
        return None

    profile_data = profile["data"]
    career_template = profile.get("career_template")
    identity = profile_data.get("identity", {})

    update_data: dict = {"characterProfileId": profile["id"]}
    if career_template and career_template.get("title"):
        update_data["occupation"] = career_template["title"]
    if identity.get("location"):
        update_data["city"] = str(identity["location"])
    if identity.get("age"):
        update_data["age"] = int(identity["age"])
    await db.aiagent.update(where={"id": agent_id}, data=update_data)
    return profile, update_data


async def generate_life_story_memories(
    agent_id: str,
    user_id: str,
    name: str,
    gender: str | None,
    seven_dim: dict | None,
    profile: dict,
    workspace_id: str | None = None,
) -> int:
    """Step B: 基于已选定的 profile 生成 + 存储所有 L1 记忆。

    依赖 prepare_profile_for_agent 已返回的 profile dict。可与其他依赖 agent
    身份字段的任务（如 life_overview）并行执行。

    返回写入的记忆条数。失败时抛出，由调用方统一处理。
    """
    profile_data = profile["data"]
    career_template = profile.get("career_template")

    # ── Phase 1: 结构化字段直转 (无 LLM, 极快) ──
    await set_progress(agent_id, "converting_profile", message="正在转换背景信息为记忆...")
    profile_memories = convert_profile_to_memories(profile_data, career_template)
    profile_memories.insert(0, {
        "summary": f"我叫{name}",
        "main_category": "身份", "sub_category": "姓名",
        "type": "identity", "importance": 1.0,
    })
    logger.info(f"Profile → {len(profile_memories)} direct memories for agent {agent_id}")

    # ── Phase 2: LLM 生成大纲 + 并发提取章节经历 ──
    await set_progress(agent_id, "generating_outline", message="正在构思人生大纲...")
    chapters = await generate_outline(
        profile_data, name, gender, seven_dim,
        career_template=career_template,
    )

    experience_memories: list[dict] = []
    if chapters:
        total = len(chapters)
        # 大纲 summary 全程共享给每一章，保证章节间逻辑/时间线连续不冲突
        outline_summary = "\n".join(
            f"{i+1}. {ch.get('title', '')} ({ch.get('age_range', '')}): "
            f"{', '.join(ch.get('key_events', []))}"
            for i, ch in enumerate(chapters)
        )
        profile_text = _format_profile_for_prompt(profile_data, career_template)

        # 全并发：所有章节同时发起，以 Semaphore 限流避免 burst
        sem = asyncio.Semaphore(_CHAPTER_CONCURRENCY)
        # 单线程 asyncio 协作调度下 += 不需要锁。进度是 advisory，
        # 仅每完成 _PROGRESS_EVERY 章或最后一章时刷一次 Redis。
        completed = 0
        await set_progress(
            agent_id, "generating_chapter",
            current=0, total=total,
            message=f"正在并发生成 {total} 段经历...",
        )

        async def _extract_one(idx: int, ch: dict) -> list[dict]:
            nonlocal completed
            async with sem:
                try:
                    mems = await extract_chapter_memories(
                        ch, outline_summary, name, seven_dim,
                        idx, total, profile_text,
                    )
                except Exception as e:
                    logger.warning(f"Chapter {idx+1} memory extraction failed: {e}")
                    mems = []
            completed += 1
            if completed % _PROGRESS_EVERY == 0 or completed == total:
                await set_progress(
                    agent_id, "generating_chapter",
                    current=completed, total=total,
                    message=f"已完成 {completed}/{total} 段经历",
                )
            return mems

        results = await asyncio.gather(
            *[_extract_one(i, ch) for i, ch in enumerate(chapters)]
        )
        for mems in results:
            experience_memories.extend(mems)
    else:
        logger.warning(f"Outline generation returned empty for agent {agent_id}")

    # ── Phase 3: 批量写库 ──
    all_memories = profile_memories + experience_memories
    stored_ids = await store_memories_batch(
        agent_id, user_id, all_memories,
        workspace_id=workspace_id,
    )

    logger.info(
        f"Life story memories stored for agent {agent_id}: "
        f"{len(stored_ids)} ({len(profile_memories)} profile + {len(experience_memories)} experience)"
    )
    return len(stored_ids)


async def generate_full_life_story(
    agent_id: str,
    user_id: str,
    name: str,
    gender: str | None,
    seven_dim: dict | None,
    workspace_id: str | None = None,
) -> None:
    """Backward-compat 包装：select profile → direct convert → LLM experiences → store L1。

    新代码应优先使用 prepare_profile_for_agent + generate_life_story_memories
    以便让 life_overview 等下游任务并行启动。本函数保证总会写 progress 并把
    agent 置为 active（即使中途失败）。
    """
    try:
        result = await prepare_profile_for_agent(agent_id, gender)
        if not result:
            await set_progress(agent_id, "complete", message="无可用背景模板，已跳过")
            await activate_agent(agent_id)
            return

        profile, _ = result
        count = await generate_life_story_memories(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
            gender=gender,
            seven_dim=seven_dim,
            profile=profile,
            workspace_id=workspace_id,
        )
        await set_progress(
            agent_id, "complete",
            message=f"生成完成: {count} 条记忆",
        )
        await activate_agent(agent_id)

    except Exception as e:
        logger.error(f"Life story generation failed for agent {agent_id}: {e}", exc_info=True)
        await set_progress(agent_id, "failed", message=f"生成失败: {str(e)[:200]}")
        await activate_agent(agent_id)


async def activate_agent(agent_id: str) -> None:
    """Set agent status to active."""
    try:
        await db.aiagent.update(
            where={"id": agent_id},
            data={"status": "active"},
        )
    except Exception as e:
        logger.error(f"Failed to activate agent {agent_id}: {e}")
