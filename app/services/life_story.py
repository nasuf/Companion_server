"""AI 人生经历生成服务。

流程:
  1. 选择匹配的 CharacterProfile (性别匹配, published 状态)
  2. Phase 1: 将 Profile 结构化字段直接转换为 L1 记忆 (无需LLM)
  3. Phase 2: 基于完整 Profile + 性格生成人生大纲 → 逐章提取经历型记忆
  4. 通过 Redis 实时更新进度, 供前端轮询
"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from datetime import UTC, datetime, timedelta

from collections import Counter

from app.db import db, ensure_connected
from app.redis_client import get_redis
from app.services.llm.models import get_embedding_model, get_utility_model, invoke_json
from app.services.prompting.store import get_prompt_text
from app.services.memory.config import level_for_importance
from app.services.memory.demographics import (
    derive_constellation,
    derive_zodiac,
    sample_ethnicity,
)
from app.services.memory.generation_lock import (
    MemoryGenerationLocked,
    memory_generation_lock,
)
from app.services.memory.init_report import (
    InitReport,
    init_report,
    phase_timer,
)
from app.services.memory.normalization import cosine_similarity
from app.services.memory.storage.persistence import normalize_memory_type
from app.services.memory.taxonomy import (
    L1_CONDITIONAL_SUBS,
    L1_COVERAGE_EXEMPT,
    TAXONOMY_MATRIX,
    analyze_conditional_subs,
    as_dict,
    l1_target_count,
    resolve_taxonomy,
)
from app.services.memory.retrieval.vector_search import format_vector
from app.services.runtime.cache import bump_cache_version
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

PROGRESS_KEY_PREFIX = "provision_progress:"
PROGRESS_TTL = 3600

# Cosine similarity threshold for intra-(main, sub) memory dedup.
_DEDUPE_THRESHOLD = 0.88

# (profile likes field, taxonomy sub-category, 动词式前缀)
# 必须用动词形式直接拼 item, 名词式前缀 (e.g. "喜欢的水果") 拼出来语法错误.
_LIKES_TO_SUB: tuple[tuple[str, str, str], ...] = (
    ("foods", "饮食喜好", "喜欢吃"),
    ("fruits", "饮食喜好", "喜欢吃"),
    ("colors", "审美爱好", "喜欢"),
    ("season", "审美爱好", "喜欢"),
    ("weather", "审美爱好", "喜欢"),
    ("plants", "审美爱好", "喜欢"),
    ("animals", "审美爱好", "喜欢"),
    ("music", "审美爱好", "喜欢听"),
    ("songs", "审美爱好", "喜欢听"),
    ("sounds", "审美爱好", "喜欢"),
    ("scents", "审美爱好", "喜欢"),
    ("books", "审美爱好", "喜欢看"),
    ("movies", "审美爱好", "喜欢看"),
    ("sports", "生活习惯", "喜欢"),
)


# Plan B 9 段进度表 (单 LLM 调用 + 转换 + 入库). LLM 调用期间静止 15%, 前端按
# elapsed time 本地推进到 70%. 旧 stage 名 (selecting_profile / converting_profile /
# storing_memories) 保留兼容: store_memories_batch 内部仍写 storing_memories 用于
# 进度回报. generating_chapter 现已不用, 但保留 entry 以防旧调用路径漏迁移.
_STAGE_PERCENT: dict[str, int] = {
    "initializing": 2,
    "mbti_deriving": 5,
    "mbti_done": 10,
    "prompt_building": 12,
    "llm_generating": 15,
    "llm_done": 72,
    "converting": 78,
    "embedding": 85,
    "storing": 92,
    "first_greeting": 96,
    "complete": 100,
    "failed": -1,
    # 旧名兼容
    "selecting_profile": 2,
    "converting_profile": 78,
}


async def set_progress(agent_id: str, stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
    """Update provisioning progress in Redis. See _STAGE_PERCENT for stage table."""
    if stage in _STAGE_PERCENT:
        percent = _STAGE_PERCENT[stage]
    elif stage == "generating_chapter":
        # 旧路径: percent 随 current/total 线性推进
        percent = 15 + int(55 * current / max(total, 1))
    elif stage == "storing_memories":
        # 内层 store_memories_batch 的 fine-grained 进度: 92 → 100 单调推进,
        # 与 "storing" stage 92 的起点一致, 避免进度条回退
        percent = 92 + int(8 * current / max(total, 1))
    else:
        percent = 0

    redis = await get_redis()
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


# DB / API 一律存英文 male/female; convert_profile_to_memories 写记忆时翻为中文。
_GENDER_EN_TO_ZH = {"male": "男", "female": "女"}

# NOTE: 旧架构 admin 预生 character_profiles → 用户创建时抽取 (select_character_profile
# / prepare_profile_for_agent) 已随 Plan B (spec §1.4 单步 LLM 含 MBTI) 废弃,
# 由 agents.py:_init_and_generate_story 直接调用 character_generation.generate_full_profile.


# ── Phase 1: Profile 结构化字段 → L1 记忆 (无需 LLM) ──


def _clean_text(value: object) -> str | None:
    """Non-list value → trimmed string or None."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _as_list(value: object) -> list[str]:
    """List/tag field → list of non-empty trimmed items. Scalars wrapped as [one]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if _clean_text(v)]
    s = _clean_text(value)
    return [s] if s else []


# Placeholder answers ("兄弟姐妹：无") carry zero semantic value as standalone
# memories and pollute retrieval; drop them for relational/pet list fields.
_PLACEHOLDER_ITEMS = frozenset({"无", "没有", "暂无", "无养", "不养"})


def _substantive_items(value: object) -> list[str]:
    items = []
    for item in _as_list(value):
        if item.strip("。.！!？?，, ") in _PLACEHOLDER_ITEMS:
            continue
        items.append(item)
    return items


# 建号人设从"整份进 L1"改为按事实种类分层 (2026-07).
#
# 原来每个字段都被硬编码成 ≥0.85, 于是整份角色档案都是 L1 —— 永不衰减、永不被
# 淘汰。拿 423 条真实检索配对逐条判"这条记忆对回复有没有用"实测 (仅 AI 侧, 已排
# 除来源混淆):
#
#     L1 · 建号人设   n=279   有用率 11% / 20%
#     L2 · 聊天学到   n= 87   有用率 29% / 28%
#     差 18 个百分点, 置换检验 p=0.0001
#
# 也就是说唯一永不衰减的那一类, 恰好是最派不上用场的一类。而且 importance ≥0.85
# 的有用率 (12%) 反而低于 0.70-0.85 档 (37%) —— 分数在高端与有用性反相关。
#
# 哪些留 L1 不由我另立标准, 直接用代码库已有的"核心身份"定义 L1_SINGLETON_SUBS
# (每个子类只能有一条的那 11 项: 姓名/性别/年龄/生日/现居地…)。它们是会被直接
# 问到、且必须永远答得上的事实。
#
# 其余人设 (偏好 / 思维 / 生活 / 非单值身份) 一律降到 L2: 原值减 0.13, 落在
# 0.72-0.82。**保持原有相对高低** —— 那些 literal 里编着人工判断, 不该被抹平。
# 这个区间的衰减很温和: 从未被访问也要约一年才跌破 0.50, 而任何一次检索都会
# 重置计时。人设不是错的, 只是可能用不上, 慢慢淡出比永久占位合理。
_NON_CORE_IMPORTANCE_DROP = 0.13


def _tiered_importance(main: str, sub: str, importance: float) -> float:
    """核心身份保持原值 (L1); 其余人设降一档到 L2。"""
    from app.services.memory.taxonomy import L1_SINGLETON_SUBS

    if (main, sub) in L1_SINGLETON_SUBS:
        return importance
    return round(max(0.55, importance - _NON_CORE_IMPORTANCE_DROP), 2)


def _add(
    memories: list, content: str, main: str, sub: str, mem_type: str, importance: float,
    *, occur_time: datetime | None = None,
) -> None:
    entry = {
        "content": content,
        "main_category": main,
        "sub_category": sub,
        "type": mem_type,
        "importance": _tiered_importance(main, sub, importance),
    }
    if occur_time is not None:
        entry["occur_time"] = occur_time
    memories.append(entry)


# life_events 10 字段 → taxonomy.生活 子类映射. Plan B 下 prompt 不再生 interaction
# (taxonomy.L1_COVERAGE_EXEMPT 已含 (生活, 交互), 那是 AI↔当前用户的实际聊天历史,
# 由 memory pipeline 累积). 历史 profile 仍带 interaction 时也会被忽略.
_LIFE_EVENT_SUB_MAP: dict[str, str] = {
    "education": "教育",
    "work": "工作",
    "travel": "旅行",
    "living": "居住",
    "health": "健康",
    "pet": "宠物",
    "relationships": "人际",
    "skill_learning": "技能",
    "life": "生活",
    "special": "其他特殊事件",
}

# emotion_events 15 字段 → taxonomy.情绪 子类映射
_EMOTION_EVENT_SUB_MAP: dict[str, str] = {
    "happy": "高兴",
    "sad": "悲伤",
    "angry": "愤怒",
    "fear": "恐惧",
    "disgust": "厌恶",
    "anxiety": "焦虑",
    "disappointment": "失望",
    "pride": "自豪",
    "moved": "感动",
    "embarrassed": "尴尬",
    "regret": "遗憾",
    "lonely": "孤独",
    "surprised": "惊讶",
    "grateful": "感激",
    "relieved": "释怀",
}

# 生活事件按字段类型选合理过往时间区间 (years_ago_min, years_ago_max)
_LIFE_EVENT_TIME_RANGE: dict[str, tuple[float, float]] = {
    "education": (3.0, 12.0),        # 求学阶段, 离当前 3-12 年
    "work": (0.0, 5.0),              # 职业生涯任意一点
    "travel": (0.5, 8.0),            # 旅行
    "living": (0.5, 10.0),           # 搬家/装修
    "health": (1.0, 15.0),           # 健康事件可能很早
    "pet": (1.0, 15.0),              # 宠物记忆
    "relationships": (1.0, 10.0),    # 人际转折
    "skill_learning": (1.0, 12.0),   # 学习关键技能
    "life": (0.5, 12.0),             # 普通生活事件
    "special": (0.5, 15.0),          # 特殊偶然事件
}


def _random_past_time(min_years_ago: float, max_years_ago: float) -> datetime:
    """生成一个 [min, max] 年前的随机时间点 (UTC)."""
    now = datetime.now(UTC)
    span_days = (max_years_ago - min_years_ago) * 365.25
    offset_days = min_years_ago * 365.25 + random.random() * span_days
    return now - timedelta(days=offset_days)


def convert_profile_to_memories(profile_data: dict, career_template: dict | None) -> list[dict]:
    """将 CharacterProfile 结构化字段直接转换为 L1 记忆 (无需 LLM)。

    策略: singleton 字段(姓名/性别/年龄/生日/…) 一条; 列表字段(tags)
    每个元素一条, 保证覆盖度。相同 sub_category 下允许多条。
    """
    memories: list[dict] = []

    identity = as_dict(profile_data, "identity")
    appearance = as_dict(profile_data, "appearance")
    edu = as_dict(profile_data, "education_knowledge")
    values = as_dict(profile_data, "values")
    abilities = as_dict(profile_data, "abilities")
    likes = as_dict(profile_data, "likes")
    dislikes = as_dict(profile_data, "dislikes")
    interpersonal = as_dict(profile_data, "interpersonal")
    lifestyle = as_dict(profile_data, "lifestyle")
    taboo = as_dict(profile_data, "taboo")
    life_events = as_dict(profile_data, "life_events")
    emotion_events = as_dict(profile_data, "emotion_events")
    # 旧 schema 兼容: fears 分类已删除, 仍读取以处理历史 profile_data
    fears = as_dict(profile_data, "fears")

    # ── 身份: singleton 事实 ──
    if (v := _clean_text(identity.get("name"))):
        content = f"我叫{v}"
        # Document imports keep the full naming line (大名/小名/称呼) in
        # name_detail; fold it into the singleton 姓名 memory so aliases
        # survive the admin-form name override.
        detail = _clean_text(identity.get("name_detail"))
        if detail and detail != v:
            content = f"我叫{v}。{detail}"
        _add(memories, content, "身份", "姓名", "identity", 0.95)
    gender = identity.get("gender")
    if isinstance(gender, str):
        gender = _GENDER_EN_TO_ZH.get(gender, gender)
    if (v := _clean_text(gender)):
        _add(memories, f"我性别是{v}", "身份", "性别", "identity", 0.95)
    if (v := _clean_text(identity.get("age"))):
        _add(memories, f"我今年{v}岁", "身份", "年龄", "identity", 0.95)
    if (v := _clean_text(identity.get("birthday"))):
        _add(memories, f"我生日是{v}", "身份", "生日", "identity", 0.90)
    if (v := _clean_text(identity.get("zodiac"))):
        _add(memories, f"我属{v}", "身份", "生肖", "identity", 0.85)
    if (v := _clean_text(identity.get("constellation"))):
        _add(memories, f"我是{v}", "身份", "星座", "identity", 0.85)
    if (v := _clean_text(identity.get("location"))):
        content = f"我现在住在{v}"
        # location_note is the parenthetical context stripped off the address
        # (e.g. why the persona rents there); keep it inside the singleton.
        note = _clean_text(identity.get("location_note"))
        if note:
            content = f"{content}。{note}"
        _add(memories, content, "身份", "现居地", "identity", 0.90)
    if (v := _clean_text(identity.get("birthplace"))):
        _add(memories, f"我出生在{v}", "身份", "出生地", "identity", 0.90)
    if (v := _clean_text(identity.get("growing_up_location"))):
        _add(memories, f"我在{v}长大", "身份", "成长地", "identity", 0.90)
    if (v := _clean_text(identity.get("ethnicity"))):
        _add(memories, f"我是{v}", "身份", "民族", "identity", 0.85)
    if (v := _clean_text(identity.get("blood_type"))):
        _add(memories, f"我是{v}血", "身份", "血型", "identity", 0.85)
    # 亲属关系 / 社会关系 / 宠物：LLM schema 要求每条都是"事实/描述"完整句,
    # 不再加 "家人:" / "我的社会关系:" 等冗余前缀 (前缀已在 sub_category 里,
    # 嵌入和检索都不需要重复, 还会让 AI 复述时带颗粒).
    for item in _substantive_items(identity.get("family")):
        _add(memories, item, "身份", "亲属关系", "identity", 0.90)
    for item in _substantive_items(identity.get("social_relations")):
        _add(memories, item, "身份", "社会关系", "identity", 0.85)
    for item in _substantive_items(identity.get("pet_profile")):
        _add(memories, item, "身份", "宠物", "identity", 0.85)

    # ── 身份: 外貌特征 —— height/weight 是单值 (e.g. "168cm" / "匀称"), 用 "是"
    # 连词组成完整句; features/style/voice 已是完整描述句, 直接入库.
    if (v := _clean_text(appearance.get("height"))):
        _add(memories, f"我的身高是{v}", "身份", "外貌特征", "identity", 0.86)
    if (v := _clean_text(appearance.get("weight"))):
        _add(memories, f"我的体型是{v}", "身份", "外貌特征", "identity", 0.86)
    for key in ("features", "style", "voice"):
        for item in _as_list(appearance.get(key)):
            _add(memories, item, "身份", "外貌特征", "identity", 0.86)

    # ── 身份: 教育背景 (degree 每条已含学历/学校/时间线, 无需前缀) ──
    for item in _as_list(edu.get("degree")):
        _add(memories, item, "身份", "教育背景", "identity", 0.85)

    # ── 生活: 技能 —— 知识擅长 + 自学 每项一条 ──
    for item in _as_list(edu.get("strengths")):
        # Short topic nouns ("心理学") read well as "我擅长X相关的知识"; full
        # descriptions imported from a document ("沟通与倾听技巧：这是...") already
        # stand alone, so the trailing suffix would dangle — keep them verbatim.
        is_topic = len(item) <= 12 and not any(c in item for c in "：:。！？!?")
        content = f"我擅长{item}相关的知识" if is_topic else f"我擅长{item}"
        _add(memories, content, "生活", "技能", "life", 0.87)
    for item in _as_list(edu.get("self_taught")):
        _add(memories, f"我自学过{item}", "生活", "技能", "life", 0.86)
    for item in _as_list(abilities.get("good_at")):
        _add(memories, f"我擅长{item}", "生活", "技能", "life", 0.87)

    # ── 职业: 优先 career_template, 回退 profile_data.career ──
    ct = career_template
    if not ct:
        pc = profile_data.get("career")
        if isinstance(pc, dict) and pc.get("title"):
            ct = pc
    if ct:
        if (v := _clean_text(ct.get("title"))):
            _add(memories, f"我的职业是{v}", "身份", "职业/与经济", "identity", 0.95)
        if (v := _clean_text(ct.get("duties"))):
            _add(memories, f"我的工作是{v}", "生活", "工作", "life", 0.88)
        # clients 是 tag 数组 — 每项一条工作相关记忆
        for client in _as_list(ct.get("clients")):
            _add(memories, f"我的服务对象包括{client}", "生活", "工作", "life", 0.86)
        if (v := _clean_text(ct.get("social_value") or ct.get("socialValue"))):
            _add(memories, f"我做这份工作的意义在于{v}", "生活", "工作", "life", 0.86)
        # 经济状况 was parsed/pooled all along but never converted — the whole
        # income section silently vanished for every creation path. The career
        # pool rows carry no income column, so fall back to profile.career
        # where _apply_postprocess_overrides always writes one.
        income = _clean_text(ct.get("income"))
        if not income:
            pc = profile_data.get("career")
            if isinstance(pc, dict):
                income = _clean_text(pc.get("income"))
        if income:
            _add(memories, f"我的经济状况：{income}", "身份", "职业/与经济", "identity", 0.88)

    for key, sub, prefix in _LIKES_TO_SUB:
        for item in _as_list(likes.get(key)):
            _add(memories, f"我{prefix}{item}", "偏好", sub, "preference", 0.86)
    # 小癖好 (quirks "每条一句", 已是完整描述, 不加前缀).
    for item in _as_list(likes.get("quirks")):
        _add(memories, item, "偏好", "生活习惯", "preference", 0.86)

    # ── 偏好: dislikes 每项一条 ──
    for item in _as_list(dislikes.get("foods")):
        _add(memories, f"我讨厌吃{item}", "偏好", "饮食厌恶", "preference", 0.86)
    for item in _as_list(dislikes.get("sounds")):
        # Skip the categorizing suffix when the item already names a sound
        # ("广场舞的高音喇叭声这种声音" reads broken).
        suffix = "" if item.endswith(("声", "声音", "音", "响")) else "这种声音"
        _add(memories, f"我讨厌{item}{suffix}", "偏好", "审美厌恶", "preference", 0.86)
    for item in _as_list(dislikes.get("smells")):
        suffix = "" if item.endswith(("味", "气味", "味道", "香气", "臭")) else "的气味"
        _add(memories, f"我讨厌{item}{suffix}", "偏好", "审美厌恶", "preference", 0.86)
    for item in _as_list(dislikes.get("habits")):
        _add(memories, f"我讨厌别人{item}", "偏好", "审美厌恶", "preference", 0.86)

    # ── 偏好: 人际偏好 (item 是特质/行为短语, 用动词连接避免 "的人的人" 重叠) ──
    for item in _as_list(interpersonal.get("liked_traits")):
        _add(memories, f"我欣赏{item}", "偏好", "人际喜好", "preference", 0.87)
    for item in _as_list(interpersonal.get("disliked_traits")):
        _add(memories, f"我反感{item}", "偏好", "人际厌恶", "preference", 0.87)

    # ── 偏好: 生活习惯 (routine/hygiene/leisure 已是描述句, 不加前缀) ──
    for key in ("routine", "hygiene", "leisure"):
        for item in _as_list(lifestyle.get(key)):
            _add(memories, item, "偏好", "生活习惯", "preference", 0.88)

    # ── 偏好: 禁忌/雷区 (taboo "不可触碰的底线" 已自带语义) ──
    for item in _as_list(taboo.get("items")):
        _add(memories, item, "偏好", "禁忌/雷区", "preference", 0.93)
    for item in _as_list(abilities.get("never_do")):
        _add(memories, f"我绝对不会{item}", "偏好", "禁忌/雷区", "preference", 0.93)
    # 旧 schema fears 兼容: 历史 profile_data 仍含 fears 分类时, 仍能转记忆
    for item in _as_list(fears.get("animals")):
        _add(memories, f"我害怕{item}", "偏好", "禁忌/雷区", "preference", 0.88)
    for item in _as_list(fears.get("objects")):
        _add(memories, f"我害怕{item}", "偏好", "禁忌/雷区", "preference", 0.86)
    for item in _as_list(fears.get("atmospheres")):
        _add(memories, f"我害怕{item}的氛围", "偏好", "禁忌/雷区", "preference", 0.86)

    # ── 思维: 价值观 / 世界观 / 理想与目标 / 人际关系观 / 社会观点 / 信仰 / 自我认知 ──
    # 所有字段都是 list, schema 要求"每条 X" — 已是完整观点/陈述句, 不重复加
    # 主题前缀 (主题已在 sub_category 里).
    for item in _as_list(values.get("motto")):
        _add(memories, item, "思维", "人生观", "thought", 0.92)
    for item in _as_list(values.get("believes")):
        _add(memories, f"我相信{item}", "思维", "价值观", "thought", 0.90)
    for item in _as_list(values.get("opposes")):
        _add(memories, f"我反对{item}", "思维", "价值观", "thought", 0.90)
    for item in _as_list(values.get("worldview")):
        _add(memories, item, "思维", "世界观", "thought", 0.90)
    for item in _as_list(values.get("goal")):
        _add(memories, item, "思维", "理想与目标", "thought", 0.90)
    for item in _as_list(values.get("interpersonal_view")):
        _add(memories, item, "思维", "人际关系观", "thought", 0.88)
    for item in _as_list(values.get("social_view")):
        _add(memories, item, "思维", "社会观点", "thought", 0.85)
    for item in _as_list(values.get("faith")):
        _add(memories, item, "思维", "信仰/寄托", "thought", 0.90)
    for item in _as_list(abilities.get("limits")):
        # limits are bare capability phrases ("宏观战略思维"); frame them in
        # first person so the stored memory states a self-view, not a topic.
        content = item if item.startswith(("我", "不")) else f"我不太擅长{item}"
        _add(memories, content, "思维", "自我认知", "thought", 0.88)

    # ── 生活: life_events 11 字段, 每条 50-100 字"深远的事/关键节点" + occur_time ──
    # spec §1.4: agent 创建期生成的记忆全部入 L1, 故 importance ≥ 0.85. 比纯偏好
    # 略低 (经历型 vs 身份型), 给 0.85 让 retrieval 时仍能与日常记忆匹配上.
    for field_key, sub in _LIFE_EVENT_SUB_MAP.items():
        scenes = _as_list(life_events.get(field_key))
        time_range = _LIFE_EVENT_TIME_RANGE.get(field_key, (0.5, 10.0))
        for scene in scenes:
            occur = _random_past_time(*time_range)
            _add(memories, scene, "生活", sub, "life", 0.85, occur_time=occur)

    # ── 情绪: emotion_events 15 字段, 每条 50-100 字 + 全生命周期随机 occur_time ──
    try:
        agent_age = int(identity.get("age", 25))
    except (ValueError, TypeError):
        agent_age = 25
    emotion_max_years = max(float(agent_age) - 3.0, 5.0)
    for field_key, sub in _EMOTION_EVENT_SUB_MAP.items():
        scenes = _as_list(emotion_events.get(field_key))
        for scene in scenes:
            occur = _random_past_time(0.5, emotion_max_years)
            _add(memories, scene, "情绪", sub, "emotion", 0.85, occur_time=occur)

    return memories


# ── Phase 2: LLM 按子类补齐覆盖 (taxonomy-driven) ──


# NOTE: 旧架构 phase-3 LLM gap-fill (_build_constraints / _derive_timeline /
# _slice_profile / _digest_existing / _spec_for_gap / _fill_main_gaps +
# _PROFILE_RELEVANCE) 在 Plan B 下已废弃: 单步 character.generation 已通过
# prompt 强制每子类最小条数, 不再需要 main 级补齐。Prompt 调优应该在
# character.generation, 而非加 gap-fill 补丁。


async def _detect_and_resolve_contradictions(memories: list[dict]) -> list[dict]:
    """LLM 扫描全部 L1 记忆找语义矛盾对, drop 较低 importance 那条.

    spec《背景信息》§1.4 要求 agent 创建期 L1 记忆内部一致. character.generation
    单步 prompt 已加跨字段一致性硬约束 (rule 11), 这里是事后兜底:
    - _embed_and_dedupe 只能抓向量近似 (similarity > 0.88) 的复述句, 抓不到
      "养了糯米" vs "没养宠" 这类同主题相反立场 (相似度仅 0.6-0.7).
    - 用 utility model 单次调用扫 80+ 条记忆, ~3K tokens / 5-10s, agent 创建
      总耗时 60-180s 里可接受.

    失败策略: LLM 调用失败、JSON 解析失败、返回空数组 → 跳过, 原样返回.
    不阻塞 agent 创建.
    """
    if len(memories) < 2:
        return memories

    items = [{"id": i, "text": m["content"]} for i, m in enumerate(memories)]
    memory_list_json = json.dumps(items, ensure_ascii=False, indent=2)

    try:
        tpl = await get_prompt_text("memory.pairwise_contradiction")
        prompt = tpl.format(memory_list=memory_list_json)
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning(f"[CONTRA-SCAN] LLM call failed, skipping: {e}")
        return memories

    pairs = result.get("contradictions") if isinstance(result, dict) else None
    if not isinstance(pairs, list) or not pairs:
        return memories

    # Drop the lower-importance side of each pair; on tie drop b. Defer drops by
    # collecting indices first so a single memory caught in two pairs only logs once.
    drop_ids: set[int] = set()
    n = len(memories)
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        a, b = pair.get("a"), pair.get("b")
        if not isinstance(a, int) or not isinstance(b, int):
            continue
        if not (0 <= a < n and 0 <= b < n) or a == b:
            continue
        if a in drop_ids or b in drop_ids:
            continue
        loser = b if memories[a]["importance"] >= memories[b]["importance"] else a
        drop_ids.add(loser)
        logger.info(
            f"[CONTRA-SCAN] drop #{loser} ({memories[loser]['content'][:40]!r}) "
            f"vs #{a if loser == b else b} — {pair.get('reason', '')}"
        )

    if drop_ids:
        logger.info(f"[CONTRA-SCAN] removed {len(drop_ids)} contradictory memories of {n}")
    return [m for i, m in enumerate(memories) if i not in drop_ids]


async def _embed_and_dedupe(memories: list[dict], intra_threshold: float = _DEDUPE_THRESHOLD) -> list[dict]:
    """同一 (main, sub) 内语义相似度 > threshold 合并, 保高 importance。

    Attaches each kept memory's vector as `_embedding` so store_memories_batch
    can reuse it and skip a second embed pass. Embedding failure is fatal —
    silently skipping would double-embed at write time anyway.
    """
    if len(memories) < 2:
        for m in memories:
            m.setdefault("_embedding", None)
        return memories

    model = get_embedding_model()
    vectors = await model.aembed_documents([m["content"] for m in memories])
    for m, v in zip(memories, vectors):
        m["_embedding"] = v

    # Bucket by (main, sub) so comparison is only intra-sub (O(sum k_i²) ≪ O(n²)).
    buckets: dict[tuple[str, str], list[int]] = {}
    for idx, m in enumerate(memories):
        buckets.setdefault((m["main_category"], m["sub_category"]), []).append(idx)

    drop: set[int] = set()
    for indices in buckets.values():
        for a_pos, i in enumerate(indices):
            if i in drop:
                continue
            for j in indices[a_pos + 1:]:
                if j in drop:
                    continue
                if cosine_similarity(vectors[i], vectors[j]) >= intra_threshold:
                    if memories[i]["importance"] >= memories[j]["importance"]:
                        drop.add(j)
                    else:
                        drop.add(i)
                        break
    if drop:
        logger.info(f"dedupe removed {len(drop)} near-duplicate memories")
    return [m for idx, m in enumerate(memories) if idx not in drop]


def _compute_l1_gaps(
    memories: list[dict],
    conditional_include: frozenset[tuple[str, str]] | set[tuple[str, str]] | None = None,
) -> dict[str, dict[str, int]]:
    """按 AI L1 taxonomy 遍历, 返回每个 main 下还缺的子类与缺口数。

    跳过 EXEMPT 子类; CONDITIONAL 子类只在 conditional_include 中明示包含
    或已有 phase-1 记录时才补齐 (避免对不符合角色设定的子类编造内容)。
    """
    have: Counter[tuple[str, str]] = Counter(
        (m["main_category"], m["sub_category"]) for m in memories
    )
    include = conditional_include or frozenset()
    gaps: dict[str, dict[str, int]] = {}
    for main, subs in TAXONOMY_MATRIX["ai"][1].items():
        for sub in subs:
            key = (main, sub)
            if key in L1_COVERAGE_EXEMPT:
                continue
            target = l1_target_count(main, sub)
            if target <= 0:
                continue
            if key in L1_CONDITIONAL_SUBS and have[key] == 0 and key not in include:
                continue
            missing = target - have[key]
            if missing > 0:
                gaps.setdefault(main, {})[sub] = missing
    return gaps


async def store_memories_batch(
    agent_id: str,
    user_id: str,
    all_memories: list[dict],
    workspace_id: str | None = None,
    *,
    force: bool = False,
) -> list[str]:
    """Store memories with batch optimizations for initial provisioning.

    Optimizations vs per-item store_memory():
    - Batch embedding: aembed_documents() instead of N × aembed_query()
    - Dedup guard: refuse to run if the target workspace already has AI
      memories (re-running provisioning would otherwise duplicate records).
      Set `force=True` to clear existing L1 in the workspace first —
      retry / re-generation path uses this.
    - Bulk insert: client-side UUIDs + create_many for memories/changelog,
      one multi-VALUES SQL for embeddings
    - Best-effort rollback: if the embedding insert fails we delete the
      freshly-inserted memory rows so retrieval never sees a half-state
    """
    # Filter empty contents
    valid = [m for m in all_memories if m.get("content")]
    if not valid:
        return []

    total = len(valid)

    # Resolve workspace once (shared by all memories + changelogs)
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)

    # 仅在非 force 路径需要查 existing (force=True 会无条件 delete_many);
    # provisioning 主路径走 force=True, 跳过这次 count 节省一次往返.
    existing = 0
    if not force:
        existing = await db.aimemory.count(
            where={"userId": user_id, "workspaceId": workspace_id}
        )
        if existing > 0:
            # Re-running provisioning without force would silently duplicate;
            # caller must opt into replacement via force=True.
            logger.warning(
                f"store_memories_batch skipped for agent {agent_id}: workspace "
                f"already has {existing} AI memories (re-run would duplicate)"
            )
            return []

    # Embed BEFORE deleting any existing rows — if embedding fails, the
    # original L1 remains intact and caller can retry safely.
    pending_idx = [i for i, m in enumerate(valid) if not m.get("_embedding")]
    if pending_idx:
        await set_progress(agent_id, "storing_memories", current=0, total=total,
                          message=f"正在生成向量 ({len(pending_idx)} 条)...")
        embed_model = get_embedding_model()
        new_vecs = await embed_model.aembed_documents([valid[i]["content"] for i in pending_idx])
        for i, v in zip(pending_idx, new_vecs):
            valid[i]["_embedding"] = v
    embeddings = [m["_embedding"] for m in valid]

    # Refresh DB connection (may have gone stale during embedding)
    await ensure_connected()

    if force:
        # 无条件 delete_many: rows 数没意义, 由 delete_many 返回值取代之前的 existing 计数
        deleted = await db.aimemory.delete_many(
            where={"userId": user_id, "workspaceId": workspace_id}
        )
        if deleted:
            logger.info(
                f"store_memories_batch force=True for agent {agent_id}: cleared "
                f"{deleted} existing AI memories in workspace {workspace_id}"
            )

    # Pre-generate IDs so we can bulk-insert memories AND link embeddings
    # without a per-row RETURNING round-trip.
    ids = [str(uuid.uuid4()) for _ in valid]

    # Resolve taxonomy + normalize types in pure Python (no I/O)
    memory_rows: list[dict] = []
    changelog_rows: list[dict] = []
    for mid, mem in zip(ids, valid):
        content = mem["content"]
        mem_type = normalize_memory_type(mem.get("type", "life"))
        # Provisioning always writes AI-source L1 memories.
        taxonomy = resolve_taxonomy(
            main_category=mem.get("main_category", "生活"),
            sub_category=mem.get("sub_category", "其他"),
            legacy_type=mem_type,
            source="ai",
            level=level_for_importance(float(mem.get("importance", 0.85))),
        )
        # 曾经这里 clamp 到 ≥0.85 并硬写 level=1, 依据是"spec §1.4: 创建期所有
        # 记忆都入 L1"。两件事都改了 (见 _tiered_importance 的实测数据):
        # clamp 会把 LLM 补齐缺口时给出的低分抬上去 —— 模型判断这条只值 0.4 也
        # 照样变成永不衰减的核心记忆, 那份判断被白白丢掉。层级现在由分数推导,
        # 跟录入管线共用同一个换算, 不再各写各的。
        importance = float(mem.get("importance", 0.85))
        row = {
            "id": mid,
            "userId": user_id,
            "content": content,
            "level": level_for_importance(importance),
            "importance": importance,
            "type": mem_type,
            "mainCategory": taxonomy.main_category,
            "subCategory": taxonomy.sub_category,
            "workspaceId": workspace_id,
            # Persona ground truth — write-time reconciliation never mutates
            # profile_seed rows (contradictions go through spec §4 instead).
            "provenance": "profile_seed",
        }
        # Part 5 §3.1: life_events / emotion_events 等过去事件带 occur_time,
        # 让 retrieval 能按时间过滤、L3 awakening 能找到久远记忆.
        if mem.get("occur_time") is not None:
            row["occurTime"] = mem["occur_time"]
        memory_rows.append(row)
        changelog_rows.append({
            "userId": user_id,
            "memoryId": mid,
            "operation": "insert",
            "newValue": content,
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

    # Phase 2-6: seed the entity graph for relation-type persona memories
    # (pet/family/friend names) so entity recall works for the persona, not
    # just chat-time memories. Best-effort — failures never abort provisioning.
    try:
        await _seed_persona_entities(memory_rows, user_id, workspace_id)
    except Exception as e:
        logger.debug(f"persona entity seeding skipped: {e}")

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


# ── Phase 2-6: persona entity seeding (no LLM) ──
#
# Chat-time extraction records entities per memory, but provisioning bulk-insert
# never did — so entity recall was dead for the persona memories that mention
# the most-asked-about names (pet / family / best friend). Derive those names
# deterministically from the relation-type memory texts.

_ENTITY_SUBS: dict[str, tuple[str, str]] = {
    # sub_category → (entity_type, default_role)
    "宠物": ("pet", "pet"),
    "亲属关系": ("person", "family"),
    "社会关系": ("person", "friend"),
}

# Conservative name patterns; a missed name is fine, a wrong name is not.
#
# 教训 (2026-07-20 review): 旧正则把描述性续写误吞成名字 —
#   "我父亲今年退休了" → "今年"     (lookahead 含 退休/今年, {2,3} 回溯捕获时间词)
#   "我妈妈是护士"     → "是护士"   ({2,3} 把系动词"是"吞进名字, $ 命中)
#   "名叫大橘的橘猫很可爱" → "大橘的橘猫很"  ({1,6} 无右边界过度捕获)
# 收紧三处, 仍保留"档案里名字常直接跟谓语"的裸名抽取能力 (父亲林国安是…/母亲陈秀芬年轻时…):
#   ① 名字首字排除代词/助词/系动词/副词 (_NAME_HEAD) → 挡住 "是护士" 被当名字;
#   ② 右边界 (_NAME_TAIL) 只允许 标点/引号/"的"/连接词/句末 + 两个真实续接词
#      "是"/"年轻"; 刻意剔除 "退休/今年/在" → "今年退休"因无合法边界而不匹配;
#   ③ 宠物名限定显式命名词 (名叫/叫做/…) + 可选引号, 右边界收口 → 不再过度捕获.
_NAME_HEAD = r"(?![我你他她它了着过的地得是在有很也都就还没却便更])"
_NAME_TAIL = r"""(?=[”"'‘’，,。、！!？?\s的和跟与]|是|年轻|$)"""
_NAME_BODY = _NAME_HEAD + r"([一-龥A-Za-z]{2,4})" + _NAME_TAIL
_PET_NAME_BODY = _NAME_HEAD + r"([一-龥A-Za-z]{1,4})" + _NAME_TAIL
_PET_INTRO = r"""(?:名叫|叫做|叫作|唤作)[“"'‘’]?"""
_NAME_AFTER_CALLED_RE = re.compile(_PET_INTRO + _PET_NAME_BODY)
_FAMILY_NAME_RE = re.compile(r"(父亲|母亲|爸爸|妈妈)(?:叫|名叫|叫做)?" + _NAME_BODY)
_FRIEND_NAME_RE = re.compile(
    r"(?:闺蜜|好友|好朋友|兄弟|发小)(?:是|叫|名叫)?(?:[一-龥]{0,6}同学)?" + _NAME_BODY
)

_FAMILY_ROLE_MAP = {"父亲": "father", "爸爸": "father", "母亲": "mother", "妈妈": "mother"}


def _extract_persona_entities(content: str, sub_category: str | None) -> list[dict]:
    """Deterministic entity extraction for relation-type persona memories."""
    if sub_category not in _ENTITY_SUBS:
        return []
    entity_type, default_role = _ENTITY_SUBS[sub_category]
    found: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, role: str) -> None:
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            found.append({"name": name, "type": entity_type, "role": role})

    if sub_category == "宠物":
        for m in _NAME_AFTER_CALLED_RE.finditer(content):
            _add(m.group(1), default_role)
    elif sub_category == "亲属关系":
        for m in _FAMILY_NAME_RE.finditer(content):
            _add(m.group(2), _FAMILY_ROLE_MAP.get(m.group(1), default_role))
    elif sub_category == "社会关系":
        for m in _FRIEND_NAME_RE.finditer(content):
            _add(m.group(1), default_role)
    return found[:3]


async def _seed_persona_entities(
    memory_rows: list[dict], user_id: str, workspace_id: str | None,
) -> int:
    """Best-effort entity seeding after the provisioning bulk insert."""
    from app.services.memory.storage.entity_repo import record_entities_for_memory

    linked = 0
    for row in memory_rows:
        entities = _extract_persona_entities(
            row.get("content") or "", row.get("subCategory"),
        )
        if not entities:
            continue
        try:
            linked += await record_entities_for_memory(
                memory_id=row["id"],
                memory_source="ai",
                user_id=user_id,
                workspace_id=workspace_id,
                entities=entities,
            )
        except Exception as e:
            logger.debug(f"persona entity seed failed for {row['id'][:8]}: {e}")
    if linked:
        logger.info(f"persona entity seeding created {linked} edges")
    return linked


# ── Main Entry Point ──

async def generate_l1_coverage(
    agent_id: str,
    user_id: str,
    profile: dict,
    career_template: dict | None,
    workspace_id: str | None = None,
) -> int:
    """Plan B 核心: 把已生成的 profile dict 转成 L1 记忆库 + 入库.

    调用方负责:
    - 通过 character_generation.generate_full_profile 单步 LLM 生成 profile dict
      (含 MBTI / 7 维 / 职业输入, postprocess 兜底必填字段)
    - 通过 career_templates 池随机选 career_template

    本函数职责:
    1. profile JSON → 记忆条目 (convert_profile_to_memories, 字段→sub 1:1 / 1:N 映射)
    2. 缺口验证 (_compute_l1_gaps): 仅 log warning + InitReport, 不再 LLM 补
       (Plan B 单步 prompt 已强制每子类最小条数, 缺口 → 应该改 prompt 而非加补丁)
    3. 语义去重 (_embed_and_dedupe)
    4. 批量写库 (store_memories_batch, force=True 清旧 AI 记忆)
    5. Redis 分布式锁防并发重入

    NOTE: name/gender/mbti 不参与本函数: 它们已在 generate_full_profile 阶段融入
    prompt; convert_profile_to_memories 直接从 profile.identity 拿姓名/性别 (硬覆盖).
    旧架构的 phase3/4 LLM gap-fill / select_character_profile 已废弃。
    """
    try:
        async with memory_generation_lock(agent_id):
            async with init_report(agent_id, profile_id=None) as report:
                return await _run_l1_coverage(
                    agent_id, user_id, profile, career_template, workspace_id, report,
                )
    except MemoryGenerationLocked:
        logger.warning(f"generate_l1_coverage already running for agent {agent_id}; skipping")
        return 0


async def _run_l1_coverage(
    agent_id: str, user_id: str, profile: dict, career_template: dict | None,
    workspace_id: str | None, report: InitReport,
) -> int:
    """Plan B 简化版主流程. profile 是 character_generation 输出的字典 (字段名
    与 character.generation prompt JSON schema 对齐). career_template 是池里
    随机选的 dict (用于 convert_profile_to_memories 中职业相关 sub 映射)."""
    await set_progress(agent_id, "converting", message="正在转换背景信息为长期记忆...")
    with phase_timer(report, "phase1_convert"):
        memories = convert_profile_to_memories(profile, career_template)
        # 兜底派生: postprocess 已硬覆盖 ethnicity / blood_type / name. 但若 LLM
        # 漏了 zodiac/constellation 字段, convert 不会产生对应记忆 → 这里从
        # birthday 派生补齐 (taxonomy 这两个 sub 都是 SINGLETON, 必须有 1 条)。
        existing_subs = {
            (m["main_category"], m["sub_category"]) for m in memories
        }
        identity = as_dict(profile, "identity")
        birthday = _clean_text(identity.get("birthday"))
        if ("身份", "星座") not in existing_subs:
            v = derive_constellation(birthday)
            if v:
                memories.append({"content": f"我是{v}", "main_category": "身份",
                                "sub_category": "星座", "type": "identity", "importance": 0.85})
        if ("身份", "生肖") not in existing_subs:
            v = derive_zodiac(birthday)
            if v:
                memories.append({"content": f"我属{v}", "main_category": "身份",
                                "sub_category": "生肖", "type": "identity", "importance": 0.85})
        if ("身份", "民族") not in existing_subs:
            memories.append({"content": f"我是{sample_ethnicity(agent_id)}",
                            "main_category": "身份", "sub_category": "民族",
                            "type": "identity", "importance": 0.85})

    conditional_include = analyze_conditional_subs(profile)
    report.direct_count = len(memories)
    report.conditional_included = [f"{m}/{s}" for m, s in sorted(conditional_include)]
    logger.info(
        f"Plan B convert → {len(memories)} memories for agent {agent_id}, "
        f"conditional includes: {report.conditional_included or '(none)'}"
    )

    # 覆盖度验证 (仅 log + report, 不 retry; Plan B 单 prompt 已强制最小条数)
    gaps = _compute_l1_gaps(memories, conditional_include=conditional_include)
    total_gap_count = sum(sum(s.values()) for s in gaps.values())
    report.gaps_after_phase1 = total_gap_count
    report.gaps_after_llm = total_gap_count
    report.llm_count = 0
    if gaps:
        logger.warning(
            f"agent {agent_id}: L1 coverage gaps after Plan B single-LLM: "
            f"{total_gap_count} entries across {len(gaps)} mains. Detail: "
            f"{ {m: dict(s) for m, s in gaps.items()} }. "
            f"调整 character.generation prompt 强化最小条数提示可减少缺口."
        )

    # spec《背景信息》§1.4: 跨字段一致性兜底. character.generation prompt 已加
    # 硬约束 (rule 11), 这里 LLM 二次扫描捕捉漏网矛盾对 (e.g. 身份/宠物 vs
    # 生活/宠物). 失败回退原数据, 不阻塞.
    await set_progress(agent_id, "consistency", message="正在校对人设一致性...")
    pre_contra = len(memories)
    with phase_timer(report, "phase1b_contradiction_scan"):
        memories = await _detect_and_resolve_contradictions(memories)
    report.contradiction_removed = pre_contra - len(memories)

    await set_progress(agent_id, "embedding", message="正在向量化与去重...")
    pre_dedupe = len(memories)
    with phase_timer(report, "phase2_dedupe"):
        memories = await _embed_and_dedupe(memories)
    report.dedupe_removed = pre_dedupe - len(memories)

    await set_progress(agent_id, "storing", message="正在写入记忆库...")
    with phase_timer(report, "phase3_store"):
        stored_ids = await store_memories_batch(
            agent_id, user_id, memories,
            workspace_id=workspace_id,
            force=True,
        )

    report.total_stored = len(stored_ids)
    report.distinct_subs = len({(m["main_category"], m["sub_category"]) for m in memories})
    logger.info(
        f"L1 coverage stored for agent {agent_id}: {len(stored_ids)} memories, "
        f"{report.distinct_subs} distinct (main, sub) pairs"
    )
    return len(stored_ids)


# NOTE: generate_full_life_story 旧 wrapper 已废弃 (依赖 prepare_profile_for_agent
# 选 character_profile 池, Plan B 改 agents.py:_init_and_generate_story 直接编排)。


async def activate_agent(agent_id: str) -> None:
    """Set agent status to active."""
    try:
        await db.aiagent.update(
            where={"id": agent_id},
            data={"status": "active"},
        )
    except Exception as e:
        logger.error(f"Failed to activate agent {agent_id}: {e}")
