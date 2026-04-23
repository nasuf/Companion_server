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
import time
import uuid

from collections import Counter

from app.db import db, ensure_connected
from app.redis_client import get_redis
from app.services.llm.models import (
    get_chat_model,
    get_embedding_model,
    invoke_json,
    invoke_json_with_usage,
)
from app.services.memory.demographics import (
    derive_constellation,
    derive_zodiac,
    sample_blood_type,
    sample_ethnicity,
)
from app.services.memory.generation_lock import (
    MemoryGenerationLocked,
    memory_generation_lock,
)
from app.services.memory.init_report import (
    InitReport,
    MainStats,
    init_report,
    phase_timer,
)
from app.services.memory.normalization import cosine_similarity
from app.services.memory.storage.persistence import normalize_memory_type
from app.services.memory.taxonomy import (
    L1_CONDITIONAL_SUBS,
    L1_COVERAGE_EXEMPT,
    L1_SINGLETON_SUBS,
    L1_TARGET_EMOTION,
    L1_TARGET_MULTI,
    MAIN_CATEGORY_TO_LEGACY_TYPE,
    TAXONOMY_MATRIX,
    allowed_sub_categories,
    analyze_conditional_subs,
    as_dict,
    l1_min_importance,
    l1_target_count,
    resolve_taxonomy,
)
from app.services.memory.retrieval.vector_search import format_vector
from app.services.runtime.cache import bump_cache_version
from app.services.workspace.workspaces import resolve_workspace_id

logger = logging.getLogger(__name__)

PROGRESS_KEY_PREFIX = "provision_progress:"
PROGRESS_TTL = 3600

# Concurrent LLM calls cap. 5 matches the AI L1 main count so all mains
# can run in parallel; raise if upstream TPM allows more.
_LLM_MAX_CONCURRENCY = 5
_LLM_SEMAPHORE = asyncio.Semaphore(_LLM_MAX_CONCURRENCY)

# Cosine similarity threshold for intra-(main, sub) memory dedup.
_DEDUPE_THRESHOLD = 0.88

# Inserted into prompt when filling the 情绪 main category. Emotion memories
# are rare high-impact events, not routine ups and downs — this guidance
# keeps LLM from producing generic fluff.
_EMOTION_GUIDANCE = (
    "\n\n【本大类特殊要求 — 情绪】\n"
    "情绪类记忆不是日常琐碎情绪波动, 而是 **此生难忘、塑造人格** 的过往事件:\n"
    "- 每条描述一个具体时刻/事件, 带时间或场景锚点 (如 '高三那年母亲住院', '第一次获奖上台时')\n"
    "- 情绪强度应极高, 至今回想仍有感受\n"
    "- 情绪反应模式必须与 MBTI 性格一致 (内向型在独处时更易感孤独; 直觉型对未来可能性更敏感; 等等)\n"
    "- 与 profile 的 fears/values/career 逻辑自洽, 不得凭空造职业/家庭情节\n"
)
_RETRY_NOTE = "\n⚠️ 上一轮生成遗漏了下列子类，本轮必须逐个补齐（不得再空返）。\n"

# (profile likes field, taxonomy sub-category, 记忆句式前缀)
# Hoisted to module scope to avoid re-allocating the tuple on every call.
_LIKES_TO_SUB: tuple[tuple[str, str, str], ...] = (
    ("foods", "饮食喜好", "喜欢吃"),
    ("fruits", "饮食喜好", "喜欢的水果"),
    ("colors", "审美爱好", "喜欢的颜色"),
    ("season", "审美爱好", "喜欢的季节"),
    ("weather", "审美爱好", "喜欢的天气"),
    ("plants", "审美爱好", "喜欢的植物"),
    ("animals", "审美爱好", "喜欢的动物"),
    ("music", "审美爱好", "喜欢的音乐"),
    ("songs", "审美爱好", "喜欢的歌曲"),
    ("sounds", "审美爱好", "喜欢的声音"),
    ("scents", "审美爱好", "喜欢的气味"),
    ("books", "审美爱好", "喜欢看的书"),
    ("movies", "审美爱好", "喜欢的电影"),
    ("sports", "生活习惯", "喜欢的运动"),
)


async def set_progress(agent_id: str, stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
    """Update provisioning progress in Redis.

    Stages (new pipeline):
      selecting_profile  →  2%    匹配角色背景
      converting_profile →  5%    profile 直转 + 派生
      generating_chapter → 10-85% 按 main 并发 LLM 补齐 + retry
      storing_memories   → 88-98% 去重 + embedding + 写库
      complete           → 100%
      failed             → -1
    """
    redis = await get_redis()
    percent = 0
    if stage == "selecting_profile":
        percent = 2
    elif stage == "converting_profile":
        percent = 5
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


# DB / API 一律存英文 male/female; profile 种子数据是历史中文, 读取时翻译一次。
_PROFILE_GENDER_ZH_TO_EN = {"男": "male", "女": "female"}
_GENDER_EN_TO_ZH = {"male": "男", "female": "女"}


def _profile_gender_en(profile_data: dict) -> str | None:
    raw = as_dict(profile_data, "identity").get("gender")
    if not raw:
        return None
    # profile 种子数据是中文, 新数据已是英文, 两者都透传为英文 male/female。
    return _PROFILE_GENDER_ZH_TO_EN.get(raw, raw)


async def select_character_profile(gender: str | None) -> dict | None:
    """全栈约定: gender ∈ {"male", "female", None}. 指定但 0 匹配 → None。"""
    profiles = await db.characterprofile.find_many(
        where={"status": "published"},
        include={"career": True},
    )
    matching = []
    for p in profiles:
        data = p.data if isinstance(p.data, dict) else {}
        p_gender = _profile_gender_en(data)
        if gender and p_gender and p_gender != gender:
            continue
        matching.append(p)
    if not matching:
        if gender is not None:
            logger.warning(f"No CharacterProfile matches gender={gender!r}; skip life-story")
        return None

    selected = random.choice(matching)
    return {
        "id": selected.id,
        "data": selected.data if isinstance(selected.data, dict) else {},
        "career_template": {
            "title": selected.career.title,
            "duties": selected.career.duties,
            "social_value": selected.career.socialValue,
            "clients": selected.career.clients,
        } if selected.career else None,
    }


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


def _add(memories: list, summary: str, main: str, sub: str, mem_type: str, importance: float) -> None:
    memories.append({
        "summary": summary,
        "main_category": main,
        "sub_category": sub,
        "type": mem_type,
        "importance": importance,
    })


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
    fears = as_dict(profile_data, "fears")

    # ── 身份: singleton 事实 ──
    gender = identity.get("gender")
    if isinstance(gender, str):
        gender = _GENDER_EN_TO_ZH.get(gender, gender)
    if (v := _clean_text(gender)):
        _add(memories, f"我性别是{v}", "身份", "性别", "identity", 0.95)
    if (v := _clean_text(identity.get("age"))):
        _add(memories, f"我今年{v}岁", "身份", "年龄", "identity", 0.95)
    if (v := _clean_text(identity.get("birthday"))):
        _add(memories, f"我生日是{v}", "身份", "生日", "identity", 0.90)
    if (v := _clean_text(identity.get("location"))):
        _add(memories, f"我现在住在{v}", "身份", "现居地", "identity", 0.90)
    if (v := _clean_text(identity.get("birthplace"))):
        _add(memories, f"我出生在{v}", "身份", "出生地", "identity", 0.90)
    if (v := _clean_text(identity.get("growing_up_location"))):
        _add(memories, f"我在{v}长大", "身份", "成长地", "identity", 0.90)
    if (v := _clean_text(identity.get("ethnicity"))):
        _add(memories, f"我是{v}", "身份", "民族", "identity", 0.85)
    if (v := _clean_text(identity.get("blood_type"))):
        _add(memories, f"我是{v}血", "身份", "血型", "identity", 0.80)
    if (v := _clean_text(identity.get("family"))):
        _add(memories, f"我的家庭情况: {v}", "身份", "亲属关系", "identity", 0.90)

    # ── 身份: 外貌特征 —— 每个维度一条, 合计 3-5 ──
    for key, label in (("height", "身高"), ("weight", "体型"), ("features", "外貌特征"),
                       ("style", "穿搭风格"), ("voice", "声音特点")):
        if (v := _clean_text(appearance.get(key))):
            _add(memories, f"我的{label}: {v}", "身份", "外貌特征", "identity", 0.78)

    # ── 身份: 教育背景 ──
    if (v := _clean_text(edu.get("degree"))):
        _add(memories, f"我的学历: {v}", "身份", "教育背景", "identity", 0.85)

    # ── 生活: 技能 —— 知识擅长 + 自学 每项一条 ──
    for item in _as_list(edu.get("strengths")):
        _add(memories, f"我擅长 {item} 相关的知识", "生活", "技能", "life", 0.80)
    for item in _as_list(edu.get("self_taught")):
        _add(memories, f"我自学过 {item}", "生活", "技能", "life", 0.78)
    for item in _as_list(abilities.get("good_at")):
        _add(memories, f"我擅长 {item}", "生活", "技能", "life", 0.80)

    # ── 职业: 优先 career_template, 回退 profile_data.career ──
    ct = career_template
    if not ct:
        pc = profile_data.get("career")
        if isinstance(pc, dict) and pc.get("title"):
            ct = pc
    if ct:
        if (v := _clean_text(ct.get("title"))):
            _add(memories, f"我的职业是 {v}", "身份", "职业/与经济", "identity", 0.95)
        if (v := _clean_text(ct.get("duties"))):
            _add(memories, f"我的工作内容: {v}", "生活", "工作", "life", 0.88)
        # clients 是 tag 数组 — 每项一条工作相关记忆
        for client in _as_list(ct.get("clients")):
            _add(memories, f"我的服务对象包括 {client}", "生活", "工作", "life", 0.78)
        if (v := _clean_text(ct.get("social_value") or ct.get("socialValue"))):
            _add(memories, f"我工作的社会价值: {v}", "生活", "工作", "life", 0.78)

    for key, sub, prefix in _LIKES_TO_SUB:
        for item in _as_list(likes.get(key)):
            _add(memories, f"我{prefix} {item}", "偏好", sub, "preference", 0.73)
    # 小癖好是 textarea, 整段一条
    if (v := _clean_text(likes.get("quirks"))):
        _add(memories, f"我的小癖好: {v}", "偏好", "生活习惯", "preference", 0.75)

    # ── 偏好: dislikes 每项一条 ──
    for item in _as_list(dislikes.get("foods")):
        _add(memories, f"我讨厌吃 {item}", "偏好", "饮食厌恶", "preference", 0.78)
    for item in _as_list(dislikes.get("sounds")):
        _add(memories, f"我讨厌 {item} 这种声音", "偏好", "审美厌恶", "preference", 0.75)
    for item in _as_list(dislikes.get("smells")):
        _add(memories, f"我讨厌 {item} 的气味", "偏好", "审美厌恶", "preference", 0.75)
    for item in _as_list(dislikes.get("habits")):
        _add(memories, f"我讨厌别人 {item}", "偏好", "人际厌恶", "preference", 0.78)

    # ── 偏好: fears + abilities.never_do 归 禁忌/雷区 ──
    for item in _as_list(fears.get("animals")):
        _add(memories, f"我害怕 {item}", "偏好", "禁忌/雷区", "preference", 0.85)
    for item in _as_list(fears.get("objects")):
        _add(memories, f"我害怕 {item}", "偏好", "禁忌/雷区", "preference", 0.82)
    for item in _as_list(fears.get("atmospheres")):
        _add(memories, f"我害怕 {item} 的氛围", "偏好", "禁忌/雷区", "preference", 0.82)
    for item in _as_list(abilities.get("never_do")):
        _add(memories, f"我绝对不会做: {item}", "偏好", "禁忌/雷区", "preference", 0.88)

    # ── 思维: 价值观 / 理想与目标 / 自我认知 ──
    if (v := _clean_text(values.get("motto"))):
        _add(memories, f"我的人生信条: {v}", "思维", "价值观", "thought", 0.92)
    for item in _as_list(values.get("believes")):
        _add(memories, f"我相信 {item}", "思维", "价值观", "thought", 0.90)
    for item in _as_list(values.get("opposes")):
        _add(memories, f"我反对 {item}", "思维", "价值观", "thought", 0.90)
    if (v := _clean_text(values.get("goal"))):
        _add(memories, f"我的人生目标: {v}", "思维", "理想与目标", "thought", 0.90)
    for item in _as_list(abilities.get("limits")):
        _add(memories, f"我清楚自己的局限: {item}", "思维", "自我认知", "thought", 0.88)

    return memories


# ── Phase 2: LLM 按子类补齐覆盖 (taxonomy-driven) ──


def _build_constraints(
    name: str,
    gender: str | None,
    mbti: dict | None,
    profile_data: dict,
    career_template: dict | None,
) -> str:
    """硬约束块: LLM 任何子类输出都不得违反这些事实。"""
    from app.services.mbti import format_mbti_for_prompt

    identity = as_dict(profile_data, "identity")
    gender_zh = _GENDER_EN_TO_ZH.get(str(gender or ""), gender or "未指定")
    lines = [
        "=== 硬约束 (任何子类输出都不得违反) ===",
        f"姓名: {name}",
        f"性别: {gender_zh}",
        f"年龄: {identity.get('age', '未知')}",
        f"生日: {identity.get('birthday', '未知')}",
    ]
    for key, label in (("location", "现居地"), ("birthplace", "出生地"),
                       ("growing_up_location", "成长地")):
        if identity.get(key):
            lines.append(f"{label}: {identity[key]}")
    if career_template and career_template.get("title"):
        lines.append(f"职业: {career_template['title']}")
    if (mbti_text := format_mbti_for_prompt(mbti)):
        lines.append(f"MBTI: {mbti_text}")
    lines.append("=======================================")
    return "\n".join(lines)


def _derive_timeline(profile_data: dict) -> str:
    """派生简单时间线锚点, 作为硬约束让 5 个 main 并发调用时间线一致。"""
    identity = as_dict(profile_data, "identity")
    birthplace = identity.get("birthplace") or identity.get("location") or "家乡"
    grown = identity.get("growing_up_location") or birthplace
    current = identity.get("location") or grown
    age_raw = identity.get("age")
    try:
        age = int(age_raw) if age_raw is not None else 25
    except (ValueError, TypeError):
        age = 25
    return (
        "=== 人生时间线 (各子类记忆若涉及年龄/地点必须与此一致) ===\n"
        f"- 童年 (0-6 岁): 在 {birthplace}\n"
        f"- 少年 (7-15 岁): 在 {grown} 上学\n"
        f"- 青年 (16-22 岁): 离家求学/工作, 在 {grown} 或周边城市\n"
        f"- 成年 (23-{age} 岁, 当前): 在 {current} 生活工作\n"
        "=================================================="
    )


# 每个 main 调用时只注入与该 main 相关的 profile 切片, 减少 token 冗余。
_PROFILE_RELEVANCE: dict[str, tuple[str, ...]] = {
    "身份": ("identity", "appearance", "education_knowledge", "career"),
    "偏好": ("likes", "dislikes", "fears", "abilities", "identity"),
    "生活": ("career", "education_knowledge", "abilities", "identity", "likes"),
    "情绪": ("values", "fears", "abilities", "identity"),
    "思维": ("values", "abilities", "education_knowledge", "identity"),
}


def _slice_profile(main: str, profile_data: dict, career_template: dict | None) -> str:
    """按 main 取 profile 子集并格式化."""
    keys = _PROFILE_RELEVANCE.get(main, tuple(profile_data.keys()))
    lines: list[str] = []
    for k in keys:
        v = profile_data.get(k)
        if v:
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    if career_template and main in ("身份", "生活"):
        lines.append(f"career: {json.dumps(career_template, ensure_ascii=False)}")
    return "\n".join(lines) if lines else "(无相关背景)"


def _digest_existing(memories: list[dict], per_sub: int = 2) -> str:
    """已生成记忆摘要 (每 sub 取前 N 条) 给 LLM 看, 避免重复生成。"""
    by_sub: dict[tuple[str, str], list[str]] = {}
    for m in memories:
        key = (m.get("main_category", ""), m.get("sub_category", ""))
        by_sub.setdefault(key, []).append(m.get("summary", ""))
    lines = []
    for (main, sub), summaries in by_sub.items():
        lines.append(f"[{main}/{sub}]: " + "; ".join(summaries[:per_sub]))
    return "\n".join(lines) if lines else "(无)"


def _spec_for_gap(main: str, gap_subs: dict[str, int]) -> str:
    """格式化缺口子类规格(目标条数 + 最低 importance)给 prompt."""
    lines = []
    for sub in gap_subs:
        if (main, sub) in L1_SINGLETON_SUBS:
            n_min = n_max = 1
        elif main == "情绪":
            n_min, n_max = L1_TARGET_EMOTION
        else:
            n_min, n_max = L1_TARGET_MULTI
        lines.append(
            f"  - {sub}: {n_min}-{n_max} 条, "
            f"importance ≥ {l1_min_importance(main, sub):.2f}"
        )
    return "\n".join(lines)


async def _fill_main_gaps(
    main: str,
    gap_subs: dict[str, int],
    name: str,
    profile_data: dict,
    career_template: dict | None,
    constraints: str,
    timeline: str,
    existing_digest: str,
    *,
    is_retry: bool = False,
    stats: MainStats | None = None,
) -> list[dict]:
    """针对一个 main 大类, 用一次 LLM 调用输出所有缺口子类的 3-5 条记忆。

    gender/mbti 已通过 constraints 注入。is_retry=True 时提示 LLM 上一轮
    遗漏了这些子类, 必须逐个补齐。stats 非 None 时写入 token/duration/count 统计。
    """
    profile_slice = _slice_profile(main, profile_data, career_template)
    spec = _spec_for_gap(main, gap_subs)
    retry_note = _RETRY_NOTE if is_retry else ""
    emotion_guidance = _EMOTION_GUIDANCE if main == "情绪" else ""

    prompt = f"""你在为 AI 伙伴「{name}」生成记忆库中「{main}」大类的 L1 记忆。

{constraints}

{timeline}

=== 角色相关背景 ===
{profile_slice}

=== 已有记忆(请勿重复) ===
{existing_digest}

=== 需要生成的子类 ==={retry_note}
{spec}{emotion_guidance}

要求:
1. 每条 summary 以"我"开头, 20-80 字, 描述一个具体事实/偏好/经历/体验
2. 同一子类的多条记忆应展现该事实/偏好的不同侧面(场景/来源/情感/频次), 句式不要雷同
3. 所有内容必须与上方 "硬约束 + 时间线" 严格一致, 不得矛盾
4. importance 在指定下限到 0.95 之间, 重要事实更高
5. sub_category 必须严格使用 "=== 需要生成的子类 ===" 列出的名字, 不得自造

输出严格 JSON, 顶层按子类分组:
{{
  "<子类名>": [
    {{"summary": "...", "importance": 0.85}},
    ...
  ],
  ...
}}
"""
    start_ms = int(time.time() * 1000)
    async with _LLM_SEMAPHORE:
        result, usage = await invoke_json_with_usage(get_chat_model(), prompt)
    if stats is not None:
        stats.duration_ms = int(time.time() * 1000) - start_ms
        stats.tokens_in = usage.get("input_tokens", 0)
        stats.tokens_out = usage.get("output_tokens", 0)

    if not isinstance(result, dict):
        logger.warning(f"_fill_main_gaps({main}) returned non-dict: {type(result)}")
        if stats is not None:
            stats.failed = True
        return []

    allowed_subs = set(allowed_sub_categories(main, "ai", 1))
    mem_type = MAIN_CATEGORY_TO_LEGACY_TYPE[main]

    out: list[dict] = []
    for sub, items in result.items():
        if sub not in allowed_subs or sub not in gap_subs:
            continue
        if not isinstance(items, list):
            continue
        min_imp = l1_min_importance(main, sub)
        for item in items:
            if not isinstance(item, dict):
                continue
            summary = _clean_text(item.get("summary"))
            if not summary or len(summary) < 6:
                continue
            try:
                imp = float(item.get("importance", min_imp))
            except (TypeError, ValueError):
                imp = min_imp
            imp = max(min_imp, min(0.95, imp))
            out.append({
                "summary": summary,
                "main_category": main,
                "sub_category": sub,
                "type": mem_type,
                "importance": imp,
            })
    return out


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
    vectors = await model.aembed_documents([m["summary"] for m in memories])
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
    # Filter empty summaries
    valid = [m for m in all_memories if m.get("summary")]
    if not valid:
        return []

    total = len(valid)

    # Resolve workspace once (shared by all memories + changelogs)
    workspace_id = workspace_id or await resolve_workspace_id(user_id=user_id)

    existing = await db.aimemory.count(
        where={"userId": user_id, "workspaceId": workspace_id}
    )
    if existing > 0 and not force:
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
        new_vecs = await embed_model.aembed_documents([valid[i]["summary"] for i in pending_idx])
        for i, v in zip(pending_idx, new_vecs):
            valid[i]["_embedding"] = v
    embeddings = [m["_embedding"] for m in valid]

    # Refresh DB connection (may have gone stale during embedding)
    await ensure_connected()

    if existing > 0:  # force=True path
        logger.info(
            f"store_memories_batch force=True for agent {agent_id}: clearing "
            f"{existing} existing AI memories in workspace {workspace_id}"
        )
        await db.aimemory.delete_many(
            where={"userId": user_id, "workspaceId": workspace_id}
        )

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


async def generate_l1_coverage(
    agent_id: str,
    user_id: str,
    name: str,
    gender: str | None,
    mbti: dict | None,
    profile: dict,
    workspace_id: str | None = None,
) -> int:
    """Step B (refactored): 生成完整 L1 覆盖的记忆库。

    取代旧的 "大纲 + 12-15 章节" 流程; taxonomy-driven:
      1. profile 直转 (singleton 1 条 / list 每项 1 条)
      2. 生日派生 星座/生肖, 兜底 血型/民族 (人口分布)
      3. 遍历 AI L1 taxonomy 算缺口, 尊重 EXEMPT + OPTIONAL
      4. 按 main 并发 LLM 一次性补齐 (硬约束 + 时间线 + 多样性要求)
      5. 缺口 retry (按 main 批) + Embedding 语义去重
      6. 写库 (force 清理旧 AI 记忆以支持重试)
      7. Redis 分布式锁防并发重入
    """
    try:
        async with memory_generation_lock(agent_id):
            async with init_report(agent_id, profile_id=profile.get("id")) as report:
                return await _run_l1_coverage(
                    agent_id, user_id, name, gender, mbti, profile,
                    workspace_id, report,
                )
    except MemoryGenerationLocked:
        logger.warning(f"generate_l1_coverage already running for agent {agent_id}; skipping")
        return 0


def _phase1_direct_memories(name: str, profile_data: dict,
                             career_template: dict | None, agent_id: str) -> list[dict]:
    """Build the full set of deterministic profile-derived memories (phase 1).

    Covers: profile singletons/lists, 姓名 seed, 星座/生肖 from birthday,
    and blood type / ethnicity fallback sampled from population distributions.
    """
    memories = convert_profile_to_memories(profile_data, career_template)
    memories.insert(0, {
        "summary": f"我叫{name}",
        "main_category": "身份", "sub_category": "姓名",
        "type": "identity", "importance": 1.0,
    })

    identity = as_dict(profile_data, "identity")
    birthday = _clean_text(identity.get("birthday"))
    if (v := derive_constellation(birthday)):
        memories.append({"summary": f"我是{v}", "main_category": "身份",
                        "sub_category": "星座", "type": "identity", "importance": 0.75})
    if (v := derive_zodiac(birthday)):
        memories.append({"summary": f"我属{v}", "main_category": "身份",
                        "sub_category": "生肖", "type": "identity", "importance": 0.75})

    existing_identity_subs = {
        m["sub_category"] for m in memories if m["main_category"] == "身份"
    }
    if "血型" not in existing_identity_subs:
        memories.append({"summary": f"我是{sample_blood_type(agent_id)}血", "main_category": "身份",
                        "sub_category": "血型", "type": "identity", "importance": 0.75})
    if "民族" not in existing_identity_subs:
        memories.append({"summary": f"我是{sample_ethnicity(agent_id)}", "main_category": "身份",
                        "sub_category": "民族", "type": "identity", "importance": 0.80})

    return memories


async def _phase3_4_llm_fill(
    agent_id: str, name: str, gender: str | None, mbti: dict | None,
    profile_data: dict, career_template: dict | None,
    memories: list[dict], gaps: dict[str, dict[str, int]],
    conditional_include: set[tuple[str, str]],
    report: InitReport,
) -> None:
    """Run phase 3 (per-main LLM fill) + phase 4 (retry still-missing subs).

    Mutates `memories` in place and writes per-main stats into `report`.
    """
    constraints = _build_constraints(name, gender, mbti, profile_data, career_template)
    timeline = _derive_timeline(profile_data)
    digest = _digest_existing(memories)

    def _fill(m: str, subs: dict[str, int], digest_text: str, *, retry: bool = False,
              stats: MainStats | None = None):
        return _fill_main_gaps(m, subs, name, profile_data, career_template,
                               constraints, timeline, digest_text,
                               is_retry=retry, stats=stats)

    phase3_stats = {m: MainStats() for m in gaps}
    with phase_timer(report, "phase3_fill_llm"):
        results = await asyncio.gather(
            *(_fill(m, subs, digest, stats=phase3_stats[m]) for m, subs in gaps.items()),
            return_exceptions=True,
        )
    done_mains: list[str] = []
    for main, res in zip(gaps.keys(), results):
        if isinstance(res, list):
            memories.extend(res)
            phase3_stats[main].produced = len(res)
            done_mains.append(f"{main} ✓")
        else:
            phase3_stats[main].failed = True
            done_mains.append(f"{main} ✗")
            logger.warning(f"Phase3 main={main} failed: {res}")
    report.main_stats = phase3_stats
    await set_progress(
        agent_id, "generating_chapter",
        current=len(gaps), total=len(gaps),
        message=" · ".join(done_mains),
    )

    still_missing = _compute_l1_gaps(memories, conditional_include=conditional_include)
    if not still_missing:
        return

    await set_progress(
        agent_id, "generating_chapter",
        current=len(gaps), total=len(gaps),
        message="正在补齐剩余子类...",
    )
    retry_digest = _digest_existing(memories)
    phase4_stats = {m: MainStats() for m in still_missing}
    with phase_timer(report, "phase4_retry"):
        retry_results = await asyncio.gather(
            *(_fill(m, subs, retry_digest, retry=True, stats=phase4_stats[m])
              for m, subs in still_missing.items()),
            return_exceptions=True,
        )
    for main, res in zip(still_missing.keys(), retry_results):
        if isinstance(res, list) and res:
            memories.extend(res)
            phase4_stats[main].produced = len(res)
        else:
            phase4_stats[main].failed = True
            logger.warning(f"L1 coverage retry for main={main} produced nothing: {res!r}")
    report.retry_stats = phase4_stats


async def _run_l1_coverage(
    agent_id: str, user_id: str, name: str, gender: str | None,
    mbti: dict | None, profile: dict, workspace_id: str | None,
    report: InitReport,
) -> int:
    profile_data = profile["data"]
    career_template = profile.get("career_template")

    await set_progress(agent_id, "converting_profile", message="正在转换背景信息为记忆...")
    with phase_timer(report, "phase1_direct"):
        memories = _phase1_direct_memories(name, profile_data, career_template, agent_id)

    conditional_include = analyze_conditional_subs(profile_data)
    report.direct_count = len(memories)
    report.conditional_included = [f"{m}/{s}" for m, s in sorted(conditional_include)]
    logger.info(
        f"Phase1 profile direct → {len(memories)} memories for agent {agent_id}, "
        f"conditional includes: {report.conditional_included or '(none)'}"
    )

    gaps = _compute_l1_gaps(memories, conditional_include=conditional_include)
    report.gaps_after_phase1 = sum(sum(s.values()) for s in gaps.values())

    if not gaps:
        logger.info(f"agent {agent_id}: no L1 gaps after phase 1, skip LLM")
    else:
        await set_progress(
            agent_id, "generating_chapter",
            current=0, total=len(gaps),
            message=f"正在补齐 {len(gaps)} 类 / {report.gaps_after_phase1} 条 L1 记忆覆盖...",
        )
        await _phase3_4_llm_fill(
            agent_id, name, gender, mbti, profile_data, career_template,
            memories, gaps, conditional_include, report,
        )

    report.gaps_after_llm = sum(
        sum(s.values()) for s in _compute_l1_gaps(memories,
                                                 conditional_include=conditional_include).values()
    )
    report.llm_count = len(memories) - report.direct_count

    await set_progress(agent_id, "storing_memories", message="正在去重并整理记忆...")
    pre_dedupe = len(memories)
    with phase_timer(report, "phase5_dedupe"):
        memories = await _embed_and_dedupe(memories)
    report.dedupe_removed = pre_dedupe - len(memories)

    with phase_timer(report, "phase6_store"):
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


async def generate_full_life_story(
    agent_id: str,
    user_id: str,
    name: str,
    gender: str | None,
    mbti: dict | None,
    workspace_id: str | None = None,
) -> None:
    """Wrapper: select profile → generate L1 coverage → activate.

    prepare_profile_for_agent + generate_l1_coverage 更灵活 (允许
    life_overview 等下游任务并行)。本函数保证总会写 progress 并把 agent
    置为 active (即使中途失败)。
    """
    try:
        result = await prepare_profile_for_agent(agent_id, gender)
        if not result:
            await set_progress(agent_id, "complete", message="无可用背景模板，已跳过")
            await activate_agent(agent_id)
            return

        profile, _ = result
        count = await generate_l1_coverage(
            agent_id=agent_id,
            user_id=user_id,
            name=name,
            gender=gender,
            mbti=mbti,
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
