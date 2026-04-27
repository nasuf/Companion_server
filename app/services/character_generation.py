"""Plan B 单步生成 character profile.

Spec §1.4「AI背景生成」: agent 创建时输入 (姓名/性别/年龄/职业/MBTI/7维) →
单次 LLM 调用 → 输出全 5 维 L1 记忆 JSON。

替代旧架构:
- 旧: admin 批量预生 character_profiles (无 MBTI) → 用户创建时随机抽 → 直转 + gap-fill
- 新: 用户创建 agent 时即时 LLM 生成 (含 MBTI) → 直转

复用:
- prompt registry: character.generation
- profile JSON 字段名: 与 convert_profile_to_memories 字面对齐 (contract test 锁)
- 后处理: character._apply_postprocess_overrides (ethnicity 汉族 / blood_type
  4 选 1 / name / career 硬覆盖)
- 字段缺失修复: character._detect_missing_fields + _repair_missing_fields
- LLM 韧性: invoke_json(profile="background") (max_retries=2, timeout 120s)
"""

from __future__ import annotations

import logging
import random
from datetime import date
from typing import Any

from app.services.character import (
    _apply_postprocess_overrides,
    _detect_missing_fields,
    _repair_missing_fields,
    _split_clients,
    AGE_MAX,
    AGE_MIN,
    clamp_agent_age,
)
from app.services.llm.models import get_chat_model, invoke_json
from app.services.mbti import format_mbti_for_prompt
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)


# ── 简化 schema: 仅供 _detect_missing_fields / _repair_missing_fields 用 ──
# 与 character.generation prompt 中的 JSON schema 字面对齐 (contract test
# tests/test_convert_profile_contract.py 锁定字段名)。type 标记决定 _is_field_empty
# 的判断策略:
#   - "tags" 字段: 必须是非空 list
#   - "text" / "textarea" / "select": 必须是非空字符串
#   - "number": 必须是 int/float
GENERATION_SCHEMA: dict[str, Any] = {
    "categories": [
        {
            "key": "identity",
            "name": "基础身份",
            "fields": [
                # 这些字段在 _REPAIR_SKIP_IDENTITY_FIELDS 中被跳过 (硬覆盖):
                # name / gender / ethnicity / blood_type
                {"key": "age", "name": "年龄", "type": "number"},
                {"key": "birthday", "name": "生日", "type": "text"},
                {"key": "zodiac", "name": "生肖", "type": "text"},
                {"key": "constellation", "name": "星座", "type": "text"},
                {"key": "birthplace", "name": "出生地", "type": "text"},
                {"key": "growing_up_location", "name": "成长地", "type": "text"},
                {"key": "location", "name": "现居地", "type": "text"},
                {"key": "family", "name": "亲属关系", "type": "tags",
                 "hint": "3-5 条父母/兄弟姐妹相关事实"},
                {"key": "social_relations", "name": "社会关系", "type": "tags",
                 "hint": "3-5 条社交圈层/朋友/同事描述"},
                {"key": "pet_profile", "name": "宠物", "type": "tags",
                 "hint": "若养宠则 3-5 条事实, 不养则空数组 []"},
            ],
        },
        {
            "key": "appearance",
            "name": "外貌与形象",
            "fields": [
                {"key": "height", "name": "身高", "type": "text"},
                {"key": "weight", "name": "体型", "type": "text"},
                {"key": "features", "name": "外貌特征", "type": "tags",
                 "hint": "3-5 条外貌特征"},
                {"key": "style", "name": "穿搭风格", "type": "tags",
                 "hint": "3-5 条穿搭描述"},
                {"key": "voice", "name": "声音特点", "type": "tags",
                 "hint": "3-5 条声音描述"},
            ],
        },
        {
            "key": "education_knowledge",
            "name": "教育与知识",
            "fields": [
                {"key": "degree", "name": "学历", "type": "tags",
                 "hint": "3-5 条求学经历"},
                {"key": "strengths", "name": "知识擅长", "type": "tags",
                 "hint": "3-5 个专业/知识领域"},
                {"key": "self_taught", "name": "自学技能", "type": "tags",
                 "hint": "1-2 项自学技能"},
            ],
        },
        {
            "key": "likes",
            "name": "喜好",
            "fields": [
                {"key": "foods", "name": "食物", "type": "tags"},
                {"key": "fruits", "name": "水果", "type": "tags"},
                {"key": "colors", "name": "颜色", "type": "tags"},
                {"key": "season", "name": "季节", "type": "tags"},
                {"key": "weather", "name": "天气", "type": "tags"},
                {"key": "plants", "name": "植物", "type": "tags"},
                {"key": "animals", "name": "动物", "type": "tags"},
                {"key": "music", "name": "音乐", "type": "tags"},
                {"key": "songs", "name": "歌曲", "type": "tags"},
                {"key": "sounds", "name": "声音", "type": "tags"},
                {"key": "scents", "name": "气味", "type": "tags"},
                {"key": "books", "name": "书籍类型", "type": "tags"},
                {"key": "movies", "name": "电影", "type": "tags"},
                {"key": "sports", "name": "运动", "type": "tags"},
                {"key": "quirks", "name": "小癖好", "type": "tags",
                 "hint": "3-5 条小癖好"},
            ],
        },
        {
            "key": "dislikes",
            "name": "讨厌",
            "fields": [
                {"key": "foods", "name": "食物", "type": "tags"},
                {"key": "sounds", "name": "声音", "type": "tags"},
                {"key": "smells", "name": "气味", "type": "tags"},
                {"key": "habits", "name": "习惯", "type": "tags"},
            ],
        },
        {
            "key": "interpersonal",
            "name": "人际偏好",
            "fields": [
                {"key": "liked_traits", "name": "欣赏特质", "type": "tags",
                 "hint": "3-5 条"},
                {"key": "disliked_traits", "name": "反感行为", "type": "tags",
                 "hint": "3-5 条"},
            ],
        },
        {
            "key": "lifestyle",
            "name": "生活习惯",
            "fields": [
                {"key": "routine", "name": "作息规律", "type": "tags"},
                {"key": "hygiene", "name": "卫生习惯", "type": "tags"},
                {"key": "leisure", "name": "休闲方式", "type": "tags"},
            ],
        },
        {
            "key": "taboo",
            "name": "禁忌雷区",
            "fields": [
                {"key": "items", "name": "底线", "type": "tags",
                 "hint": "3-5 条不可触碰的底线"},
            ],
        },
        {
            "key": "values",
            "name": "价值观",
            "fields": [
                {"key": "motto", "name": "人生信条", "type": "tags"},
                {"key": "believes", "name": "相信什么", "type": "tags"},
                {"key": "opposes", "name": "反对什么", "type": "tags"},
                {"key": "worldview", "name": "世界观", "type": "tags"},
                {"key": "goal", "name": "理想与目标", "type": "tags"},
                {"key": "interpersonal_view", "name": "人际关系观", "type": "tags"},
                {"key": "social_view", "name": "社会观点", "type": "tags"},
                {"key": "faith", "name": "精神寄托", "type": "tags"},
            ],
        },
        {
            "key": "abilities",
            "name": "能力边界",
            "fields": [
                {"key": "good_at", "name": "擅长", "type": "tags"},
                {"key": "never_do", "name": "原则禁止", "type": "tags"},
                {"key": "limits", "name": "能力上限", "type": "tags"},
            ],
        },
        {
            "key": "life_events",
            "name": "生活记忆事件",
            "fields": [
                # interaction 不预填: 该子类对应 (生活, 交互), 是 AI ↔ 当前用户
                # 的实际交互, 由聊天过程在 memory pipeline 累积. (taxonomy.py:
                # L1_COVERAGE_EXEMPT 含此条).
                # min_items 对齐 L1_TARGET_MULTI=(3,5) 中位数 3, pet/health/special
                # 等独立小子类降到 1-2 (符合 prompt 文案).
                {"key": "education", "name": "教育事件", "type": "tags", "min_items": 3},
                {"key": "work", "name": "工作事件", "type": "tags", "min_items": 3},
                {"key": "travel", "name": "旅行事件", "type": "tags", "min_items": 3},
                {"key": "living", "name": "居住事件", "type": "tags", "min_items": 3},
                {"key": "health", "name": "健康事件", "type": "tags", "min_items": 2},
                # pet 在不养宠场景允许 0; convert 已剔除"无养"等占位; 不强制 min
                {"key": "pet", "name": "宠物事件", "type": "tags"},
                {"key": "relationships", "name": "人际事件", "type": "tags", "min_items": 3},
                {"key": "skill_learning", "name": "技能学习", "type": "tags", "min_items": 3},
                {"key": "life", "name": "生活事件", "type": "tags", "min_items": 3},
                {"key": "special", "name": "特殊事件", "type": "tags"},
            ],
        },
        {
            "key": "emotion_events",
            "name": "情绪记忆事件",
            # min_items=2 对齐 L1_TARGET_EMOTION (2,3) 下界 (taxonomy.py:119).
            "fields": [
                {"key": "happy", "name": "高兴", "type": "tags", "min_items": 2},
                {"key": "sad", "name": "悲伤", "type": "tags", "min_items": 2},
                {"key": "angry", "name": "愤怒", "type": "tags", "min_items": 2},
                {"key": "fear", "name": "恐惧", "type": "tags", "min_items": 2},
                {"key": "disgust", "name": "厌恶", "type": "tags", "min_items": 2},
                {"key": "anxiety", "name": "焦虑", "type": "tags", "min_items": 2},
                {"key": "disappointment", "name": "失望", "type": "tags", "min_items": 2},
                {"key": "pride", "name": "自豪", "type": "tags", "min_items": 2},
                {"key": "moved", "name": "感动", "type": "tags", "min_items": 2},
                {"key": "embarrassed", "name": "尴尬", "type": "tags", "min_items": 2},
                {"key": "regret", "name": "遗憾", "type": "tags", "min_items": 2},
                {"key": "lonely", "name": "孤独", "type": "tags", "min_items": 2},
                {"key": "surprised", "name": "惊讶", "type": "tags", "min_items": 2},
                {"key": "grateful", "name": "感激", "type": "tags", "min_items": 2},
                {"key": "relieved", "name": "释怀", "type": "tags", "min_items": 2},
            ],
        },
    ],
}


# PersonalityInput (agent 创建表单 7 维) → CHARACTER_GENERATION_PROMPT 7 维占位符.
# 注: build_mbti 输出只含 4 轴 (EI/NS/TF/JP), 不含 7 维. 7 维必须从 personality_dict
# 直接读, 否则 prompt 会拿到 50/50/...的"无个性"默认值, 导致每个 agent 性格趋同.
_PERSONALITY_TO_PROMPT: dict[str, str] = {
    "lively": "liveliness",       # 活泼度 → E
    "rational": "rationality",    # 理性度 → T
    "emotional": "sensitivity",   # 感性度 → 100-T 同侧
    "planned": "planning",        # 计划度 → J
    "spontaneous": "spontaneity", # 随性度 → 100-J 同侧
    "creative": "imagination",    # 脑洞度 → N
    "humor": "humor",             # 幽默度 → 复合
}


def _build_prompt_args(
    *,
    name: str,
    gender: str | None,
    age: int,
    mbti: dict | None,
    personality: dict | None,
    career_template: dict | None,
) -> dict[str, Any]:
    """构造 character.generation prompt 的占位符 dict.

    所有字段必须非空, 否则 LLM 输出可能漂移 (空字符串 LLM 自由发挥)。
    """
    gender_zh = "男" if gender == "male" else "女"
    mbti = mbti or {}
    ei = int(mbti.get("EI", 50))
    ns = int(mbti.get("NS", 50))
    tf = int(mbti.get("TF", 50))
    jp = int(mbti.get("JP", 50))
    mbti_type = mbti.get("type") or ""
    mbti_summary = format_mbti_for_prompt(mbti) or f"MBTI {mbti_type}"

    # 7 维直接从 PersonalityInput 取; spec §1.2 7 维和 MBTI 4 轴是独立维度
    p = personality or {}
    seven_dim = {
        prompt_key: int(p.get(form_key, 50))
        for form_key, prompt_key in _PERSONALITY_TO_PROMPT.items()
    }

    career = career_template or {}
    # _split_clients 把 list / 顿号字符串 / 多分隔符 统一成 list, 再 join 给 prompt
    career_clients = "、".join(
        _split_clients(career.get("clients") or career.get("clientList"))
    )

    return {
        "name": name,
        "gender_zh": gender_zh,
        "age": age,
        "mbti_type": mbti_type,
        "mbti_summary": mbti_summary,
        "ei": ei, "inv_ei": 100 - ei,
        "ns": ns, "inv_ns": 100 - ns,
        "tf": tf, "inv_tf": 100 - tf,
        "jp": jp, "inv_jp": 100 - jp,
        **seven_dim,
        "career_title": career.get("title") or "自由职业者",
        "career_duties": career.get("duties") or "",
        "career_clients": career_clients,
        "career_income": career.get("income") or "年薪 5-10 万",
        "career_social_value": career.get("socialValue") or career.get("social_value") or "",
    }


async def generate_full_profile(
    *,
    name: str,
    gender: str | None,
    mbti: dict | None,
    personality: dict | None,
    career_template: dict | None,
    age: int | None = None,
) -> dict:
    """spec §1.4 单步 LLM 生成 character profile JSON.

    输入用户在 agent 创建表单填的姓名/性别 + 系统派生的年龄/MBTI + career_template
    池随机抽的职业, 一次 LLM 调用产出全 5 维 L1 字段。

    流程:
    1. 拉 prompt template (registry: character.generation)
    2. 注入占位符 (mbti 4 维 + 7 维 + career)
    3. invoke_json(profile="background") (max_retries=2, timeout 120s)
       - 内部 _extract_json salvage 处理截断输出
    4. _detect_missing_fields → 缺关键字段 → 一次 _repair_missing_fields
    5. _apply_postprocess_overrides 兜底:
       - identity.name = name (硬覆盖)
       - identity.gender = 中文男/女 (硬覆盖)
       - identity.ethnicity = 汉族
       - identity.blood_type = 4 选 1
       - career = career_template 数据 (硬覆盖)
    6. age 由 birthday 反推 + clamp_agent_age 钳到 20-29

    Raises:
        Exception: LLM 调用最终失败 (caller 应 set_progress("failed"))
    """
    if age is None:
        age = random.randint(AGE_MIN, AGE_MAX)

    template = await get_prompt_text("character.generation")
    args = _build_prompt_args(
        name=name, gender=gender, age=age, mbti=mbti,
        personality=personality, career_template=career_template,
    )
    prompt = template.format(**args)

    model = get_chat_model()
    profile = await invoke_json(model, prompt, profile="background")
    if not isinstance(profile, dict):
        raise ValueError(f"character.generation LLM returned non-dict: {type(profile).__name__}")

    # 兜底: 关键字段缺失走一次 repair
    missing = _detect_missing_fields(GENERATION_SCHEMA, profile)
    if missing:
        logger.warning(
            f"character.generation produced {len(missing)} missing fields, running repair pass: "
            f"{[(c, f.get('key')) for c, f in missing[:5]]}{'...' if len(missing) > 5 else ''}"
        )
        repaired = await _repair_missing_fields(GENERATION_SCHEMA, profile, missing, model)
        for cat_key, fields in repaired.items():
            profile.setdefault(cat_key, {}).update(fields)

    # 强制覆盖性别 (中文) — 在 postprocess 之前, 让后续逻辑读到统一格式
    profile.setdefault("identity", {})
    profile["identity"]["gender"] = "男" if gender == "male" else "女"

    # 由 birthday 反推 age 钳到 20-29. 若 birthday 超出 spec 区间, 同时改写
    # birthday 年份让 (birthday, age) 一致 — 否则 convert 会同时写
    # 「我生日是 1995-01-01」+「我今年 29 岁」造成自相矛盾的 L1 记忆.
    identity = profile["identity"]
    derived_age = age
    parsed_bd: date | None = None
    bd_str = identity.get("birthday")
    if isinstance(bd_str, str) and bd_str.strip():
        try:
            parsed_bd = date.fromisoformat(bd_str.strip())
            today = date.today()
            derived_age = today.year - parsed_bd.year - (
                (today.month, today.day) < (parsed_bd.month, parsed_bd.day)
            )
        except (ValueError, TypeError):
            parsed_bd = None
    else:
        # birthday 缺时尊重 LLM 自填的 age, 兜底 fallback 到入参
        try:
            derived_age = int(identity.get("age") or age)
        except (ValueError, TypeError):
            pass
    final_age = clamp_agent_age(derived_age)
    identity["age"] = final_age
    if final_age != derived_age:
        # clamp 改了 age → 同步改 birthday 年份, 月日不变 (若可解析)
        today = date.today()
        new_year = today.year - final_age
        if parsed_bd is not None:
            try:
                identity["birthday"] = parsed_bd.replace(year=new_year).isoformat()
            except ValueError:
                # parsed_bd 是 2/29 在非闰年时 raise; 退化到 2/28
                identity["birthday"] = parsed_bd.replace(year=new_year, day=28).isoformat()
        else:
            identity["birthday"] = f"{new_year}-01-01"

    # 后处理硬覆盖 (ethnicity / blood_type / name / career)
    profile = _apply_postprocess_overrides(profile, agent_name=name, career=career_template)

    return profile
