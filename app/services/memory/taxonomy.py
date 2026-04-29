"""Canonical memory taxonomy.

Per-(owner, level) allowed categories. The product spec splits by both:

- owner: memories_user vs memories_ai
- level: L1 (core facts) / L2 (consolidated) / L3 (long-term abstractions)

Design intent:
- L1 is the full semantic set (姓名/年龄/亲属关系/…). Specific facts.
- L2 collapses身份/偏好 into a small set + adds "变化" to track state deltas
  ("user changed jobs"). 生活/情绪/思维 stay rich.
- L3 further shrinks身份; adds "闲聊" to 生活 for low-value chat memories.
- AI at L2/L3 forbids身份/偏好/思维 entirely — the companion's core persona
  is locked at L1 and never drifts into fuzzy L2/L3 forms. AI L1 uniquely
  allows 生活/交互 (interaction memories with the user).

Back-compat: the flat `TAXONOMY` dict (= user L1) is still exported so
callers that don't care about level/owner keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Source = Literal["user", "ai"]

LEGACY_TYPE_TO_MAIN_CATEGORY = {
    "identity": "身份",
    "preference": "偏好",
    "life": "生活",
    "emotion": "情绪",
    "thought": "思维",
    "consolidated": "生活",
}

MAIN_CATEGORY_TO_LEGACY_TYPE = {
    "身份": "identity",
    "偏好": "preference",
    "生活": "life",
    "情绪": "emotion",
    "思维": "thought",
}

# Legacy quota — retained for portrait generation's priority ordering.
# NOT used for prompt injection (spec §3 uses retrieval, not quotas).
L1_CATEGORY_QUOTAS: tuple[tuple[str, int], ...] = (
    ("身份", 20),
    ("偏好", 10),
    ("思维", 8),
    ("生活", 10),
    ("情绪", 6),
)

PROMOTION_RULES: dict[tuple[str, str], dict[str, float | int | bool]] = {
    ("身份", "姓名"): {"allow_l1": True, "min_importance": 0.8, "min_mentions": 1},
    ("身份", "年龄"): {"allow_l1": True, "min_importance": 0.8, "min_mentions": 1},
    ("身份", "性别"): {"allow_l1": True, "min_importance": 0.8, "min_mentions": 1},
    ("身份", "生日"): {"allow_l1": True, "min_importance": 0.85, "min_mentions": 1},
    ("身份", "教育背景"): {"allow_l1": True, "min_importance": 0.85, "min_mentions": 2},
    ("身份", "职业/与经济"): {"allow_l1": True, "min_importance": 0.85, "min_mentions": 2},
    ("身份", "亲属关系"): {"allow_l1": True, "min_importance": 0.85, "min_mentions": 2},
    ("身份", "社会关系"): {"allow_l1": True, "min_importance": 0.85, "min_mentions": 2},
    ("身份", "现居地"): {"allow_l1": True, "min_importance": 0.85, "min_mentions": 2},
    ("偏好", "禁忌/雷区"): {"allow_l1": True, "min_importance": 0.85, "min_mentions": 1},
    ("偏好", "生活习惯"): {"allow_l1": True, "min_importance": 0.88, "min_mentions": 3},
    ("思维", "人生观"): {"allow_l1": True, "min_importance": 0.9, "min_mentions": 2},
    ("思维", "价值观"): {"allow_l1": True, "min_importance": 0.9, "min_mentions": 2},
    ("思维", "世界观"): {"allow_l1": True, "min_importance": 0.9, "min_mentions": 2},
    ("思维", "理想与目标"): {"allow_l1": True, "min_importance": 0.88, "min_mentions": 3},
    ("思维", "自我认知"): {"allow_l1": True, "min_importance": 0.9, "min_mentions": 2},
}

# ── Initial L1 coverage policy (agent provisioning) ───────────────────
# spec §1.2: agent 创建时应保证每个 (main, sub) 子类都有 L1 记忆。
# Singleton: 只能有 1 条(事实型,多了就矛盾)。Multi: 3-5 条。
# 少数子类豁免: 刚创建时既无经历也无用户互动，硬塞只能编造。

L1_SINGLETON_SUBS: frozenset[tuple[str, str]] = frozenset({
    ("身份", "姓名"),
    ("身份", "年龄"),
    ("身份", "性别"),
    ("身份", "生日"),
    ("身份", "星座"),
    ("身份", "生肖"),
    ("身份", "血型"),
    ("身份", "民族"),
    ("身份", "出生地"),
    ("身份", "成长地"),
    ("身份", "现居地"),
})

# 豁免: agent 刚创建时不应预填; 留给聊天/时间累积。
# 注: 情绪子类不豁免 — agent 过往的"刻骨铭心"情绪事件是人格基石,
# 需要 LLM 根据 MBTI + profile 生成, 而不是留空。
L1_COVERAGE_EXEMPT: frozenset[tuple[str, str]] = frozenset({
    # AI 与当前用户的互动, 用户 0 互动时不能预填
    ("生活", "交互"),
    # 提醒类记忆是聊天时用户对 AI 说"提醒我X"产生, character.generation 不预填
    ("生活", "提醒"),
    # "其他" 类目不强制初始化, 留给后续记忆沉淀
    ("身份", "其他"),
    ("偏好", "其他"),
    ("生活", "其他"),
    ("情绪", "其他"),
    ("思维", "其他"),
})

# 条件可选: 是否生成依赖 profile 分析 (非简单的 "phase-1 have > 0")。
# 调用方使用 analyze_conditional_subs(profile) 动态决定哪些 OPTIONAL 要生成。
L1_CONDITIONAL_SUBS: frozenset[tuple[str, str]] = frozenset({
    ("身份", "宠物"),       # 依赖 profile 养宠信号
    ("生活", "宠物"),       # 同上
    ("思维", "信仰/寄托"),  # 依赖 profile 信仰信号
    ("偏好", "人际厌恶"),   # profile 若无 dislikes.habits 则跳过
})

# 情绪类记忆目标条数 — 少而深, 每类 2-3 条"刻骨铭心"的过往事件。
L1_TARGET_EMOTION: tuple[int, int] = (2, 3)
L1_TARGET_MULTI: tuple[int, int] = (3, 5)


def is_singleton(main: str | None, sub: str | None) -> bool:
    """spec §1.4: 该 (main, sub) 是否为'独占'类 (姓名/年龄/生日 等), L1 永远只 1 条."""
    return (main or "", sub or "") in L1_SINGLETON_SUBS


def l1_target_count(main: str, sub: str) -> int:
    """返回 (main, sub) 在初始 L1 生成时的目标条数。

    - singleton: 1
    - EXEMPT: 0
    - 情绪 (非其他): 取 L1_TARGET_EMOTION 中值 (2-3 → 2)
    - 其他多值: 取 L1_TARGET_MULTI 中值 (3-5 → 4)
    """
    if (main, sub) in L1_COVERAGE_EXEMPT:
        return 0
    if (main, sub) in L1_SINGLETON_SUBS:
        return 1
    if main == "情绪":
        return (L1_TARGET_EMOTION[0] + L1_TARGET_EMOTION[1]) // 2
    return (L1_TARGET_MULTI[0] + L1_TARGET_MULTI[1]) // 2


def l1_min_importance(main: str, sub: str) -> float:
    """初始写入时该子类的最低 importance (对齐 PROMOTION_RULES.min_importance)。"""
    rule = PROMOTION_RULES.get((main, sub))
    if rule:
        return float(rule.get("min_importance", 0.75))
    return 0.75


_PET_KEYWORDS: tuple[str, ...] = (
    # 泛指 / 动作
    "宠物", "养", "喂", "遛",
    # 常见物种
    "狗", "猫", "喵", "犬", "兔", "仓鼠", "龟", "鹦鹉", "锦鲤", "猫咪",
    # 常见品种 (family/quirks 文本里容易出现)
    "拉布拉多", "金毛", "泰迪", "柯基", "哈士奇", "柴犬", "边牧",
    "布偶", "英短", "美短", "暹罗", "橘猫", "狸花",
)

_FAITH_KEYWORDS: tuple[str, ...] = (
    "信仰", "宗教", "神", "佛", "道", "基督", "天主",
    "祷告", "冥想", "因果", "菩萨", "耶稣", "真主",
    "信奉", "皈依", "灵性",
)

_INTERPERSONAL_DISLIKE_SIGNALS: tuple[str, ...] = (
    "虚伪", "背后", "说谎", "打断", "喧哗", "打扰",
    "抢功", "邀功", "推卸", "拖延", "敷衍",
)


def as_dict(container: dict, key: str) -> dict:
    """Return `container[key]` if it's a dict, else `{}`. Defensive accessor
    for profile traversal where sections may be missing or malformed."""
    v = container.get(key)
    return v if isinstance(v, dict) else {}


def analyze_conditional_subs(profile_data: dict) -> set[tuple[str, str]]:
    """Decide which L1_CONDITIONAL_SUBS to include in the initial gap plan.

    启发式分析 profile, 返回应该生成的 conditional 子类集合。无信号则跳过
    (不编造)。调用方把结果并入 gap 计划, 让 LLM 生成。
    """
    include: set[tuple[str, str]] = set()

    identity = as_dict(profile_data, "identity")
    likes = as_dict(profile_data, "likes")
    fears = as_dict(profile_data, "fears")
    values = as_dict(profile_data, "values")
    dislikes = as_dict(profile_data, "dislikes")

    # 宠物: 只看不矛盾的正向信号。likes.animals 中被 fears 列为害怕的动物先剔除,
    # 避免 "喜欢猫" 和 "怕猫" 同时出现时误判。
    la = likes.get("animals") if isinstance(likes.get("animals"), list) else []
    fa = fears.get("animals") if isinstance(fears.get("animals"), list) else []
    liked_animals = set(la or [])
    feared_animals = set(fa or [])
    non_conflicting_pets = liked_animals - feared_animals
    positive_pet_text = " ".join([
        str(identity.get("family") or ""),
        " ".join(sorted(non_conflicting_pets)),
        str(likes.get("quirks") or ""),
    ])
    pet_keyword_hit = any(kw in positive_pet_text for kw in _PET_KEYWORDS)
    if pet_keyword_hit or non_conflicting_pets:
        include.add(("身份", "宠物"))
        include.add(("生活", "宠物"))

    # 信仰: motto / believes / family 中出现宗教/灵性关键词
    faith_text = " ".join([
        str(values.get("motto") or ""),
        " ".join(values.get("believes") or []) if isinstance(values.get("believes"), list) else "",
        str(identity.get("family") or ""),
    ])
    if any(kw in faith_text for kw in _FAITH_KEYWORDS):
        include.add(("思维", "信仰/寄托"))

    # 人际厌恶: dislikes.habits 有内容, 或 abilities/values 含人际负面信号
    if dislikes.get("habits"):
        include.add(("偏好", "人际厌恶"))
    else:
        interp_text = " ".join([
            " ".join(values.get("opposes") or []) if isinstance(values.get("opposes"), list) else "",
            str(values.get("motto") or ""),
        ])
        if any(kw in interp_text for kw in _INTERPERSONAL_DISLIKE_SIGNALS):
            include.add(("偏好", "人际厌恶"))

    return include

# ── Full sub-category sets (used to compose the L1/L2/L3 slices below) ──

_IDENTITY_FULL: tuple[str, ...] = (
    "姓名", "年龄", "性别", "生日", "星座", "生肖", "血型", "民族",
    "出生地", "成长地", "现居地", "外貌特征",
    "教育背景", "职业/与经济", "亲属关系", "社会关系", "宠物", "其他",
)

_PREFERENCE_L1: tuple[str, ...] = (
    "饮食喜好", "饮食厌恶", "审美爱好", "审美厌恶",
    "人际喜好", "人际厌恶", "生活习惯", "禁忌/雷区", "其他",
)

# L2/L3 偏好 drops "禁忌/雷区" and adds "变化" (preference shifts over time).
_PREFERENCE_L23: tuple[str, ...] = (
    "变化",
    "饮食喜好", "饮食厌恶", "审美爱好", "审美厌恶",
    "人际喜好", "人际厌恶", "生活习惯", "其他",
)

_LIFE_BASE: tuple[str, ...] = (
    "教育", "工作", "旅行", "居住", "健康", "宠物",
    "人际", "技能", "生活", "提醒",
    # spec §1.4 明列，独立于通用兜底 "其他" —
    # 表达"重要但不属于具体类别的特殊经历/事件"
    "其他特殊事件",
    "其他",
)

_EMOTION_FULL: tuple[str, ...] = (
    "高兴", "悲伤", "愤怒", "恐惧", "厌恶", "焦虑",
    "失望", "自豪", "感动", "尴尬", "遗憾", "孤独",
    "惊讶", "感激", "释怀", "其他",
)

_THOUGHT_L1: tuple[str, ...] = (
    "人生观", "价值观", "世界观", "理想与目标",
    "人际关系观", "社会观点", "自我认知", "信仰/寄托", "其他",
)

# L2/L3 思维 与 L1 共用同一份 schema. 之前 strict 8 项 (无"其他") 是工程层
# over-spec 引入的洁癖, spec 没要求 L2/L3 思维必须是成熟意识形态; 实际后果
# 是同一份 LLM 输出 L1 能进、importance 落 L3 时被静默拒收, 数据丢失.
_THOUGHT_L23: tuple[str, ...] = _THOUGHT_L1

# Per-(owner, level) matrix.
# An EMPTY tuple means memories of that (owner, level, main) are NOT allowed
# at all — e.g. AI 身份 at L2 must either stay at L1 or be archived.
TAXONOMY_MATRIX: dict[Source, dict[int, dict[str, tuple[str, ...]]]] = {
    "user": {
        1: {
            "身份": _IDENTITY_FULL,
            "偏好": _PREFERENCE_L1,
            "生活": _LIFE_BASE,
            "情绪": _EMOTION_FULL,
            "思维": _THOUGHT_L1,
        },
        2: {
            # L2 身份 keeps 其他 per spec (3 items).
            "身份": ("社会关系", "变化", "其他"),
            "偏好": _PREFERENCE_L23,
            "生活": _LIFE_BASE,
            "情绪": _EMOTION_FULL,
            # L2 思维 strict — no 其他 per spec.
            "思维": _THOUGHT_L23,
        },
        3: {
            # L3 身份: 加 "其他" 兜底, 防止 L1 降级到 L3 时 sub_category 不在
            # ("社会关系", "变化") 二选一里被静默拒收. spec 没要求 L3 身份必须
            # 是这两类之一.
            "身份": ("社会关系", "变化", "其他"),
            "偏好": _PREFERENCE_L23,
            "生活": ("闲聊",) + _LIFE_BASE,
            "情绪": _EMOTION_FULL,
            "思维": _THOUGHT_L23,
        },
    },
    "ai": {
        1: {
            "身份": _IDENTITY_FULL,
            "偏好": _PREFERENCE_L1,
            # AI L1 生活 uniquely has "交互" (interactions with the user).
            "生活": _LIFE_BASE + ("交互",),
            "情绪": _EMOTION_FULL,
            "思维": _THOUGHT_L1,
        },
        2: {
            "身份": (),   # locked — AI persona facts only live at L1
            "偏好": (),
            "生活": _LIFE_BASE,
            "情绪": _EMOTION_FULL,
            "思维": (),
        },
        3: {
            "身份": (),
            "偏好": (),
            "生活": ("闲聊",) + _LIFE_BASE,
            "情绪": _EMOTION_FULL,
            "思维": (),
        },
    },
}

# Flat backward-compat view: the user L1 superset.
# Callers that don't care about (owner, level) continue to import this.
TAXONOMY: dict[str, tuple[str, ...]] = TAXONOMY_MATRIX["user"][1]


@dataclass(frozen=True)
class TaxonomyResult:
    main_category: str
    sub_category: str
    legacy_type: str | None
    # Whether (main, sub) is permitted at the requested (source, level).
    # False means the caller should refuse to write or archive instead
    # (e.g. demoting an AI 身份 memory from L1 to L2 — not allowed).
    allowed: bool = True


def allowed_main_categories(source: Source = "user", level: int = 1) -> tuple[str, ...]:
    """Main categories that actually allow memories at (source, level).

    Mains with an empty sub-cat set (e.g. AI 身份/偏好/思维 at L2/L3) are
    excluded — the spec forbids memories there, so returning them would
    mislead callers that iterate to build UIs or validators.
    """
    return tuple(
        main for main, subs in TAXONOMY_MATRIX[source][level].items() if subs
    )


def allowed_sub_categories(
    main_category: str,
    source: Source = "user",
    level: int = 1,
) -> tuple[str, ...]:
    """Allowed sub-categories for (source, level, main_category).

    Defaults to (user, L1) which is the full set — keeps back-compat with
    callers that only pass main_category.
    """
    return TAXONOMY_MATRIX.get(source, {}).get(level, {}).get(main_category, ())


def is_allowed_at(
    source: Source,
    level: int,
    main_category: str,
    sub_category: str | None = None,
) -> bool:
    """Is (main_category, sub_category) permitted at (source, level)?

    When sub_category is None, only main_category presence is checked.
    """
    subs = allowed_sub_categories(main_category, source=source, level=level)
    if not subs:
        return False
    if sub_category is None:
        return True
    return sub_category in subs


def l1_category_quotas() -> tuple[tuple[str, int], ...]:
    return L1_CATEGORY_QUOTAS


def conflict_candidate_scope(main_category: str | None, sub_category: str | None) -> dict:
    normalized_main = (main_category or "").strip()
    normalized_sub = (sub_category or "").strip()

    if normalized_main == "情绪":
        return {"should_check": False}

    if normalized_main == "偏好":
        return {
            "should_check": True,
            "prefer_same_sub_category": True,
            "default_resolution": "demote_old",
        }

    if normalized_main == "身份":
        return {
            "should_check": True,
            "prefer_same_sub_category": True,
            "default_resolution": "update_l1",
        }

    if normalized_main == "思维":
        return {
            "should_check": True,
            "prefer_same_sub_category": bool(normalized_sub),
            "default_resolution": "update_l1",
        }

    return {
        "should_check": bool(normalized_sub),
        "prefer_same_sub_category": bool(normalized_sub),
        "default_resolution": "ignore",
    }


def summarize_batch_taxonomy(items: list) -> tuple[str, str]:
    if not items:
        return ("生活", "其他")

    counts: dict[tuple[str, str], int] = {}
    for item in items:
        main_category = getattr(item, "mainCategory", None) or "生活"
        sub_category = getattr(item, "subCategory", None) or "其他"
        key = (main_category, sub_category)
        counts[key] = counts.get(key, 0) + 1

    return max(counts.items(), key=lambda pair: pair[1])[0]


SUBCATEGORY_ALIASES: dict[str, str] = {
    # ── Renamed canonical names (back-compat with pre-Phase-1 data) ──
    "相貌": "外貌特征",
    "信仰": "信仰/寄托",
    "日常生活": "生活",   # prior local name; spec字面是"生活"
    # ── 身份 ──
    # 宠物
    "猫": "宠物",
    "狗": "宠物",
    "猫咪": "宠物",
    "小猫": "宠物",
    "猫猫": "宠物",
    "猫的名字": "宠物",
    "养猫": "宠物",
    "养狗": "宠物",
    "养宠物": "宠物",
    "养宠": "宠物",
    "狗狗": "宠物",
    "狗子": "宠物",
    "小狗": "宠物",
    "喵星人": "宠物",
    "汪星人": "宠物",
    "宠物信息": "宠物",
    "宠物名": "宠物",
    "宠物名字": "宠物",
    "毛孩子": "宠物",
    "铲屎官": "宠物",
    "仓鼠": "宠物",
    "兔子": "宠物",
    "金鱼": "宠物",
    "鹦鹉": "宠物",
    "乌龟": "宠物",
    "遛狗": "宠物",
    "喂猫": "宠物",
    "喂鱼": "宠物",
    # 职业/与经济
    "工作": "职业/与经济",
    "职业": "职业/与经济",
    "赚钱": "职业/与经济",
    "薪水": "职业/与经济",
    "财务": "职业/与经济",
    "上班": "职业/与经济",
    "打工": "职业/与经济",
    "收入": "职业/与经济",
    "工资": "职业/与经济",
    "程序员": "职业/与经济",
    "老师": "职业/与经济",
    "医生": "职业/与经济",
    "创业": "职业/与经济",
    "副业": "职业/与经济",
    "失业": "职业/与经济",
    "离职": "职业/与经济",
    "跳槽": "职业/与经济",
    "求职": "职业/与经济",
    "面试": "职业/与经济",
    # 亲属关系
    "亲人": "亲属关系",
    "家人": "亲属关系",
    "爸爸": "亲属关系",
    "妈妈": "亲属关系",
    "父亲": "亲属关系",
    "母亲": "亲属关系",
    "哥哥": "亲属关系",
    "姐姐": "亲属关系",
    "弟弟": "亲属关系",
    "妹妹": "亲属关系",
    "爷爷": "亲属关系",
    "奶奶": "亲属关系",
    "外公": "亲属关系",
    "外婆": "亲属关系",
    "老公": "亲属关系",
    "老婆": "亲属关系",
    "丈夫": "亲属关系",
    "妻子": "亲属关系",
    "儿子": "亲属关系",
    "女儿": "亲属关系",
    "孩子": "亲属关系",
    # 社会关系
    "朋友": "社会关系",
    "同学": "社会关系",
    "同事": "社会关系",
    "闺蜜": "社会关系",
    "室友": "社会关系",
    "男朋友": "社会关系",
    "女朋友": "社会关系",
    "对象": "社会关系",
    "前任": "社会关系",
    "恋人": "社会关系",
    # 教育背景
    "学校": "教育背景",
    "大学": "教育背景",
    "专业": "教育背景",
    "学历": "教育背景",
    "研究生": "教育背景",
    "本科": "教育背景",
    "高中": "教育背景",
    # ── 偏好 ──
    "喜欢吃": "饮食喜好",
    "爱喝": "饮食喜好",
    "爱吃": "饮食喜好",
    "口味": "饮食喜好",
    "讨厌吃": "饮食厌恶",
    "忌口": "饮食厌恶",
    "不能吃": "饮食厌恶",
    "过敏": "饮食厌恶",
    "雷区": "禁忌/雷区",
    "早睡": "生活习惯",
    "熬夜": "生活习惯",
    "作息": "生活习惯",
    # ── 生活 ──
    "出差": "旅行",
    "度假": "旅行",
    "旅游": "旅行",
    "出国": "旅行",
    "生病": "健康",
    "吃药": "健康",
    "医院": "健康",
    "体检": "健康",
    "感冒": "健康",
    "发烧": "健康",
    "住院": "健康",
    "搬家": "居住",
    "租房": "居住",
    "买房": "居住",
    "装修": "居住",
    "考试": "教育",
    "上课": "教育",
    "学习": "教育",
    "加班": "工作",
    "开会": "工作",
    # Part 5 §4.2 提醒子类关键词
    "提醒我": "提醒",
    "提醒一下": "提醒",
    "别忘了": "提醒",
    "不要忘": "提醒",
    "记得": "提醒",
    "帮我记": "提醒",
    "帮我记住": "提醒",
    "帮我记一下": "提醒",
    "到时候提醒": "提醒",
    "定个闹钟": "提醒",
    "备忘": "提醒",
    # spec §1.4 "其他特殊事件" — 重要但不属于上述具体子类的事件
    "特殊事件": "其他特殊事件",
    "特殊经历": "其他特殊事件",
    "重要事件": "其他特殊事件",
    "难忘经历": "其他特殊事件",
}


# Pre-sorted alias keys by length descending for contains matching.
# Longer keys first so "铲屎官" is checked before "猫".
_ALIAS_KEYS_BY_LENGTH: list[str] = sorted(
    SUBCATEGORY_ALIASES.keys(), key=len, reverse=True
)


def _resolve_by_contains(text: str, allowed: tuple[str, ...]) -> str | None:
    """Scan *text* for any known alias key or canonical sub-category name
    that is present in *allowed*. Returns the matched sub-category or None.
    """
    if not text or not allowed:
        return None

    # Canonical names first (e.g. text="关于宠物" contains "宠物")
    for name in allowed:
        if name != "其他" and name in text:
            return name

    # Alias keys (longer first)
    for key in _ALIAS_KEYS_BY_LENGTH:
        if key in text:
            mapped = SUBCATEGORY_ALIASES[key]
            if mapped in allowed:
                return mapped

    return None


def resolve_taxonomy(
    *,
    main_category: str | None = None,
    sub_category: str | None = None,
    legacy_type: str | None = None,
    source: Source = "user",
    level: int = 1,
) -> TaxonomyResult:
    """Normalize a (main, sub) pair against the (source, level) allowed set.

    Backward-compat: source/level default to (user, L1), the full superset,
    so callers that don't pass them behave exactly as before.

    The returned `allowed` flag is False only when the target (source, level)
    forbids this main_category entirely (e.g. AI 身份/偏好/思维 at L2/L3).
    In that case main_category is preserved — the caller decides whether to
    archive, reject, or keep the memory at a different level.
    """
    raw_main = (main_category or "").strip()
    normalized_sub = (sub_category or "").strip()
    normalized_legacy = (legacy_type or "").strip() or None

    # Resolve main: explicit first, then legacy type, else fallback to 生活.
    # If the caller gave a non-empty but UNRECOGNIZED main (e.g. the LLM
    # hallucinated "某乱"), we refuse rather than silently dumping it into
    # 生活/其他 — that would pollute retrieval with mis-categorized rows.
    if raw_main in TAXONOMY_MATRIX["user"][1]:
        normalized_main = raw_main
    elif not raw_main and normalized_legacy:
        normalized_main = LEGACY_TYPE_TO_MAIN_CATEGORY.get(normalized_legacy, "生活")
        if normalized_main not in TAXONOMY_MATRIX["user"][1]:
            normalized_main = "生活"
    elif not raw_main:
        normalized_main = "生活"
    else:
        # Unrecognized non-empty main → refuse.
        return TaxonomyResult(
            main_category=raw_main,
            sub_category=normalized_sub or "其他",
            legacy_type=normalized_legacy,
            allowed=False,
        )

    legacy = MAIN_CATEGORY_TO_LEGACY_TYPE.get(normalized_main, normalized_legacy)

    allowed_subs = allowed_sub_categories(normalized_main, source=source, level=level)
    if not allowed_subs:
        # Main category is forbidden at this (source, level). Keep the
        # original sub text so the caller can see what the LLM produced,
        # but mark as not-allowed for the write path to decide.
        return TaxonomyResult(
            main_category=normalized_main,
            sub_category=normalized_sub or "其他",
            legacy_type=legacy,
            allowed=False,
        )

    # Step 1: Direct match in the allowed set
    if normalized_sub in allowed_subs:
        return TaxonomyResult(normalized_main, normalized_sub, legacy)

    # Step 2: Exact alias mapping (e.g. "妈妈" -> "亲属关系")
    mapped = SUBCATEGORY_ALIASES.get(normalized_sub)
    if mapped and mapped in allowed_subs:
        return TaxonomyResult(normalized_main, mapped, legacy)

    # Step 3: Contains matching (text contains a canonical or alias key)
    contains_hit = _resolve_by_contains(normalized_sub, allowed_subs)
    if contains_hit:
        return TaxonomyResult(normalized_main, contains_hit, legacy)

    # Step 4: Fuzzy prefix match (e.g. "宠物类" -> "宠物")
    for allowed in allowed_subs:
        if (
            normalized_sub
            and allowed != "其他"
            and (normalized_sub.startswith(allowed) or allowed.startswith(normalized_sub))
        ):
            return TaxonomyResult(normalized_main, allowed, legacy)

    # Step 5: Safe fallback or refusal
    # - If "其他" is in the allowed set, use it as the catch-all.
    # - Otherwise the spec deliberately omitted "其他" for this
    #   (source, level, main) — e.g. user L2/L3 思维, user L3 身份 — so
    #   an unresolvable sub shouldn't be silently collapsed into the first
    #   alphabetical slot. Return allowed=False and let the caller decide.
    if "其他" in allowed_subs:
        return TaxonomyResult(normalized_main, "其他", legacy)
    return TaxonomyResult(
        normalized_main,
        normalized_sub or allowed_subs[0],
        legacy,
        allowed=False,
    )
