"""Canonical memory taxonomy.

This module defines the only allowed memory categories/sub-categories.
The taxonomy follows the product-defined Chinese classification exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

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

L1_CATEGORY_QUOTAS: tuple[tuple[str, int], ...] = (
    ("身份", 6),
    ("偏好", 4),
    ("思维", 4),
    ("生活", 4),
    ("情绪", 2),
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

COMPRESSION_RULES: dict[str, dict[str, int | bool]] = {
    "身份": {"batch_size": 6, "allow_cross_subcategory": False},
    "偏好": {"batch_size": 8, "allow_cross_subcategory": False},
    "生活": {"batch_size": 10, "allow_cross_subcategory": True},
    "情绪": {"batch_size": 6, "allow_cross_subcategory": False},
    "思维": {"batch_size": 6, "allow_cross_subcategory": False},
}

TAXONOMY: dict[str, tuple[str, ...]] = {
    "身份": (
        "姓名",
        "年龄",
        "性别",
        "生日",
        "星座",
        "生肖",
        "血型",
        "民族",
        "出生地",
        "成长地",
        "现居地",
        "相貌",
        "教育背景",
        "职业/与经济",
        "亲属关系",
        "社会关系",
        "宠物",
        "其他",
    ),
    "偏好": (
        "饮食喜好",
        "饮食厌恶",
        "审美爱好",
        "审美厌恶",
        "人际喜好",
        "人际厌恶",
        "生活习惯",
        "禁忌/雷区",
        "其他",
    ),
    "生活": (
        "教育",
        "工作",
        "旅行",
        "居住",
        "健康",
        "宠物",
        "人际",
        "技能",
        "日常生活",
        "其他",
    ),
    "情绪": (
        "高兴",
        "悲伤",
        "愤怒",
        "恐惧",
        "厌恶",
        "焦虑",
        "失望",
        "自豪",
        "感动",
        "尴尬",
        "遗憾",
        "孤独",
        "惊讶",
        "感激",
        "释怀",
        "其他",
    ),
    "思维": (
        "人生观",
        "价值观",
        "世界观",
        "理想与目标",
        "人际关系观",
        "社会观点",
        "自我认知",
        "信仰",
        "其他",
    ),
}


@dataclass(frozen=True)
class TaxonomyResult:
    main_category: str
    sub_category: str
    legacy_type: str | None


def allowed_main_categories() -> tuple[str, ...]:
    return tuple(TAXONOMY.keys())


def allowed_sub_categories(main_category: str) -> tuple[str, ...]:
    return TAXONOMY.get(main_category, ())


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


def get_promotion_rule(main_category: str | None, sub_category: str | None) -> dict[str, float | int | bool]:
    key = ((main_category or "").strip(), (sub_category or "").strip())
    return PROMOTION_RULES.get(
        key,
        {"allow_l1": False, "min_importance": 0.95, "min_mentions": 12},
    )


def get_compression_rule(main_category: str | None) -> dict[str, int | bool]:
    key = (main_category or "").strip()
    return COMPRESSION_RULES.get(
        key,
        {"batch_size": 10, "allow_cross_subcategory": True},
    )


def resolve_taxonomy(
    *,
    main_category: str | None = None,
    sub_category: str | None = None,
    legacy_type: str | None = None,
) -> TaxonomyResult:
    normalized_main = (main_category or "").strip()
    normalized_sub = (sub_category or "").strip()
    normalized_legacy = (legacy_type or "").strip() or None

    if not normalized_main and normalized_legacy:
        normalized_main = LEGACY_TYPE_TO_MAIN_CATEGORY.get(normalized_legacy, "")

    if normalized_main not in TAXONOMY:
        normalized_main = "生活"

    if normalized_sub not in TAXONOMY[normalized_main]:
        normalized_sub = "其他"

    return TaxonomyResult(
        main_category=normalized_main,
        sub_category=normalized_sub,
        legacy_type=MAIN_CATEGORY_TO_LEGACY_TYPE.get(normalized_main, normalized_legacy),
    )
