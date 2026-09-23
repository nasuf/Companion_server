"""Safety helpers for hidden offline-activity targets.

Recognition may use target names. User-visible prompts and messages may only use
validated abstract cues from ``guidance_profile``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_GENERIC_GUIDANCE = {
    "weak": "不妨留意一下周围那些容易被忽略的小细节。",
    "medium": "换个角度看看附近的颜色、轮廓和明暗变化，也许会有新发现。",
    "strong": "可以停一下，看看视线之外或高低不同的位置有没有特别的画面。",
}

_CATEGORY_GUIDANCE: tuple[tuple[tuple[str, ...], dict[str, str]], ...] = (
    (
        ("植物", "花", "树", "草"),
        {
            "weak": "附近的颜色和细小纹理好像也值得多看两眼。",
            "medium": "可以留意有生命感、颜色比较鲜明的角落。",
            "strong": "往自然生长的细节靠近一点，看看形状和颜色的变化。",
        },
    ),
    (
        ("天空", "云"),
        {
            "weak": "偶尔把视线放远一点，现场也许会换一种感觉。",
            "medium": "不妨看看更高、更开阔的方向，光线可能很不一样。",
            "strong": "试着抬高视线，找找远处明暗和层次的变化。",
        },
    ),
    (
        ("光影", "倒影"),
        {
            "weak": "这里的明暗变化也许藏着挺有意思的画面。",
            "medium": "换个方向看看亮处和暗处交界的地方。",
            "strong": "可以找找被光线切开的轮廓，或者映出来的另一层画面。",
        },
    ),
    (
        ("建筑", "街景"),
        {
            "weak": "周围的线条和空间层次好像也挺耐看的。",
            "medium": "试着注意一下门窗、转角或重复出现的轮廓。",
            "strong": "沿着有结构感的线条看过去，也许能找到一个特别的角度。",
        },
    ),
    (
        ("水", "湖泊", "河流"),
        {
            "weak": "有些会流动或映出周围的画面，也挺容易让人停下来。",
            "medium": "可以听听附近的声音，再看看哪里有轻微的波纹和反光。",
            "strong": "找找会映出周围颜色、又一直在变化的地方。",
        },
    ),
    (
        ("动物", "鸟", "猫", "狗"),
        {
            "weak": "周围如果有突然的小动静，别急着走过去。",
            "medium": "可以留意那些自己会移动、偶尔停下来的身影。",
            "strong": "听到细碎动静时往附近看看，也许正好会有小家伙经过。",
        },
    ),
)


def _clean_terms(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = re.sub(r"\s+", "", str(value or "")).strip().lower()
        if len(text) >= 1 and text not in out:
            out.append(text)
    return out


def _criteria_aliases(criteria: Any) -> list[str]:
    text = str(criteria or "").strip()
    if not text:
        return []
    text = re.sub(
        r"(照片|画面|主体|清楚|清晰|明显|呈现|出现|主要|中心|对象|满足|可见|是|的)",
        " ",
        text,
    )
    return _clean_terms(re.split(r"[\s、，,；;/]|或者|或|以及|和|及", text))


def forbidden_terms(
    condition: dict[str, Any],
    *,
    include_category: bool = False,
) -> list[str]:
    profile = condition.get("guidance_profile")
    aliases = profile.get("aliases") if isinstance(profile, dict) else []
    if not isinstance(aliases, list):
        aliases = []
    return _clean_terms(
        [
            condition.get("short_name"),
            condition.get("category") if include_category else None,
            *aliases,
            *_criteria_aliases(condition.get("criteria")),
        ]
    )


def contains_hidden_target(
    text: str,
    conditions: Iterable[dict[str, Any]],
) -> bool:
    compact = re.sub(r"\s+", "", str(text or "")).lower()
    if not compact:
        return False
    conditions = list(conditions)
    if any(
        term in compact
        for condition in conditions
        for term in forbidden_terms(condition)
    ):
        return True
    # Broad categories are only leaks when used as a direction. This avoids
    # blocking natural place names such as "植物园".
    if re.search(r"(拍|找|寻找|留意|注意|看看|观察|对准|镜头)", compact):
        return any(
            term in compact
            for condition in conditions
            for term in forbidden_terms(condition, include_category=True)
        )
    return False


def _category_fallback(short_name: str, category: str) -> dict[str, str]:
    probe = f"{category}{short_name}".lower()
    for keywords, guidance in _CATEGORY_GUIDANCE:
        if any(keyword.lower() in probe for keyword in keywords):
            return dict(guidance)
    return dict(_GENERIC_GUIDANCE)


def normalize_guidance_profile(
    *,
    short_name: str,
    category: str,
    criteria: str = "",
    raw_profile: Any,
) -> dict[str, Any]:
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    aliases_raw = profile.get("aliases")
    aliases = (
        [str(item).strip()[:30] for item in aliases_raw if str(item).strip()]
        if isinstance(aliases_raw, list)
        else []
    )
    aliases = _clean_terms([*aliases, *_criteria_aliases(criteria)])[:8]
    condition = {
        "short_name": short_name,
        "category": category,
        "criteria": criteria,
        "guidance_profile": {"aliases": aliases},
    }
    fallback = _category_fallback(short_name, category)
    raw_guidance = profile.get("guidance")
    raw_guidance = raw_guidance if isinstance(raw_guidance, dict) else {}
    guidance: dict[str, str] = {}
    for level in ("weak", "medium", "strong"):
        candidate = str(raw_guidance.get(level) or "").strip()[:160]
        compact = re.sub(r"\s+", "", candidate).lower()
        leaks_profile = any(
            term in compact
            for term in forbidden_terms(condition, include_category=True)
        )
        guidance[level] = (
            fallback[level]
            if not candidate or leaks_profile
            else candidate
        )
    return {
        "aliases": aliases,
        "guidance": guidance,
    }


def safe_guidance(condition: dict[str, Any], level: str = "weak") -> str:
    profile = condition.get("guidance_profile")
    guidance = profile.get("guidance") if isinstance(profile, dict) else None
    candidate = (
        str(guidance.get(level) or "").strip()
        if isinstance(guidance, dict)
        else ""
    )
    compact = re.sub(r"\s+", "", candidate).lower()
    leaks_profile = any(
        term in compact
        for term in forbidden_terms(condition, include_category=True)
    )
    if candidate and not leaks_profile:
        return candidate
    fallback = _category_fallback(
        str(condition.get("short_name") or ""),
        str(condition.get("category") or ""),
    )
    return fallback.get(level, _GENERIC_GUIDANCE["weak"])


def sanitize_visible_text(
    text: str,
    conditions: Iterable[dict[str, Any]],
    *,
    fallback: str,
) -> str:
    candidate = str(text or "").strip()
    if not candidate or contains_hidden_target(candidate, conditions):
        return fallback
    return candidate
