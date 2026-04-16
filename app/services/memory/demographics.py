"""Deterministic demographic helpers for L1 coverage.

星座 / 生肖 从 birthday 派生 (无需 LLM)。血型 / 民族 按人口分布抽样
(profile schema 没这字段时兜底, 避免为此跑一次 LLM, 也比 LLM 编的更真)。
"""

from __future__ import annotations

import random
from datetime import date


_CONSTELLATIONS: tuple[tuple[int, int, str], ...] = (
    (1, 20, "摩羯座"), (2, 19, "水瓶座"), (3, 21, "双鱼座"),
    (4, 20, "白羊座"), (5, 21, "金牛座"), (6, 22, "双子座"),
    (7, 23, "巨蟹座"), (8, 23, "狮子座"), (9, 23, "处女座"),
    (10, 23, "天秤座"), (11, 23, "天蝎座"), (12, 22, "射手座"),
)

_ZODIAC_CYCLE: tuple[str, ...] = (
    "鼠", "牛", "虎", "兔", "龙", "蛇",
    "马", "羊", "猴", "鸡", "狗", "猪",
)

# 2000 年是龙年; zodiac[year] = (year - 2000 + 4) % 12 的索引到上面数组
_ZODIAC_ANCHOR_YEAR = 2000
_ZODIAC_ANCHOR_INDEX = 4  # 2000 = 龙

# 中国人口分布 (第七次全国人口普查大致数字)
_BLOOD_TYPE_DIST: tuple[tuple[str, float], ...] = (
    ("O型", 0.34), ("A型", 0.31), ("B型", 0.27), ("AB型", 0.08),
)

_ETHNICITY_DIST: tuple[tuple[str, float], ...] = (
    ("汉族", 0.915),
    ("壮族", 0.013), ("维吾尔族", 0.008), ("回族", 0.008),
    ("苗族", 0.007), ("满族", 0.007), ("彝族", 0.007),
    ("土家族", 0.006), ("藏族", 0.005), ("蒙古族", 0.004),
    ("侗族", 0.002), ("布依族", 0.002), ("瑶族", 0.002),
    ("白族", 0.001), ("朝鲜族", 0.001), ("哈尼族", 0.001),
    ("黎族", 0.001), ("其他民族", 0.010),
)


def _parse_date(raw: str | None) -> date | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw.strip())
    except (ValueError, TypeError):
        return None


def derive_constellation(birthday: str | None) -> str | None:
    d = _parse_date(birthday)
    if not d:
        return None
    for month, day, name in _CONSTELLATIONS:
        if (d.month, d.day) <= (month, day):
            return name
    return "摩羯座"


def derive_zodiac(birthday: str | None) -> str | None:
    d = _parse_date(birthday)
    if not d:
        return None
    offset = (d.year - _ZODIAC_ANCHOR_YEAR) % 12
    return _ZODIAC_CYCLE[(_ZODIAC_ANCHOR_INDEX + offset) % 12]


def _weighted_choice(pairs: tuple[tuple[str, float], ...], rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for label, weight in pairs:
        acc += weight
        if r < acc:
            return label
    return pairs[-1][0]


def _make_rng(seed: str | None) -> random.Random:
    return random.Random(seed) if seed else random.Random()


def sample_blood_type(seed: str | None = None) -> str:
    return _weighted_choice(_BLOOD_TYPE_DIST, _make_rng(seed))


def sample_ethnicity(seed: str | None = None) -> str:
    return _weighted_choice(_ETHNICITY_DIST, _make_rng(seed))
