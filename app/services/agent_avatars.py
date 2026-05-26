from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentAvatar:
    key: str
    url: str


MALE_AVATAR_KEYS = (
    "bansheng-male-01",
    "bansheng-male-02",
    "bansheng-male-03",
    "bansheng-male-04",
    "bansheng-male-05",
    "bansheng-male-06",
)

FEMALE_AVATAR_KEYS = (
    "bansheng-female-01",
    "bansheng-female-02",
    "bansheng-female-03",
    "bansheng-female-04",
    "bansheng-female-05",
    "bansheng-female-06",
)

_BASE_URL = "https://api.dicebear.com/9.x/open-peeps/png"
_COMMON_QUERY = (
    "radius=50"
    "&size=128"
    "&backgroundType=gradientLinear"
    "&backgroundColor=b6e3f4,c0aede,d1d4f9,ffd5dc,ffdfbf"
    "&accessoriesProbability=20"
)
_MALE_STYLE_QUERY = (
    "&head=short1,short2,short3,short4,short5,flatTop,pomp,mohawk"
    "&facialHairProbability=12"
)
_FEMALE_STYLE_QUERY = (
    "&head=long,longBangs,longCurly,bangs,bangs2,bun,bun2,buns,mediumStraight"
    "&facialHairProbability=0"
)


def build_avatar_url(key: str) -> str:
    style_query = _MALE_STYLE_QUERY if "-male-" in key else _FEMALE_STYLE_QUERY
    return f"{_BASE_URL}?seed={key}&{_COMMON_QUERY}{style_query}"


def pick_agent_avatar(gender: str | None) -> AgentAvatar:
    pool = _pool_for_gender(gender)
    key = random.choice(pool)
    return AgentAvatar(key=key, url=build_avatar_url(key))


def _pool_for_gender(gender: str | None) -> tuple[str, ...]:
    normalized = (gender or "").strip().lower()
    if normalized == "male":
        return MALE_AVATAR_KEYS
    if normalized == "female":
        return FEMALE_AVATAR_KEYS
    return MALE_AVATAR_KEYS + FEMALE_AVATAR_KEYS
