from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException


@dataclass(frozen=True)
class AgentAvatar:
    key: str
    url: str


_MALE_AVATAR_NUMBERS = (*range(1, 14), 15, 16, *range(18, 24), 25, 26, 27)
_FEMALE_AVATAR_NUMBERS = (*range(1, 21), 22)

MALE_AVATAR_KEYS = tuple(
    f"companion-male-{index:02d}" for index in _MALE_AVATAR_NUMBERS
)
FEMALE_AVATAR_KEYS = tuple(
    f"companion-female-{index:02d}" for index in _FEMALE_AVATAR_NUMBERS
)

_ALL_AVATAR_KEYS = frozenset(MALE_AVATAR_KEYS + FEMALE_AVATAR_KEYS)
_AVATAR_DIR = Path(__file__).resolve().parents[1] / "assets" / "agent_avatars"
_AVATAR_PUBLIC_PREFIX = (
    os.getenv("AGENT_AVATAR_PUBLIC_PREFIX", "/agents/avatar").strip().rstrip("/")
    or "/agents/avatar"
)
_AVATAR_KEY_RE = re.compile(r"^companion-(?:male|female)-\d{2}$")


def build_avatar_url(key: str | None) -> str | None:
    if not key:
        return None
    return f"{_AVATAR_PUBLIC_PREFIX}/{_validate_avatar_key(key)}.png"


def pick_agent_avatar(gender: str | None) -> AgentAvatar:
    key = random.choice(_pool_for_gender(gender))
    return AgentAvatar(key=key, url=build_avatar_url(key) or "")


def avatar_keys_for_gender(gender: str | None = None) -> tuple[str, ...]:
    return _pool_for_gender(gender)


def get_avatar_path(key: str) -> Path:
    safe_key = _validate_avatar_key(key)
    path = _AVATAR_DIR / f"{safe_key}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Avatar not found")
    return path


def validate_avatar_assets() -> None:
    missing = [key for key in _ALL_AVATAR_KEYS if not (_AVATAR_DIR / f"{key}.png").is_file()]
    if missing:
        raise RuntimeError(f"Missing agent avatar assets: {', '.join(sorted(missing))}")


def _pool_for_gender(gender: str | None) -> tuple[str, ...]:
    normalized = (gender or "").strip().lower()
    if normalized == "male":
        return MALE_AVATAR_KEYS
    if normalized == "female":
        return FEMALE_AVATAR_KEYS
    return MALE_AVATAR_KEYS + FEMALE_AVATAR_KEYS


def _validate_avatar_key(key: str) -> str:
    normalized = key.strip()
    if not _AVATAR_KEY_RE.fullmatch(normalized) or normalized not in _ALL_AVATAR_KEYS:
        raise HTTPException(status_code=404, detail="Avatar not found")
    return normalized
