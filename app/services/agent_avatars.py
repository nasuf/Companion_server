from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import re
from dataclasses import dataclass

import httpx
from fastapi import HTTPException

from app.db import db


@dataclass(frozen=True)
class AgentAvatar:
    key: str
    url: str


@dataclass(frozen=True)
class CachedAvatar:
    key: str
    content_type: str
    image_bytes: bytes


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
_AVATAR_PUBLIC_PREFIX = (
    os.getenv("AGENT_AVATAR_PUBLIC_PREFIX", "/agents/avatar").strip().rstrip("/")
    or "/agents/avatar"
)
_AVATAR_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")
_MAX_AVATAR_BYTES = 512 * 1024
_DOWNLOAD_TIMEOUT = httpx.Timeout(8.0, connect=4.0, read=6.0, write=4.0, pool=4.0)

logger = logging.getLogger(__name__)


def build_avatar_url(key: str) -> str:
    style_query = _MALE_STYLE_QUERY if "-male-" in key else _FEMALE_STYLE_QUERY
    return f"{_BASE_URL}?seed={key}&{_COMMON_QUERY}{style_query}"


def build_cached_avatar_url(key: str | None) -> str | None:
    if not key:
        return None
    return f"{_AVATAR_PUBLIC_PREFIX}/{_validate_avatar_key(key)}.png"


def pick_agent_avatar(gender: str | None) -> AgentAvatar:
    pool = _pool_for_gender(gender)
    key = random.choice(pool)
    return AgentAvatar(key=key, url=build_cached_avatar_url(key) or build_avatar_url(key))


async def ensure_cached_avatar(key: str) -> CachedAvatar:
    safe_key = _validate_avatar_key(key)
    row = await db.agentavatarcache.find_unique(where={"key": safe_key})
    cached = _avatar_from_row(row)
    if cached is not None:
        return cached

    content_type, blob, source_url = await _download_avatar(safe_key)
    serialized_blob = _serialize_image_bytes(blob)
    try:
        row = await db.agentavatarcache.upsert(
            where={"key": safe_key},
            data={
                "create": {
                    "key": safe_key,
                    "gender": _gender_for_key(safe_key),
                    "contentType": content_type,
                    "imageBytes": serialized_blob,
                    "sourceUrl": source_url,
                },
                "update": {
                    "gender": _gender_for_key(safe_key),
                    "contentType": content_type,
                    "imageBytes": serialized_blob,
                    "sourceUrl": source_url,
                },
            },
        )
    except Exception as exc:
        logger.warning("[agent-avatar] db cache write failed key=%s error=%s", safe_key, exc)
        raise HTTPException(status_code=503, detail="Avatar cache unavailable") from exc

    cached = _avatar_from_row(row)
    return cached or CachedAvatar(
        key=safe_key,
        content_type=content_type,
        image_bytes=blob,
    )


async def warm_avatar_pool(gender: str | None = None) -> dict[str, bool]:
    semaphore = asyncio.Semaphore(4)

    async def _warm_one(key: str) -> tuple[str, bool]:
        try:
            async with semaphore:
                await ensure_cached_avatar(key)
            return key, True
        except Exception as exc:
            logger.warning("[agent-avatar] warm failed key=%s error=%s", key, exc)
            return key, False

    return dict(await asyncio.gather(*(_warm_one(key) for key in avatar_keys_for_gender(gender))))


def avatar_keys_for_gender(gender: str | None = None) -> tuple[str, ...]:
    return _pool_for_gender(gender)


async def _download_avatar(key: str) -> tuple[str, bytes, str]:
    source_url = build_avatar_url(key)
    try:
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, trust_env=False) as client:
            response = await client.get(source_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("[agent-avatar] download failed key=%s error=%s", key, exc)
        raise HTTPException(status_code=503, detail="Avatar source unavailable") from exc

    content_type = response.headers.get("content-type", "").lower()
    if "image/png" not in content_type:
        logger.warning(
            "[agent-avatar] unexpected content-type key=%s content_type=%s",
            key,
            content_type,
        )
        raise HTTPException(status_code=502, detail="Avatar source returned invalid image")

    blob = response.content
    if not blob or len(blob) > _MAX_AVATAR_BYTES:
        raise HTTPException(status_code=502, detail="Avatar source returned invalid size")
    return "image/png", blob, source_url


def _pool_for_gender(gender: str | None) -> tuple[str, ...]:
    normalized = (gender or "").strip().lower()
    if normalized == "male":
        return MALE_AVATAR_KEYS
    if normalized == "female":
        return FEMALE_AVATAR_KEYS
    return MALE_AVATAR_KEYS + FEMALE_AVATAR_KEYS


def _validate_avatar_key(key: str) -> str:
    normalized = key.strip()
    if not _AVATAR_KEY_RE.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="Invalid avatar key")
    return normalized


def _gender_for_key(key: str) -> str | None:
    if "-male-" in key:
        return "male"
    if "-female-" in key:
        return "female"
    return None


def _avatar_from_row(row) -> CachedAvatar | None:
    if row is None:
        return None
    raw = getattr(row, "imageBytes", None)
    image_bytes = _coerce_image_bytes(raw)
    if not image_bytes:
        return None
    return CachedAvatar(
        key=str(getattr(row, "key", "")),
        content_type=str(getattr(row, "contentType", None) or "image/png"),
        image_bytes=image_bytes,
    )


def _coerce_image_bytes(raw) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if isinstance(raw, str):
        try:
            return base64.b64decode(raw)
        except ValueError:
            return raw.encode("utf-8")
    return bytes(raw)


def _serialize_image_bytes(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")
