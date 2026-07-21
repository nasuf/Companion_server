"""Runtime resolution for the global achievement mode (on / silent / off).

Source of truth is the ``SystemConfig`` singleton (id=1) column
``achievement_mode`` so admins can switch it from the web console (系统设置)
with near-immediate effect. ``NULL`` means "inherit the .env ACHIEVEMENT_MODE
default" — the same override-or-inherit semantic as the model runtime config.

Unlike the offline module flags, these gates sit on the chat hot path (every
persisted message fans out an achievement event), so reads go through a small
in-process TTL cache (~10s) instead of hitting the DB per call. Writes update
the local cache immediately; other workers converge within one TTL window.

DB failures never break gating: the resolver falls back to the env default and
skips caching so recovery is picked up on the next call.
"""

from __future__ import annotations

import logging
import time

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)

ACHIEVEMENT_MODES = ("on", "silent", "off")

_CACHE_TTL_S = 10.0
# (fetched_at_monotonic, override_or_none); None entry = "no DB override".
_override_cache: tuple[float, str | None] | None = None


def _normalize(value: object) -> str | None:
    """Return a valid lowered mode string, or None for empty/unknown values."""
    text = str(value or "").strip().lower()
    return text if text in ACHIEVEMENT_MODES else None


def evaluation_enabled_for(mode: str) -> bool:
    """Rules keep evaluating and unlocks keep persisting ("on"/"silent")."""
    return _normalize(mode) != "off"


def user_facing_enabled_for(mode: str) -> bool:
    """Notifications, APIs, timeline rows, and wallet sync ("on" only)."""
    return _normalize(mode) == "on"


def reset_achievement_mode_cache() -> None:
    global _override_cache
    _override_cache = None


async def _read_override(*, use_cache: bool = True) -> str | None:
    """Read the DB override; cache successful reads only."""
    global _override_cache
    now = time.monotonic()
    if use_cache and _override_cache and now - _override_cache[0] < _CACHE_TTL_S:
        return _override_cache[1]
    try:
        config = await db.systemconfig.find_unique(where={"id": 1})
    except Exception as e:
        # Env fallback without caching: a transient DB error must not pin a
        # stale mode for the whole TTL window.
        logger.debug(f"[ACH] mode override read skipped: {e}")
        return None
    override = _normalize(getattr(config, "achievementMode", None)) if config else None
    _override_cache = (now, override)
    return override


def _env_mode() -> str:
    return _normalize(settings.achievement_mode) or "on"


async def get_achievement_mode(*, use_cache: bool = True) -> str:
    """Effective mode: DB override when set, otherwise the .env default."""
    override = await _read_override(use_cache=use_cache)
    return override or _env_mode()


async def achievement_evaluation_enabled() -> bool:
    return evaluation_enabled_for(await get_achievement_mode())


async def achievement_user_facing_enabled() -> bool:
    return user_facing_enabled_for(await get_achievement_mode())


async def get_achievement_settings_snapshot(*, use_cache: bool = False) -> dict:
    """Admin view: override + env default + resolved effective mode."""
    override = await _read_override(use_cache=use_cache)
    return {
        "mode": override,
        "env_mode": settings.achievement_mode,
        "effective_mode": override or _env_mode(),
    }


async def set_achievement_mode(mode: str) -> dict:
    """Persist the admin override and refresh this process's cache."""
    global _override_cache
    normalized = _normalize(mode)
    if normalized is None:
        raise ValueError(f"invalid achievement mode: {mode!r}")
    await db.systemconfig.upsert(
        where={"id": 1},
        data={
            "create": {"id": 1, "achievementMode": normalized},
            "update": {"achievementMode": normalized},
        },
    )
    _override_cache = (time.monotonic(), normalized)
    logger.info(
        "achievement mode updated",
        extra={"event": "achievement_mode_updated", "mode": normalized},
    )
    return await get_achievement_settings_snapshot(use_cache=True)
