"""Runtime resolution for the global achievement mode (on / silent / off).

Mode semantics (2026-07-22 产品口径):
- on:     everything enabled.
- silent: evaluation + unlock persistence + achievement page + wallet points
          keep working; only the unlock *moments* are muted — chat WS popups,
          conversation-timeline achievement rows, and APNs/system push.
- off:    emergency stop — no evaluation, daily rollup frozen, page hidden.

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
# Last override that was actually read from the DB, kept beyond the TTL purely
# as a fail-safe: a transient DB error must never silently re-enable pushes
# that an admin turned off (see _read_override).
_last_known_override: str | None = None
_has_known_override = False


def _normalize(value: object) -> str | None:
    """Return a valid lowered mode string, or None for empty/unknown values."""
    text = str(value or "").strip().lower()
    return text if text in ACHIEVEMENT_MODES else None


def evaluation_enabled_for(mode: str) -> bool:
    """Rules keep evaluating and unlocks keep persisting ("on"/"silent")."""
    return _normalize(mode) != "off"


def display_enabled_for(mode: str) -> bool:
    """Achievement page API + wallet point sync ("on"/"silent").

    Silent mode only mutes the unlock *moments*; the achievement page and its
    accumulated points stay fully visible and usable (2026-07-22 产品调整).
    """
    return _normalize(mode) != "off"


def alerts_enabled_for(mode: str) -> bool:
    """Unlock alerts ("on" only): chat WS popups, conversation-timeline
    achievement rows, and APNs/system push notifications."""
    return _normalize(mode) == "on"


def reset_achievement_mode_cache() -> None:
    global _override_cache, _last_known_override, _has_known_override
    _override_cache = None
    _last_known_override = None
    _has_known_override = False


async def _read_override(*, use_cache: bool = True) -> str | None:
    """Read the DB override, degrading safely when the lookup fails.

    Failing over to the env default is NOT safe here: production leaves
    ACHIEVEMENT_MODE unset (= "on"), so a single transient DB error would
    resurrect unlock pushes an admin had switched off. Instead reuse the last
    value we actually read; only a process that never managed one falls back
    to env.
    """
    global _override_cache, _last_known_override, _has_known_override
    now = time.monotonic()
    if use_cache and _override_cache and now - _override_cache[0] < _CACHE_TTL_S:
        return _override_cache[1]
    try:
        config = await db.systemconfig.find_unique(where={"id": 1})
    except Exception as e:
        if _has_known_override:
            logger.warning(
                f"[ACH] mode override read failed, reusing last known "
                f"{_last_known_override!r}: {e}"
            )
            return _last_known_override
        logger.warning(f"[ACH] mode override read failed, using env default: {e}")
        return None
    override = _normalize(getattr(config, "achievementMode", None)) if config else None
    _override_cache = (now, override)
    _last_known_override = override
    _has_known_override = True
    return override


def _env_mode() -> str:
    return _normalize(settings.achievement_mode) or "on"


async def get_achievement_mode(*, use_cache: bool = True) -> str:
    """Effective mode: DB override when set, otherwise the .env default."""
    override = await _read_override(use_cache=use_cache)
    return override or _env_mode()


async def achievement_evaluation_enabled() -> bool:
    return evaluation_enabled_for(await get_achievement_mode())


async def achievement_display_enabled() -> bool:
    return display_enabled_for(await get_achievement_mode())


async def achievement_alerts_enabled() -> bool:
    return alerts_enabled_for(await get_achievement_mode())


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
    global _override_cache, _last_known_override, _has_known_override
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
    _last_known_override = normalized
    _has_known_override = True
    logger.info(
        "achievement mode updated",
        extra={"event": "achievement_mode_updated", "mode": normalized},
    )
    return await get_achievement_settings_snapshot(use_cache=True)
