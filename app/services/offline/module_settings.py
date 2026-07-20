"""Runtime master switches for the offline real-world modules.

The flags live on the ``SystemConfig`` singleton (id=1) so admins can toggle
them from the web console with immediate effect. Reads hit the DB directly —
the call sites are low frequency (the hourly trigger scan plus the manual/admin
endpoints), so a direct lookup keeps values truly real-time and consistent
across worker processes without any cache-invalidation dance.

Both flags default to ``False``: a client (e.g. the H5 page) may ship before its
activity / gift card UI exists, so the modules stay paused until an admin
explicitly enables them.
"""

from __future__ import annotations

import logging

from app.db import db

logger = logging.getLogger(__name__)

# Fallbacks used only when the singleton row is missing (fresh DB before the
# migration seed row is present). Aligned with the column defaults.
_DEFAULT_ACTIVITY_ENABLED = False
_DEFAULT_GIFT_ENABLED = False


async def get_offline_module_flags() -> dict[str, bool]:
    """Return the current offline module switches as a plain dict."""

    config = await db.systemconfig.find_unique(where={"id": 1})
    if not config:
        return {
            "activity_enabled": _DEFAULT_ACTIVITY_ENABLED,
            "gift_enabled": _DEFAULT_GIFT_ENABLED,
        }
    return {
        "activity_enabled": bool(
            getattr(config, "offlineActivityEnabled", _DEFAULT_ACTIVITY_ENABLED)
        ),
        "gift_enabled": bool(
            getattr(config, "offlineGiftEnabled", _DEFAULT_GIFT_ENABLED)
        ),
    }


async def is_activity_enabled() -> bool:
    return (await get_offline_module_flags())["activity_enabled"]


async def is_gift_enabled() -> bool:
    return (await get_offline_module_flags())["gift_enabled"]


async def set_offline_module_flags(
    *,
    activity_enabled: bool | None = None,
    gift_enabled: bool | None = None,
) -> dict[str, bool]:
    """Persist the given switches (``None`` leaves that flag untouched)."""

    create: dict = {"id": 1}
    update: dict = {}
    if activity_enabled is not None:
        create["offlineActivityEnabled"] = activity_enabled
        update["offlineActivityEnabled"] = activity_enabled
    if gift_enabled is not None:
        create["offlineGiftEnabled"] = gift_enabled
        update["offlineGiftEnabled"] = gift_enabled
    if update:
        await db.systemconfig.upsert(
            where={"id": 1},
            data={"create": create, "update": update},
        )
        logger.info(
            "offline module flags updated",
            extra={
                "event": "offline_module_flags_updated",
                "activity_enabled": activity_enabled,
                "gift_enabled": gift_enabled,
            },
        )
    return await get_offline_module_flags()
