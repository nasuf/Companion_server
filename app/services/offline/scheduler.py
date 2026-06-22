from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from app.services.offline import repository as repo
from app.services.offline.activity_service import create_recommendation_for_user
from app.services.offline.gift_service import create_gift_for_user

logger = logging.getLogger(__name__)


async def scan_offline_triggers() -> dict[str, int]:
    """Scan conservative real-world interaction triggers.

    The first release avoids noisy recommendation loops: if a user already has
    an active activity or gift, the scanner leaves them alone.
    """

    now = datetime.now(UTC)
    stats = {"activities": 0, "gifts": 0, "skipped": 0, "failed": 0}
    for ctx in await repo.list_real_world_contexts(limit=500):
        try:
            user_id = ctx["user_id"]
            workspace_id = ctx["workspace_id"]
            agent_id = ctx["agent_id"]
            await repo.ensure_trigger_state(user_id, agent_id, workspace_id)
            created_at = _aware(ctx.get("user_created_at")) or now
            day = max(1, (now.date() - created_at.date()).days + 1)

            if await _should_create_activity(user_id, workspace_id, ctx, day, now):
                if await create_recommendation_for_user(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    source="scheduled",
                ):
                    stats["activities"] += 1

            if await _should_create_gift(user_id, workspace_id, ctx, day, now):
                if await create_gift_for_user(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    trigger_type="scheduled",
                ):
                    stats["gifts"] += 1
        except Exception as exc:
            stats["failed"] += 1
            logger.warning("[offline] trigger scan failed for ctx=%s: %s", ctx, exc)
    return stats


async def _should_create_activity(
    user_id: str,
    workspace_id: str | None,
    ctx: dict,
    day: int,
    now: datetime,
) -> bool:
    active = [
        a for a in await repo.list_activities(user_id, workspace_id)
        if a["status"] in {"pending", "accepted"}
    ]
    if active:
        return False
    last = _aware(ctx.get("last_activity_recommendation_at"))
    if day in {4, 9} and (not last or last < now - timedelta(hours=20)):
        return True
    due = _aware(ctx.get("next_activity_recommendation_at"))
    return bool(due and due <= now)


async def _should_create_gift(
    user_id: str,
    workspace_id: str | None,
    ctx: dict,
    day: int,
    now: datetime,
) -> bool:
    active = [
        g for g in await repo.list_gifts(user_id, workspace_id)
        if g["status"] in {"pending_address", "selecting", "ordered", "shipping"}
    ]
    if active:
        return False
    last = _aware(ctx.get("last_gift_paid_at"))
    forced = day in {5, 20}
    if forced:
        return not last or last < now - timedelta(hours=20)
    return bool(last and last < now - timedelta(days=35))


def _aware(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except Exception:
            return None
    return None
